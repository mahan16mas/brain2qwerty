# Whole-trial data loading for brain2qwerty_v1 -- built on the REAL pipeline
# (Study, EventsTransforms, MegExtractor all imported from neuralset / this
# repo, not reimplemented). The only new logic is:
#   1. segmenting on whole Sentence trials instead of Keystroke windows
#   2. chunking each trial into fixed 25-length windows, stride 4 (sliding
#      window over the whole trial, decoupled from keystroke timing -- but
#      each individual chunk is still exactly 25 samples, same as original)
#   3. collating into plain tensor tuples instead of neuralset's Batch object
#
# Drop this file into brain2qwerty_v1/ (next to main.py) and import
# `build_wholetrial_dataloaders` from your new main.py.

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

import neuralset as ns

from .main import Data  # the real pydantic Data model: study/transforms/neuro
from .utils import BUTTON_MAPPING, CHAR_INDEX, ChannelPositions2D

# Chunking, matching the original pipeline's window length (25 timepoints =
# 0.5s @ the neuro extractor's 50Hz) but now applied as a SLIDING window over
# the whole trial instead of one window anchored per keystroke.
CHUNK_LENGTH = 25
CHUNK_STRIDE = 2

# Encoder for sentence_typed strings (NOT a reverse of CHAR_INDEX -- see note).
#
# sentence_typed is built by joining the raw `button` field (already run
# through spanishbcbl.py's _clean_buttons), then only the three TOKEN
# STRINGS "<special>"/"<space>"/"<number>" get replaced with "@"/" "/"9".
# _clean_buttons' own "is this special" detection is narrower than what
# BUTTON_MAPPING covers, so some raw characters (£, ý, ü, û, ¤, ¿, `, -, \x14)
# fall through unchanged and end up LITERALLY in sentence_typed rather than
# as the "<special>" token. BUTTON_MAPPING already has direct entries for
# these leftover raw characters (all -> class 13, same as "<special>"), so we
# use BUTTON_MAPPING's single-character entries directly as the encoder, plus
# the three post-replacement token forms.
#
# +1 SHIFT FOR CTC: the original BUTTON_MAPPING/CHAR_INDEX classes are 0-28
# (NUM_CLASSES=29, used by the original cross-entropy pipeline, no blank
# symbol). For CTC, index 0 is reserved for the blank token, so every real
# character class here is shifted up by 1 -> real classes are 1-29, and 0 is
# unambiguous as padding (no real character can ever encode to 0). Total
# class count for a CTC output layer is therefore NUM_CLASSES + 1 = 30.
_BASE_CHAR_TO_INDEX = {k: v for k, v in BUTTON_MAPPING.items() if len(k) == 1}
_BASE_CHAR_TO_INDEX[" "] = BUTTON_MAPPING["<space>"]    # post-replacement space
_BASE_CHAR_TO_INDEX["@"] = BUTTON_MAPPING["<special>"]  # post-replacement special token
_BASE_CHAR_TO_INDEX["9"] = BUTTON_MAPPING["<number>"]   # post-replacement number token

SENTENCE_TYPED_CHAR_TO_INDEX = {c: idx + 1 for c, idx in _BASE_CHAR_TO_INDEX.items()}  # 1..29
CTC_BLANK = 0
NUM_CLASSES_WITH_BLANK = len(set(BUTTON_MAPPING.values())) + 1  # 29 real classes + blank = 30
SILENT_TOKEN = SENTENCE_TYPED_CHAR_TO_INDEX[" "]  # space, appended to the end of every target

# CHAR_INDEX (index -> display char) is the original 0-28 decode table from
# utils.py; shift it the same way for decoding CTC-indexed targets/predictions.
CTC_CHAR_INDEX = {idx + 1: c for idx, c in CHAR_INDEX.items()}  # 1..29 -> display char

MIN_NEURO_LEN = 40


def encode_sentence(sentence_typed: str) -> torch.Tensor:
    """sentence_typed -> class indices (1..29), with a trailing silent
    (space) token appended -- the CTC convention of marking end-of-utterance
    with a silence/blank-adjacent symbol."""
    indices = [SENTENCE_TYPED_CHAR_TO_INDEX[c] for c in sentence_typed] + [SILENT_TOKEN]
    return torch.tensor(indices, dtype=torch.long)


def decode_target(indices) -> str:
    """Reverse of encode_sentence -- ignores 0 (padding / CTC blank)."""
    return "".join(CTC_CHAR_INDEX[int(i)] for i in indices if int(i) != 0)


class _WholeTrialTensorDataset(Dataset):
    """One item = one full trial (Sentence onset -> end), chunked into fixed
    25-length windows with stride 4 (a sliding window over the whole trial,
    NOT anchored to keystrokes -- but each individual chunk is still exactly
    25 samples, matching the original pipeline's window length). Also
    extracts subject_id / channel_positions via the real extractors, exactly
    as Data.build() does. Returns raw, unbatched tensors; padding happens in
    collate_fn."""

    def __init__(
        self,
        neuro: ns.extractors.BaseExtractor,
        subject_id: ns.extractors.BaseExtractor,
        channel_positions: ChannelPositions2D,
        segments: list,
    ):
        self.neuro = neuro
        self.subject_id = subject_id
        self.channel_positions = channel_positions
        self.segments = segments

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int):
        seg = self.segments[idx]
        extra = seg.trigger.extra

        neuro_arr = self.neuro(
            seg.ns_events, start=seg.start, duration=seg.duration, trigger=seg.trigger
        )
        neuro_t = torch.as_tensor(neuro_arr, dtype=torch.float32)  # (n_channels, T)

        # chunk into (n_chunks, n_channels, CHUNK_LENGTH) via a sliding
        # window, stride CHUNK_STRIDE -- same window length as the original
        # per-keystroke pipeline, but now uniformly spaced across the trial.
        T = neuro_t.shape[1]
        if T < CHUNK_LENGTH:
            # trial shorter than one chunk: zero-pad up to CHUNK_LENGTH so we
            # still get exactly 1 chunk instead of 0. Shouldn't trigger in
            # practice given MIN_NEURO_LEN=40 already filters shorter trials
            # out in build_loaders(), but kept as a safety net.
            neuro_t = torch.nn.functional.pad(neuro_t, (0, CHUNK_LENGTH - T))
            T = CHUNK_LENGTH
        chunks = neuro_t.unfold(dimension=1, size=CHUNK_LENGTH, step=CHUNK_STRIDE)
        chunks = chunks.permute(1, 0, 2).contiguous()  # (n_chunks, n_channels, CHUNK_LENGTH)
        neuro_len = torch.tensor(chunks.shape[0], dtype=torch.long)  # true n_chunks, pre-padding

        target_t = encode_sentence(extra["sentence_typed"])  # (L,)
        target_len = torch.tensor(target_t.shape[0], dtype=torch.long)  # true L, pre-padding

        meta_t = torch.tensor(
            [int(extra["subject"]), int(extra["session"]), int(extra["trial_id"])],
            dtype=torch.long,
        )  # (3,) = [subject_id, session, trial_id]

        subject_id_arr = self.subject_id(
            seg.ns_events, start=seg.start, duration=seg.duration, trigger=seg.trigger
        )
        subject_id_t = torch.as_tensor(subject_id_arr, dtype=torch.long).reshape(-1)[0]

        channel_positions_arr = self.channel_positions(
            seg.ns_events, start=seg.start, duration=seg.duration, trigger=seg.trigger
        )
        channel_positions_t = torch.as_tensor(channel_positions_arr, dtype=torch.float32)  # (n_channels, 2)

        return chunks, neuro_len, target_t, target_len, meta_t, subject_id_t, channel_positions_t


def _collate_whole_trial(batch):
    """Pads variable-length neuro chunks/target across the batch.

    Returns exactly 7 outputs:
      neuro             : (total_chunks_in_batch, n_channels, CHUNK_LENGTH)  flat, no padding
      neuro_len         : (B,)   chunk count per trial -- use to split `neuro` back into
                                  per-trial groups (sums to total_chunks_in_batch), e.g.:
                                      trial_chunks = torch.split(neuro, neuro_len.tolist(), dim=0)
      target            : (B, L_max)  padded with 0 (0 = CTC blank / padding, never a real class)
      target_len        : (B,)   true (pre-padding) L per item
      meta              : (B, 3) [subject_id, session, trial_id]  (raw `subject` field from events)
                                  -- still one row per TRIAL, unchanged
      subject_id        : (total_chunks_in_batch,)  from the real LabelEncoder extractor,
                                  expanded via repeat_interleave(neuro_len) to align with
                                  `neuro`'s flat chunk dimension -- one row per CHUNK now,
                                  matching the original pipeline's per-window repetition
      channel_positions : (total_chunks_in_batch, n_channels, 2)  same expansion as subject_id
                                  -- same sensor layout repeated per chunk, not per trial

    `neuro`, `subject_id`, and `channel_positions` are all now flat along the
    same total_chunks_in_batch dimension -- matching the shape the original
    per-keystroke pipeline hands the conv encoder (one row per window, not
    per trial). `target`/`target_len`/`meta` remain one row per TRIAL (B).
    `neuro_len` is what lets you split the flat tensors back into per-trial
    groups, e.g.:
        trial_chunks = torch.split(neuro, neuro_len.tolist(), dim=0)

    Lengths are captured in __getitem__ before any padding happens, so both
    neuro_len and target_len are exact.
    """
    neuros, neuro_lens, targets, target_lens, metas, subject_ids, channel_positions = zip(*batch)

    neuro_flat = torch.cat(neuros, dim=0)  # (total_chunks_in_batch, n_channels, CHUNK_LENGTH)
    neuro_len_batched = torch.stack(neuro_lens, dim=0)  # (B,) chunk count per trial

    target_padded = pad_sequence(targets, batch_first=True, padding_value=0)  # (B, L_max); 0 = pad
    target_len_batched = torch.stack(target_lens, dim=0)  # (B,)

    meta_batched = torch.stack(metas, dim=0)  # (B, 3)
    subject_id_batched = torch.stack(subject_ids, dim=0)  # (B,)
    channel_positions_batched = torch.stack(channel_positions, dim=0)  # (B, n_channels, 2)

    # Expand subject_id/channel_positions from one-row-per-trial to
    # one-row-per-chunk, aligned with the flat `neuro` batch dimension --
    # same repeated-per-window shape the original pipeline used (every
    # keystroke-window row had its own subject_id/channel_positions row).
    subject_id_expanded = torch.repeat_interleave(subject_id_batched, neuro_len_batched)
    # (total_chunks_in_batch,)
    channel_positions_expanded = torch.repeat_interleave(
        channel_positions_batched, neuro_len_batched, dim=0
    )
    # (total_chunks_in_batch, n_channels, 2)

    return (
        neuro_flat,
        neuro_len_batched,
        target_padded,
        target_len_batched,
        meta_batched,
        subject_id_expanded,
        channel_positions_expanded,
    )


class WholeTrialData(Data):
    """Same Study / transforms / neuro config as the real `Data` model --
    only the segmenting + dataset/collate differ. `feature`/`start`/`duration`
    from the original config are still accepted (Data requires `feature`) but
    unused here, since labels now come from `sentence_typed`, not the
    per-keystroke LabelEncoder."""

    def build_loaders(self) -> dict[str, DataLoader]:
        events = self.build_events()          # real Study.run() + real EventsTransforms
        self.neuro.prepare(events)             # real MegExtractor.prepare()

        # Real extractors, built exactly as Data.build() does -- not
        # reimplemented, just reused from the actual imports.
        subject_id = ns.extractors.LabelEncoder(event_types="Meg", event_field="subject")
        subject_id.prepare(events)
        channel_positions = ChannelPositions2D(neuro=self.neuro)
        channel_positions.prepare(events)

        loaders: dict[str, DataLoader] = {}
        batch_sizes = {"train": self.batch_size, "test": self.test_batch_size}
        for split, batch_size in batch_sizes.items():
            mask = (
                (events.split == split)
                & (events.type == "Sentence")
                & (events.is_percep == False)  # noqa: E712  (production phase only)
            )
            segments = ns.segments.list_segments(events, mask, start=0.0, duration=None)
            if not segments:
                continue

            # Extract each trial once, so we can check its true time length
            # before deciding whether to keep it (unrelated to the later
            # per-chunk extraction that happens again in __getitem__ -- this
            # is a pre-filter over raw T, not the chunked sequence).
            valid_segments = []

            for seg in segments:
                neuro_arr = self.neuro(
                    seg.ns_events,
                    start=seg.start,
                    duration=seg.duration,
                    trigger=seg.trigger,
                )

                neuro_t = torch.as_tensor(neuro_arr, dtype=torch.float32)

                if neuro_t.shape[1] >= MIN_NEURO_LEN:
                    valid_segments.append(seg)

            print(
                f"{split}: keeping {len(valid_segments)}/{len(segments)} "
                f"trials with T >= {MIN_NEURO_LEN}"
            )
            if not valid_segments:
                continue

            dataset = _WholeTrialTensorDataset(
                neuro=self.neuro,
                subject_id=subject_id,
                channel_positions=channel_positions,
                segments=valid_segments,
            )
            loaders[split] = DataLoader(
                dataset,
                batch_size=16, # batch_size,
                shuffle=(split == "train"),
                collate_fn=_collate_whole_trial,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
            )
        return loaders


def build_wholetrial_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    """cfg: the `"data"` sub-dict of experiment_config()/debug_config() from
    brain2qwerty_v1/config/xp_config.py, unmodified (study/neuro config as-is).
    Returns (train_loader, test_loader)."""
    data = WholeTrialData(**cfg)
    loaders = data.build_loaders()
    return loaders["train"], loaders["test"]


# your_new_main.py

import studies  # noqa: F401  -- registers Pinet2024Meg / Pinet2024Eeg

from brain2qwerty_v1.config.xp_config import debug_config, experiment_config
# from brain2qwerty_v1.data_wholetrial import build_wholetrial_dataloaders, decode_target
from tqdm import tqdm
import torch
import os
import numpy as np
import pickle
from edit_distance import SequenceMatcher
from neuraltrain.optimizers import LightningOptimizer

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    step,
    
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
    }
    checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    
    torch.save(checkpoint, path)

def train_model(args):
    do_wandb = args.get("do_wandb", False)
    use_mahan_model_params = args.get("use_mahan_model_params", False)
    if do_wandb:
        import wandb
        exp_name = args["out_dir"]
        wandb.init(project="NeuroNLP", name=f'{exp_name}', config=args)

    print(NUM_CLASSES_WITH_BLANK)
    # debug_config() -> single timeline, fast iteration/smoke tests
    # experiment_config() -> full dataset
    cfg = experiment_config() # debug_config()

    train_loader, test_loader = build_wholetrial_dataloaders(cfg["data"])

    # batch_index = -1
    # data_count = 0
    # neural_data_len_mean = 0
    # target_data_len_mean = 0

    # for neuro, neuro_len, target, target_len, meta, subject_id, channel_positions in train_loader:
    #     batch_index += 1 
    #     print(neuro.shape)
    #     print(neuro_len.shape, neuro_len)
    #     print(target.shape)
    #     print(target_len.shape)
    #     print(subject_id.shape, subject_id)
    #     print(channel_positions.shape)
    #     # embeddings = old_model.conv_encoder(neuro)   # (total_chunks_in_batch, hidden)
    #     # regroup per trial, same role sentence_UID played in the original:
    #     # per_trial_embeddings = torch.split(embeddings, neuro_len.tolist(), dim=0)
    #     # per_trial_embeddings[i] is that trial's ordered (n_chunks_i, hidden) sequence,
    #     # ready for the transformer / CTC head
    #     print()

    #     if batch_index > 5: 
    #         break
    
    checkpoint_address = f"{args['out_dir']}/checkpoint.pt"
    
    epochs = args.get("epochs", 300)
    cnn_hidden = args.get("cnn_hidden", 2048)
    cnn_depth = args.get("cnn_depth", 8)
    cnn_initial_linear = args.get("cnn_initial_linear", 512)
    conv_dropout = args.get("conv_dropout", 0.5)
    dropout_input = args.get("dropout_input", 0.2)
    time_agg_out = args.get("time_agg_out", "att")
    os.makedirs(args["out_dir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    inf_losses = 0
    device = torch.device("cuda")

    model_input = 306
    num_classes = 30
    from .meta_model import MetaModel
    import torch.nn as nn 

    model = MetaModel(
        num_neurons=model_input,
        num_classes=num_classes,
        conv_dropout=conv_dropout,
        dropout_input=dropout_input,
        mahan_model_params = use_mahan_model_params,
        time_agg_out = time_agg_out, 
        cnn_hidden=cnn_hidden,
        cnn_depth=cnn_depth,
        cnn_initial_linear=cnn_initial_linear,
        transformer_depth=args.get("transformer_depth", 4),
        transformer_head=args.get("transformer_head", 2),
    ).to(device)
                
    print(model)
    if False: 
        from torchinfo import summary
        K = 10 # num chunks
        N = 192 if not is_speech else (512 if is_nejm else 256)
        C = chunk_size # chunk size
        x = torch.randn([K, N, C]).to(device)
        sid = torch.zeros([K]).to(device)
        cpos = torch.randn([K, N, C]).to(device)
        uids = torch.concat((torch.zeros([K//2]), torch.ones([K//2]))).to(device)
        # lengths = torch.randint(0, T_max, (B, )) + 1 
        # print(model)
        out, l = model(x, sid, cpos, uids)
        print(out.shape)
    
        summary(model, input_data=(x, sid, cpos, uids), 
                verbose=1,
        )

    # criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    criterion = nn.CTCLoss(blank=CTC_BLANK, zero_infinity=True)
    optimizer_config_dict = {
        "name": "LightningOptimizer",
        "optimizer": {"name": "AdamW", "lr": 5e-5, "kwargs": {"weight_decay": 1e-4}},
        "scheduler": {
            "name": "OneCycleLR",
            "kwargs": {"max_lr": 5e-5, "pct_start": 0.1},
        },
        "interval": "step",
    }

    opt_config = LightningOptimizer.model_validate(optimizer_config_dict)

    optimizer_assets = opt_config.build(
        model.parameters(),
        total_steps=epochs * len(train_loader),
    )
    optimizer = optimizer_assets["optimizer"]
    scheduler = optimizer_assets["lr_scheduler"]["scheduler"]
    # optimizer = torch.optim.Adam(
    #     model.parameters(),
    #     lr=0.02,
    #     betas=(0.9, 0.999),
    #     eps=0.1,
    #     weight_decay=1e-5,
    # )
    # scheduler = torch.optim.lr_scheduler.LinearLR(
    #     optimizer,
    #     start_factor=1.0,
    #     end_factor=0.002 / 0.02,
    #     total_iters=epochs,
    # )
    so_far_batch = 0
    # so_far_batch = load_checkpoint(checkpoint_address, model, optimizer, scheduler)
    testCER, testLoss = [], []
    # epochs = min(epochs, 50)

    for epoch in range(epochs):
        if epoch < so_far_batch: continue
        if inf_losses > 10: break
        epoch_loss = 0
        n_items = 0
        _t_e_m_p = True
        for batch in tqdm(train_loader):
            
            optimizer.zero_grad()
            model.train()
            neuro, neuro_len, target, target_len, meta, subject_id, channel_positions = batch
            neuro = neuro.to(device)
            
            # if do_add_noise: 
            #     if args["whiteNoiseSD"] > 0:
            #         neuro_chunks += torch.randn(neuro_chunks.shape, device=device) * args["whiteNoiseSD"]

            #     if args["constantOffsetSD"] > 0:
            #         neuro_chunks += (
            #             torch.randn([neuro_chunks.shape[0], neuro_chunks.shape[1], 1], device=device)
            #             * args["constantOffsetSD"]
            #         )
            if _t_e_m_p:
                _t_e_m_p = False 
                if epoch == 0: 
                    print('chunk shape', neuro.shape)
            target = target.to(device)
            target_len = target_len.to(device)
            channel_positions = channel_positions.to(device)
            channel_positions = torch.randn_like(channel_positions)
            subject_id = subject_id.to(device)
            channel_positions = channel_positions.to(device)
            
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                pred, lengths = model(neuro, neuro_len, subject_id, channel_positions)
                print(pred.shape) # torch.Size([64, 130, 30]) 
                # print(lengths.shape, lengths) # torch.Size([64]) 
                #  tensor([ 98,  39,  50,  85,  75, 130,  55,  67,  36,  64,  48,  45,  23,  31,
                #  55,  87,  83,  66,  63,  84,  60, 104,  32,  72,  49,  91,  37,  62,
                #  64,  62,  65,  53,  95,  51,  89,  67, 113,  73,  47,  68,  81,  77,
                #  58, 110,  72,  90,  51,  52,  44,  43,  76,  61,  60,  54,  73,  52,
                #  69,  86,  43,  36,  57,  53, 112,  50])
                print(target.shape, target_len)
                # (B, L_max)
                # tensor([39, 34, 43, 50, 45, 47, 34, 58, 29, 46, 45, 34, 23, 28, 40, 57, 51, 48,
                # 47, 35, 30, 47, 23, 32, 42, 47, 35, 40, 43, 43, 43, 35, 43, 41, 45, 36,
                # 37, 43, 39, 36, 37, 57, 46, 49, 39, 43, 29, 39, 31, 34, 42, 43, 27, 44,
                # 38, 30, 55, 46, 28, 36, 43, 40, 44, 39], device='cuda:0')
                log_probs = torch.log_softmax(pred, dim=-1).transpose(0, 1)  # (T, B, C)
                ctc_loss = criterion(log_probs, target, neuro_len, target_len)
                
                ctc_loss = torch.sum(ctc_loss)
            epoch_loss += ctc_loss.item()
            n_items += len(target)
            if not torch.isfinite(ctc_loss):
                inf_losses += 1
                if inf_losses > 10:
                    break
            ctc_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()
        epoch_loss /= n_items
        with torch.no_grad():
            model.eval()
            allLoss = []
            total_edit_distance = 0
            total_seq_length = 0
            for batch in test_loader:

                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                    neuro, neuro_len, target, target_len, meta, subject_id, channel_positions = batch
                    neuro = neuro.to(device)
                    target = target.to(device)
                    target_len = target_len.to(device)
                    channel_positions = channel_positions.to(device)
                    channel_positions = torch.randn_like(channel_positions)
                    subject_id = subject_id.to(device)
                    channel_positions = channel_positions.to(device)
                    
                    pred, lengths = model.forward(neuro, neuro_len, subject_id, channel_positions)
                    log_probs = torch.log_softmax(pred, dim=-1).transpose(0, 1)  # (T, B, C)
                    loss = criterion(log_probs, target, neuro_len, target_len)
                    
                    # loss = criterion(
                    #     torch.permute(pred.log_softmax(2), [1, 0, 2]),
                    #     targets_padded,
                    #     lengths,
                    #     target_lengths,
                    # )
                    loss = torch.sum(loss)
                    allLoss.append(loss.cpu().detach().numpy())

                    for iterIdx in range(pred.shape[0]):
                        decodedSeq = torch.argmax(
                            torch.tensor(pred[iterIdx, 0: lengths[iterIdx], :]),
                            dim=-1,
                        )  # [num_seq,]
                        decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                        decodedSeq = decodedSeq.cpu().detach().numpy()
                        decodedSeq = np.array([i for i in decodedSeq if i != 0])

                        trueSeq = np.array(
                            target[iterIdx][0: target_len[iterIdx]].cpu().detach()
                        )
                        matcher = SequenceMatcher(
                            a=trueSeq.tolist(), b=decodedSeq.tolist()
                        )
                        total_edit_distance += matcher.distance()
                        total_seq_length += len(trueSeq)

            avgDayLoss = np.sum(allLoss) / len(test_loader)
            cer = total_edit_distance / total_seq_length

            print(
                f"epoch {epoch}, ctc loss: {epoch_loss:>7f}, cer: {cer:>7f}"
            )

        if do_wandb: 
            wandb.log({
                'epoch': epoch,
                'train_loss': epoch_loss,
                'test_loss': avgDayLoss,
                'test_cer': cer,
                'lr': optimizer.param_groups[0]["lr"]
            })

        if True:

            # torch.save(model.state_dict(), args["out_dir"] + "/modelWeights")
            save_checkpoint(checkpoint_address, model, optimizer, scheduler, epoch)
        # if epoch % 10 == 0:
        #     torch.save(model.state_dict(), args["out_dir"] + f"/modelWeights_{epoch}")

        testLoss.append(avgDayLoss)
        testCER.append(cer)

        tStats = {}
        tStats["testLoss"] = np.array(testLoss)
        tStats["testCER"] = np.array(testCER)

        with open(args["out_dir"] + "/trainingStats", "wb") as file:
            pickle.dump(tStats, file)



if __name__ == "__main__":
    main()