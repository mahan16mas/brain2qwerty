# Whole-trial data loading for brain2qwerty_v1 -- built on the REAL pipeline
# (Study, EventsTransforms, MegExtractor all imported from neuralset / this
# repo, not reimplemented). The only new logic is:
#   1. segmenting on whole Sentence trials instead of Keystroke windows
#   2. collating into plain (neuro, target, meta) tensor tuples instead of
#      neuralset's Batch object
#
# Drop this file into brain2qwerty_v1/ (next to main.py) and import
# `build_wholetrial_dataloaders` from your new main.py.

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

import neuralset as ns

from .main import Data  # the real pydantic Data model: study/transforms/neuro
from .utils import BUTTON_MAPPING, CHAR_INDEX

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
    """One item = one full trial (Sentence onset -> end). Returns raw,
    unbatched (neuro, target, meta) tensors; padding happens in collate_fn."""

    def __init__(self, neuro: ns.extractors.BaseExtractor, segments: list):
        self.neuro = neuro
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
        neuro_len = torch.tensor(neuro_t.shape[1], dtype=torch.long)  # true T, pre-padding

        target_t = encode_sentence(extra["sentence_typed"])  # (L,)
        target_len = torch.tensor(target_t.shape[0], dtype=torch.long)  # true L, pre-padding

        meta_t = torch.tensor(
            [int(extra["subject"]), int(extra["session"]), int(extra["trial_id"])],
            dtype=torch.long,
        )  # (3,) = [subject_id, session, trial_id]

        return neuro_t, neuro_len, target_t, target_len, meta_t


def _collate_whole_trial(batch):
    """Pads variable-length neuro/target across the batch.

    Returns exactly 5 outputs:
      neuro       : (B, n_channels, T_max)  zero-padded
      neuro_len   : (B,)                    true (pre-padding) T per item
      target      : (B, L_max)              padded with 0 (0 = CTC blank / padding, never a real class)
      target_len  : (B,)                    true (pre-padding) L per item
      meta        : (B, 3)                  [subject_id, session, trial_id]

    Lengths are captured in __getitem__ before any padding happens, so both
    neuro_len and target_len are exact -- not inferred from the padding value
    (which would be unreliable for neuro, since a genuine all-zero timestep is
    possible after RobustScaler + clamping).
    """
    neuros, neuro_lens, targets, target_lens, metas = zip(*batch)

    neuro_padded = pad_sequence(
        [n.T for n in neuros], batch_first=True, padding_value=0.0
    ).transpose(1, 2)  # (B, C, T_max)
    neuro_len_batched = torch.stack(neuro_lens, dim=0)  # (B,)

    target_padded = pad_sequence(targets, batch_first=True, padding_value=0)  # (B, L_max); 0 = pad
    target_len_batched = torch.stack(target_lens, dim=0)  # (B,)

    meta_batched = torch.stack(metas, dim=0)  # (B, 3)

    return neuro_padded, neuro_len_batched, target_padded, target_len_batched, meta_batched


class WholeTrialData(Data):
    """Same Study / transforms / neuro config as the real `Data` model --
    only the segmenting + dataset/collate differ. `feature`/`start`/`duration`
    from the original config are still accepted (Data requires `feature`) but
    unused here, since labels now come from `sentence_typed`, not the
    per-keystroke LabelEncoder."""

    def build_loaders(self) -> dict[str, DataLoader]:
        events = self.build_events()          # real Study.run() + real EventsTransforms
        self.neuro.prepare(events)             # real MegExtractor.prepare()

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

            # Extract each trial once, so we can check its true time length.
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

            dataset = _WholeTrialTensorDataset(neuro=self.neuro, segments=valid_segments)
            loaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
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

import pickle
import torch
import torch.nn as nn

import torch
import os
import numpy as np
from tqdm import trange

import torch
from torch import nn
from typing import *
import time
from edit_distance import SequenceMatcher
from typing import Tuple, List


@torch.jit.script
def dot_similarity(ref: torch.Tensor, pos: torch.Tensor,
                   neg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cosine similarity the ref, pos and negative pairs

    Args:
        ref: The reference samples of shape `(n, d)`.
        pos: The positive samples of shape `(n, d)`.
        neg: The negative samples of shape `(n, d)`.

    Returns:
        The similarity between reference samples and positive samples of shape `(n,)`, and
        the similarities between reference samples and negative samples of shape `(n, n)`.
    """
    pos_dist = torch.einsum("ni,ni->n", ref, pos)
    neg_dist = torch.einsum("ni,mi->nm", ref, neg)
    return pos_dist, neg_dist


@torch.jit.script
def infonce(
        pos_dist: torch.Tensor, neg_dist: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """InfoNCE implementation

    See :py:class:`BaseInfoNCE` for reference.

    Note:
        - The behavior of this function changed beginning in CEBRA 0.3.0.
        The InfoNCE implementation is numerically stabilized.
    """
    with torch.no_grad():
        c, _ = neg_dist.max(dim=1, keepdim=True)
    c = c.detach()

    pos_dist = pos_dist - c.squeeze(1)
    neg_dist = neg_dist - c
    align = (-pos_dist).mean()
    uniform = torch.logsumexp(neg_dist, dim=1).mean()

    c_mean = c.mean()
    align_corrected = align - c_mean
    uniform_corrected = uniform + c_mean

    return align + uniform, align_corrected, uniform_corrected



class InfoNCE(nn.Module):
    r"""Cosine similarity function with fixed temperature.

    The similarity metric is given as

    .. math ::

        \phi(x, y) =  x^\top y  / \tau

    with fixed temperature :math:`\tau > 0`.

    Note that this loss function should typically only be used with normalized.
    This class itself does *not* perform any checks. Ensure that :math:`x` and
    :math:`y` are normalized.
    """

    def __init__(self, temp) -> None:
        super().__init__()
        self.temperature = temp

    @torch.jit.export
    def _distance(self, ref: torch.Tensor, pos: torch.Tensor,
                  neg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pos_dist, neg_dist = dot_similarity(ref, pos, neg)
        return pos_dist / self.temperature, neg_dist / self.temperature

    def forward(self, ref, pos,
                neg) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the InfoNCE loss.

        Args:
            ref: The reference samples of shape `(n, d)`.
            pos: The positive samples of shape `(n, d)`.
            neg: The negative samples of shape `(n, d)`.

        See Also:
            :py:class:`BaseInfoNCE`.
        """
        pos_dist, neg_dist = self._distance(ref, pos, neg)
        return infonce(pos_dist, neg_dist)




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

"""
python start_cebra.py --out_dir small_cebra_default --no_noise --bidir --cebra_unfolder --gru --offset 4 --batchSize 64  --random_offset --hidden 1024 --dropout 0.4 --layers 5 --nBatch 20000 --kernel 25 --stride 4 --seed 5 --do_wandb --gradClipValue 10.0

--whiteNoiseSD 1.2 --constantOffsetSD 0.6 

"""
def train_model(args):
    print(NUM_CLASSES_WITH_BLANK)
    # debug_config() -> single timeline, fast iteration/smoke tests
    # experiment_config() -> full dataset
    cfg = experiment_config() # debug_config()

    trainLoader, testLoader = build_wholetrial_dataloaders(cfg["data"])

    batch_index = 0
    data_count = 0
    neural_data_len_mean = 0
    target_data_len_mean = 0
    for neuro, neuro_len, target, target_len, meta in trainLoader:
        print("neuro:", neuro.shape)        # (B, n_channels, T_max)
        print("neuro_len:", neuro_len)       # (B,)
    #     print("target:", target.shape)       # (B, L_max)
    #     print("target_len:", target_len)     # (B,)
    #     # print("meta:", meta)                 # (B, 3) -> subject_id, session, trial_id
    #     print("first sentence:", decode_target(target[0, : target_len[0]]))
    #     # print("second sentence:", decode_target(target[1, : target_len[1]]))
    #     # print("thids sentence:", decode_target(target[2, : target_len[2]]))
    #     # print(target[0])
    #     # print(target_len[0])

    #     # print(target[1])
    #     # print(target_len[1])
                
    #     # break
    #     print()
    #     batch_index += 1
    #     data_count += neuro_len.shape[0]
    #     neural_data_len_mean += neuro_len.sum().item()
    #     target_data_len_mean += target_len.sum().item()

    # neural_data_len_mean /= data_count
    # target_data_len_mean /= data_count
    # print(f'In {data_count} number of samples, mean of neural data length: {neural_data_len_mean} and mean of target length: {target_data_len_mean}')
    # # TRAIN, In 4104 number of samples, mean of neural data length: 286.72002923976606 and mean of target length: 38.47490253411306
    # # TEST, In 521 number of samples, mean of neural data length: 286.01151631477927 and mean of target length: 39.31861804222649

    do_wandb = args.get("do_wandb", False)
    if do_wandb:
        import wandb
        exp_name = args["out_dir"]
        wandb.init(project="NeuroNLP", name=f'{exp_name}')

    no_gauss = args.get("no_gauss", False)
    device = "cuda"
    checkpoint_address = args["out_dir"] + "/checkpoint.pt"
    is_speech = args.get("is_speech", True)
    adv_norm = args.get('adv_norm', 'linf')
    sample_single = args.get("sample_single", False)
    all_ref = args.get("all_ref", False)
    random_dir = args.get("random_dir", False)
    random_offset = args.get("random_offset", False)
    no_noise = args.get('no_noise', False)
    adv = args.get('adv', False)
    adv_eps = args.get('adv_eps', 0.01)
    no_rnn = args.get("no_rnn", False)
    is_nejm = args.get("is_nejm", False)
    alpha = float(args.get("alpha", 1.0))
    do_contrastive = False # not args.get("no_contrastive", False)
    hamed_cebra_model = args.get('use_hamed_cebra_model', False)
    ode_mode = args.get('ode_mode', 'None')
    inner = args.get('inner', 'None')
    dataset_type = args.get('dataset')

    from .cebra_model import Encoder_Decoder
    
    model_input_dim = 306 
    num_classes = NUM_CLASSES_WITH_BLANK

    model = Encoder_Decoder(
        model_input_dim, 
        args['ceb_out'],
        args['kernel'],
        args['stride'],
        num_classes,
        args['hidden'],
        args['layers'],
        args['dropout'],
        args['bidir'],
        args['cebra_unfolder'],
        args['gru'],
        2.0,
        gauss_in=args.get("gauss_in", True), #  and not no_gauss,
        no_rnn=no_rnn,
        cebra_bn=args.get("ceb_bn", False),
        cebra_window_10=args.get("cebra_window_10", False),
        contrastive_on_decoder=args.get("contrastive_on_decoder", False), 
        ceb_hidden=args.get('ceb_hidden', 256),
        initial_layer_size = args.get('initial_layer_size', 0)
    )
        
    print(model)

    # from torchinfo import summary 
    # _temp_B = 64
    # _temp_T_max = 1000
    # _temp_D = model_input_dim  # if not is_speech else (256 if not is_nejm else 512), 
    # _temp_x = torch.randn((_temp_B, _temp_T_max, _temp_D))
    # _temp_lengths = torch.randint(0, _temp_T_max, (_temp_B, )) + 1 
    # _temp_y, _, embeddings, embedding_l = model(_temp_x, _temp_lengths)
    # print(summary(model, input_data=(_temp_x,  _temp_lengths), verbose=1,))
    # exit()

    model = model.to(device)
    # Parallel GPUs gives error for accessing the self.embeddings on model
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs")
    #     model = torch.nn.DataParallel(model)
    os.makedirs(args["out_dir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    
    criterion = InfoNCE(args['temperature'])
    with open(args["out_dir"] + "/args", "wb") as file:
        pickle.dump(args, file)


    ctc_criterion = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args["lrStart"],
        betas=(0.9, 0.999),
        eps=0.1,
        weight_decay=args["l2_decay"],
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=args["lrEnd"] / args["lrStart"],
        total_iters=args["nBatch"],
    )
    # uncomment later 
    ##### so_far_batch = load_checkpoint(checkpoint_address,model, optimizer, scheduler)
    so_far_batch = -1
    print(so_far_batch)
    inf_losses = 0
    
    testLoss = []
    testCER = []

    curr_train_loss = 0.0 
    curr_train_ctc_loss = 0.0 
    curr_train_contrastive_loss = 0.0 
    curr_train_count = 0 
    dummy_epoch = 0 

    batch_index = -1
    train_iter = iter(trainLoader)
    for batch in trange(args["nBatch"]):
        print('-'*15, ' [DEBUGGING] ', batch_index , '  -', '-'*14)
        """
        ---------------  [DEBUGGING]  ---------------                                                                                               
        X, X_len, y, y_len torch.Size([64, 657, 306]) tensor([350, 213, 301, 451, 238, 262, 236, 343, 294, 215, 266, 359, 274, 294,                 
                115, 254, 358, 193, 209, 269, 274, 217, 331, 237, 339, 200, 188, 240,                                                               
                275, 279, 308, 240, 220, 275, 403, 210, 203, 345, 289, 303, 426, 390,
                657, 343,   7, 511, 292, 337, 265, 361, 197, 327, 270, 429, 297, 259,
                371, 142, 290, 288, 166, 255, 184, 272], device='cuda:0') torch.Size([64, 65]) tensor([65, 42, 37, 39, 32, 39, 36, 35, 40, 38, 34, 48, 47, 48, 14, 44, 48, 34,
                42, 39, 46, 35, 57, 36, 43, 35, 32, 43, 47, 44, 48, 44, 46, 41, 51, 40,
                34, 44, 49, 42, 57, 35, 57, 42,  2, 55, 32, 43, 26, 43, 39, 41, 42, 54,
                46, 40, 37, 23, 43, 45, 27, 46, 25, 41], device='cuda:0')
        pred, lengths torch.Size([64, 159, 30]) tensor([ 81,  47,  69, 106,  53,  59,  52,  79,  67,  47,  60,  83,  62,  67,
                22,  57,  83,  42,  46,  61,  62,  48,  76,  53,  78,  43,  40,  53,
                62,  63,  70,  53,  48,  62,  94,  46,  44,  80,  66,  69, 100,  91,
                158,  79,  -4, 121,  66,  78,  60,  84,  43,  75,  61, 101,  68,  58,
                86,  29,  66,  65,  35,  57,  39,  61], device='cuda:0',
            dtype=torch.int32)

        """
        batch_index += 1 
        model.train()
        try:
            X, X_len, y, y_len, meta = next(train_iter)
        except StopIteration:
            train_iter = iter(trainLoader)
            X, X_len, y, y_len, meta = next(train_iter)
        # has_nan = torch.isnan(X).any()
        # has_zero = (X_len == 0).any()
        # print('has_nan', has_nan, 'has_zero', has_zero)
        X, X_len, y, y_len, meta = (
            X.to(device),
            X_len.to(device),
            y.to(device),
            y_len.to(device),
            meta.to(device),
        )
        # X: (B, n_channels, T_max) 
        X = X.permute(0, 2, 1) # (B, T_max, n_channels)
        
        if batch_index == 0: 
            print(X.shape, X_len.shape, y.shape, y_len.shape, y_len[0])
        if batch < so_far_batch:
            continue
        if not no_noise:
            if args["whiteNoiseSD"] > 0:
                X += torch.randn(X.shape, device=device) * args["whiteNoiseSD"]

            if args["constantOffsetSD"] > 0:
                X += (
                        torch.randn([X.shape[0], 1, X.shape[2]], device=device)
                        * args["constantOffsetSD"]
                    )
    
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            # Clean Forward
            # pred, lengths = model(X, X_len)
            # if isinstance(model, torch.nn.DataParallel):
            #     embeddings, emb_lengths = model.module.get_cebra_embs()
            # else:
            #     embeddings, emb_lengths = model.get_cebra_embs()
            print('X, X_len, y, y_len', X.shape, X_len, y.shape, y_len)
            pred, lengths, embeddings, emb_lengths = model(X, X_len)
            print('pred, lengths', pred.shape, lengths)
            if batch_index == 0:
                print(pred.shape, lengths.shape, lengths[0])
            # print('-'*40)
            #########
            # # print(pred)
            # # print(pred.log_softmax(2))
            # x= torch.permute(pred, [1, 0, 2]) # B, T, C
            # # print(x.shape)
            # # 1. B indices where there is ANY NaN anywhere in (T, C)
            # b_any_nan = torch.where(torch.isnan(x).any(dim=(1, 2)))[0]
            # print('b_any_nan', b_any_nan)

            # # 2. B indices where ANY T timestep has its COMPLETE C dimension as NaN
            # #    i.e. for some t: x[b, t, :] is entirely NaN
            # b_any_t_all_c_nan = torch.where(
            #     torch.isnan(x).all(dim=2).any(dim=1)
            # )[0]
            # print('b_any_t_all_c_nan', b_any_t_all_c_nan)

            # # 3. B indices where EVERYTHING in (T, C) is NaN
            # b_all_nan = torch.where(
            #     torch.isnan(x).all(dim=(1, 2))
            # )[0]
            # print('b_all_nan', b_all_nan)
            ##  exit()
            ############## 
            ctc_loss = ctc_criterion(
                torch.permute(pred.log_softmax(2), [1, 0, 2]),
                y,
                lengths,
                y_len,
            )
            ctc_loss = torch.sum(ctc_loss)

            if do_contrastive: 
                reference, positive, negative, ref_batch_idx, ref_time_idx, pos_time_idx, neg_batch_idx, neg_time_idx, = get_batch(embeddings, emb_lengths, args['cont_batch'], args['offset'], sample_single, random_offset, True, all_ref)

                loss_contrastive = criterion(reference, positive, negative)[0]
                loss = alpha * loss_contrastive + ctc_loss
            else: 
                loss = ctc_loss
                loss_contrastive = torch.tensor(0.0)

            # Backpropagation
            optimizer.zero_grad()
        if not torch.isfinite(loss):
            # print('inf loss')
            inf_losses += 1
            if inf_losses > 10:
                break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.get('gradClipValue', 5.0)) # max_norm=5.0
        optimizer.step()

        curr_train_loss += loss.item()
        curr_train_ctc_loss += ctc_loss.item()
        curr_train_contrastive_loss += loss_contrastive.item()
        curr_train_count += 1 

        if adv:
            epsilon = adv_eps
            steps = 10
            alpha = epsilon / 5.0
            
            X_adv = X.detach().clone().to(device)

            if adv_norm == 'linf':
                X_adv = X_adv + torch.empty_like(X_adv).uniform_(-epsilon, epsilon)
            elif adv_norm == 'l2':
                noise = torch.randn_like(X_adv)
                noise_norm = noise.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
                noise_normalized = noise / noise_norm
                noise_normalized *= (torch.rand((noise.shape[0], noise.shape[1], 1), device=noise.device) * epsilon)
                X_adv = X_adv + noise_normalized



            for i in range(steps):
                X_adv = X_adv.detach()
                X_adv.requires_grad_(True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                    # pred_adv, lengths = model(X_adv, X_len)
                    # if isinstance(model, torch.nn.DataParallel):
                    #     embeddings_adv, emb_lengths = model.module.get_cebra_embs()
                    # else:
                    #     embeddings_adv, emb_lengths = model.get_cebra_embs()
                    pred_adv, lengths, embeddings_adv, emb_lengths = model(X_adv, X_len)
                    ctc_loss_adv = ctc_criterion(
                        torch.permute(pred_adv.log_softmax(2), [1, 0, 2]),
                        y,
                        lengths,
                        y_len,
                    )
                    ctc_loss_adv = torch.sum(ctc_loss_adv)
                    reference, positive, negative = embeddings_adv[ref_batch_idx, ref_time_idx], embeddings_adv[ref_batch_idx, pos_time_idx], embeddings_adv[neg_batch_idx, neg_time_idx]
                    negative = negative.detach()
                    positive = positive.detach()
                    loss_contrastive_adv = criterion(reference, positive, negative)[0]
                    loss_adv = loss_contrastive_adv + ctc_loss_adv
                
                grad = torch.autograd.grad(loss_adv, X_adv, only_inputs=True)[0]
                
                with torch.no_grad():
                    if adv_norm == 'linf':
                        X_adv = X_adv + alpha * grad.sign()
                        delta = torch.clamp(X_adv - X, min=-epsilon, max=epsilon)
                        X_adv = X + delta
                    elif adv_norm == 'l2':
                        grad_norm = grad.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
                        grad_normalized = grad / grad_norm
                        X_adv = (X_adv + alpha * grad_normalized).detach()
                        delta = X_adv - X
                        delta_norm = delta.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
                        scale = torch.clamp(epsilon / delta_norm, max=1.0)
                        delta = delta * scale
                        X_adv = (X + delta).detach()

            optimizer.zero_grad()
            X_adv = X_adv.detach()
            X_adv.requires_grad_(False)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                
                # pred_adv, lengths = model(X_adv, X_len)
                # if isinstance(model, torch.nn.DataParallel):
                #     embeddings_adv, emb_lengths = model.module.get_cebra_embs()
                # else:
                #     embeddings_adv, emb_lengths = model.get_cebra_embs()
                pred_adv, lengths, embeddings_adv, emb_lengths = model(X_adv, X_len)
                
                ctc_loss_adv = ctc_criterion(
                        torch.permute(pred_adv.log_softmax(2), [1, 0, 2]),
                        y,
                        lengths,
                        y_len,
                    )
                ctc_loss_adv = torch.sum(ctc_loss_adv)
                reference, positive, negative = embeddings_adv[ref_batch_idx, ref_time_idx], embeddings_adv[ref_batch_idx, pos_time_idx], embeddings_adv[neg_batch_idx, neg_time_idx]
                loss_contrastive_adv = criterion(reference, positive, negative)[0]
                loss_adv = loss_contrastive_adv + ctc_loss_adv
            
            if not torch.isfinite(loss_adv):
                inf_losses += 1
                if inf_losses > 10:
                    break
            loss_adv.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()



        scheduler.step()
        if batch % 50 == 0:
            with torch.no_grad():
                model.eval()
                allLoss = []
                total_edit_distance = 0
                total_seq_length = 0
                test_batch_index = -1
                for X, X_len, y, y_len, meta in testLoader:
                    test_batch_index += 1 

                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                        X, X_len, y, y_len, meta = (
                            X.to(device),
                            X_len.to(device),
                            y.to(device),
                            y_len.to(device),
                            meta.to(device),
                        )
                        # X: (B, n_channels, T_max) 
                        X = X.permute(0, 2, 1) # (B, T_max, n_channels)
                        if test_batch_index == 0: 
                            print(X.shape, X_len.shape, y.shape, y_len.shape, y_len[0])

                        pred, lengths, _, _ = model(X, X_len)

                        if test_batch_index == 0: 
                            print(pred.shape, lengths.shape, lengths[0])

                        loss = ctc_criterion(
                            torch.permute(pred.log_softmax(2), [1, 0, 2]),
                            y,
                            lengths,
                            y_len,
                        )
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
                                y[iterIdx][0: y_len[iterIdx]].cpu().detach()
                            )
                            matcher = SequenceMatcher(
                                a=trueSeq.tolist(), b=decodedSeq.tolist()
                            )
                            total_edit_distance += matcher.distance()
                            total_seq_length += len(trueSeq)

                avgDayLoss = np.sum(allLoss) / len(testLoader)
                cer = total_edit_distance / total_seq_length

                endTime = time.time()
                print(
                    f"batch {batch}, ctc loss: {avgDayLoss:>7f}, cer: {cer:>7f}, tr_ctc: {loss:>7f}, tr_cont: {loss_contrastive:>7f}"
                )
                startTime = time.time()

            if True:
                if isinstance(model, torch.nn.DataParallel):
                    torch.save(model.module.state_dict(), args["out_dir"] + "/modelWeights")
                else:
                    torch.save(model.state_dict(), args["out_dir"] + "/modelWeights")
                
                save_checkpoint(checkpoint_address, model, optimizer, scheduler, batch)

            testLoss.append(avgDayLoss)
            testCER.append(cer)

            tStats = {}
            tStats["testLoss"] = np.array(testLoss)
            tStats["testCER"] = np.array(testCER)

            with open(args["out_dir"] + "/trainingStats", "wb") as file:
                pickle.dump(tStats, file)        

            if do_wandb: 
                wandb.log({
                    'epoch': dummy_epoch,
                    'train_loss': curr_train_loss/curr_train_count,
                    'train_ctc_loss': curr_train_ctc_loss/curr_train_count,
                    'train_contrastive_loss': curr_train_contrastive_loss/curr_train_count,
                    'test_loss': avgDayLoss,
                    'test_cer': cer,
                    'lr': optimizer.param_groups[0]["lr"]
                })
            curr_train_loss = 0.0
            curr_train_ctc_loss = 0.0  
            curr_train_contrastive_loss = 0.0 
            curr_train_count = 0 
            dummy_epoch += 1 

