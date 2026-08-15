"""
Goal: 
train META architecture with CEBRA pipeline and data exposure 
"""
import pickle
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from utils.dataset import SpeechDataset, charset, HandwritingDataset, BrainToTextDataset
from utils.load_model_states import save_checkpoint, load_checkpoint
import os
import numpy as np
from cebra_trainer_utils import InfoNCE, get_batch
from tqdm import trange
import time
from edit_distance import SequenceMatcher
from utils.data_10_loader import get_input as get_10_input
from utils.data_loader import get_input
from typing import Tuple, List

def ctc_collate(batch: List[Tuple[torch.Tensor, str, int]]):
    """
    Returns:
      x_pad: (B, T_max, F)
      input_lengths: (B,)
      targets: (sum_L,)
      target_lengths: (B,)
      transcripts: list[str]
      sessions: (B,)
    """
    xs, ys, ds = zip(*batch)

    B = len(xs)
    feat_dim = xs[0].shape[-1]

    input_lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    T_max = int(input_lengths.max().item())

    x_pad = torch.zeros(B, T_max, feat_dim, dtype=torch.float32)
    for i, x in enumerate(xs):
        T = x.shape[0]
        x_pad[i, :T] = x
        x_pad[i, T:] = x[-1:]

    target_seqs = [torch.tensor(charset.text_to_int(y), dtype=torch.long) for y in ys]
    target_lengths = torch.tensor([t.numel() for t in target_seqs], dtype=torch.long)
    targets = torch.cat(target_seqs) if len(target_seqs) else torch.tensor([], dtype=torch.long)
    max_target_len = max(target_lengths) if len(target_lengths) > 0 else 0
    targets_padded = torch.zeros(B, max_target_len, dtype=torch.long)

    offset = 0
    for i, length in enumerate(target_lengths):
        targets_padded[i, :length] = targets[offset:offset + length]
        offset += length


    sessions = torch.tensor(ds, dtype=torch.long)

    return x_pad, targets_padded, input_lengths, target_lengths, sessions


def ctc_collate_nejm(
    batch: List[
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            int,
        ]
    ]
):

    xs, ys, input_lengths, target_lengths, sessions = zip(*batch)

    B = len(xs)
    feat_dim = xs[0].shape[-1]

    input_lengths = torch.stack(input_lengths)
    target_lengths = torch.stack(target_lengths)

    T_max = int(input_lengths.max())

    x_pad = torch.zeros(
        B,
        T_max,
        feat_dim,
        dtype=torch.float32,
    )

    for i, x in enumerate(xs):
        T = x.shape[0]
        x_pad[i, :T] = x

        if T < T_max:
            x_pad[i, T:] = x[-1:]

    max_target_len = int(target_lengths.max())

    targets_padded = torch.zeros(
        B,
        max_target_len,
        dtype=torch.long,
    )

    for i, y in enumerate(ys):
        L = y.shape[0]
        targets_padded[i, :L] = y

    sessions = torch.tensor(
        sessions,
        dtype=torch.int32,
    )

    return (
        x_pad,            # (B,T,F)
        targets_padded,   # (B,L)
        input_lengths,    # (B,)
        target_lengths,   # (B,)
        sessions,         # (B,)
    )

def _padding(batch):
    X, y, X_lens, y_lens, days = zip(*batch)

    B = len(X)
    feat_dim = X[0].shape[-1]
    max_T = max(x.shape[0] for x in X)

    X_padded = torch.zeros(B, max_T, feat_dim, dtype=X[0].dtype)

    for i, x in enumerate(X):
        T = x.shape[0]
        X_padded[i, :T] = x
        if T < max_T:
            X_padded[i, T:] = x[-1:]  

    y_padded = pad_sequence(y, batch_first=True, padding_value=0)

    return (
        X_padded,
        y_padded,
        torch.stack(X_lens),
        torch.stack(y_lens),
        torch.stack(days),
    )


def get_dataset_loaders_speech_nejm(dataset_name, batch_size, gauss_in=False):
    with open(dataset_name, 'rb') as f:
        dataset_pkl = pickle.load(f)

    train_file_set = dataset_pkl['train'][:23]
    val_file_paths = dataset_pkl['test']
    train_ds = BrainToTextDataset(train_file_set, )
    valid_ds = BrainToTextDataset(val_file_paths)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=ctc_collate_nejm,
                              persistent_workers=True)

    test_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=ctc_collate_nejm,
    )
    return train_loader, test_loader, None



def get_dataset_loaders_speech(
        datasetName,
        batchSize,
        gauss_in=False
    ):
    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)

    


    train_ds = SpeechDataset(loadedData["train"], transform=None, gauss=not gauss_in)
    test_ds = SpeechDataset(loadedData["test"], gauss=not gauss_in)

    train_loader = DataLoader(train_ds, batch_size=batchSize, shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=_padding,
                              persistent_workers=True)

    test_loader = DataLoader(
        test_ds,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )

    return train_loader, test_loader, loadedData



def get_dataset_loaders_nlp_10(
        dataset_name, 
        batch_size,
        gauss_in=True
    ):
    final_day = 5
    train_input = get_10_input(dataset_name, norm=True ,train=True, days=range(final_day), gauss=not gauss_in, gauss_sigma=2.0)
    test_input =  get_10_input(dataset_name, norm=True ,train=False, days=range(final_day, 10), gauss=not gauss_in, gauss_sigma=2.0)

    valid_set = HandwritingDataset(test_input)
    train_set = HandwritingDataset(train_input)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=ctc_collate,
                              persistent_workers=True)
    test_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=ctc_collate,
    )
    return train_loader, test_loader, None


def get_dataset_loaders_nlp_21(
        dataset_name, 
        batch_size,
        gauss_in=True
    ):
    train_input = get_input(
        os.path.join(dataset_name, "seed_model_training_data/mat/"),
        norm=True,
        gauss=not gauss_in,
        train=True, 
        gauss_sigma=2.0
    )
    valid_input_1 = get_input(
        os.path.join(dataset_name, "online_evaluation_data/no_recalibration/mat/"),
        norm=True,
        gauss=not gauss_in,
        train=False,
        gauss_sigma=2.0
    )
    valid_input_2 = get_input(
        os.path.join(dataset_name, "online_evaluation_data/recalibration/mat/"),
        norm=True,
        gauss=not gauss_in,
        train=False,
        gauss_sigma=2.0
    )
    valid_input = valid_input_1 + valid_input_2
    valid_set = HandwritingDataset(valid_input)
    train_set = HandwritingDataset(train_input)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=ctc_collate,
                              persistent_workers=True)
    test_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=ctc_collate,
    )
    return train_loader, test_loader, None

def get_dataset_loaders(
        dataset_name,
        batch_size, 
        gauss_in=True, 
        speech=True,
        nlp_10=False,
        is_nejm=False
    ):
    if speech:
        if is_nejm: return get_dataset_loaders_speech_nejm(dataset_name, batch_size, gauss_in)
        return get_dataset_loaders_speech(dataset_name, batch_size, gauss_in)
    if not nlp_10:
        return get_dataset_loaders_nlp_21(dataset_name, batch_size, gauss_in)
    return get_dataset_loaders_nlp_10(dataset_name, batch_size, gauss_in)



def train_model(args : dict):
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
    do_contrastive = not args.get("no_contrastive", False)

    from meta_model_for_cebra_pipeline import MetaModel
    # model = MetaModel(
    #     num_neurons=192 if not is_speech else (256 if not is_nejm else 512), 
    #     num_classes=41 if is_speech else 32,
    # )
    """
    python start_trainer_cebra.py --datasetPath /mnt/data/hossein/Hossein_workspace/nips_cetra/mahan/CORP/CORP_data_release --offset 4 --out_dir METECEBRA_A --gru --bidir --batchSize 16 --random_offset --hidden 1024 --dropout 0.4 --layers 5 --nBatch 20000 --kernel 32 --stride 4 --seed 5 --do_wandb --no_contrastive --no_noise --cebra_unfolder --ceb_hidden 256 --ceb_out 64

    python start_trainer_cebra.py --datasetPath /data/hossein/mm_project/CORP_data_release --offset 4 --out_dir METECEBRA_A-LargeConv-CebUnfold-4_2-optMeta-Trans --gru --bidir --batchSize 8 --random_offset --hidden 1024 --dropout 0.4 --layers 5  --kernel 32 --stride 4 --seed 5 --do_wandb --no_contrastive --no_noise --cebra_unfolder --ceb_hidden 256 --ceb_out 64 --nBatch 3950

    python start_trainer_cebra.py --datasetPath /data/hossein/mm_project/CORP_data_release --offset 4 --out_dir METECEBRA_A-LargeConv-CebUnfold-4_2-optCEB-Trans --gru --bidir --batchSize 8 --random_offset --hidden 1024 --dropout 0.4 --layers 5  --kernel 32 --stride 4 --seed 5 --do_wandb --no_contrastive --no_noise --cebra_unfolder --ceb_hidden 256 --ceb_out 64 --nBatch 20000

    python start_trainer_cebra.py --datasetPath /data/hossein/mm_project/CORP_data_release --offset 4 --out_dir METECEBRA_A-CebraConv-Unfold_CEBRA_32_4-opt_CEB_diffLength --gru --bidir --batchSize 8 --random_offset --hidden 1024 --dropout 0.4 --layers 5  --kernel 32 --stride 4 --seed 5 --do_wandb --no_contrastive --no_noise --cebra_unfolder --ceb_hidden 256 --ceb_out 64 --nBatch 20000 --optimStyle CEBRA --convStyle CEBRA --unfolding CEBRA_32_4
    """

    if args['unfolding'] == "CEBRA_32_4": 
        if args['convStyle'] == "LARGE": 
            model = MetaModel(192, 32, mahan_model_params=False, time_agg_out="gap", cnn_hidden=2048, cnn_depth=8, cnn_initial_linear=512, cnn_output=64, transformer_depth=4, transformer_head=2, unfolding="CEBRA_32_4")
        else: 
            model = MetaModel(192, 32, mahan_model_params=False, time_agg_out="gap", cnn_hidden=256, cnn_depth=8, cnn_initial_linear=0, cnn_output=64, transformer_depth=2, transformer_head=1, unfolding="CEBRA_32_4") 
    elif args['unfolding'] == "CEBRA_4_4": 
        if args['convStyle'] == "LARGE": 
            model = MetaModel(192, 32, mahan_model_params=False, time_agg_out="gap", cnn_hidden=2048, cnn_depth=8, cnn_initial_linear=512, cnn_output=512, transformer_depth=4, transformer_head=2, unfolding="CEBRA_4_4")

    elif args['unfolding'] == "AVGPOOL_4_4": 
        pass 
        
    # if META_MODEL: 
    #     # 32 4 -> out = 64
    #     # 4  4 -> out = 512
    #     model = MetaModel(192, 32, mahan_model_params=False, time_agg_out="gap", cnn_hidden=2048, cnn_depth=8, cnn_initial_linear=512, cnn_output=64, transformer_depth=4, transformer_head=2, unfolder_kernel= 32, unfolder_stride= 4) 
    # else: 
    #     model = MetaModel(192, 32, mahan_model_params=False, time_agg_out="gap", cnn_hidden=256, cnn_depth=8, cnn_initial_linear=0, cnn_output=64, transformer_depth=2, transformer_head=1, unfolder_kernel= 32, unfolder_stride= 4) 
    print(model)
    # from torchinfo import summary 
    # _temp_B = 64
    # _temp_T_max = 1000
    # _temp_D = 192  # if not is_speech else (256 if not is_nejm else 512), 
    # _temp_x = torch.randn((_temp_B, _temp_T_max, _temp_D))
    # _temp_lengths = torch.randint(0, _temp_T_max, (_temp_B, )) + 1 
    # _temp_y, _, embeddings, embedding_l = model(_temp_x, _temp_lengths)

    # from torchinfo import summary 
    # print(summary(model, input_data=(_temp_x,  _temp_lengths), verbose=1,))

    model = model.to(device)
    # Parallel GPUs gives error for accessing the self.embeddings on model
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    os.makedirs(args["out_dir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    
    criterion = InfoNCE(args['temperature'])
    with open(args["out_dir"] + "/args", "wb") as file:
        pickle.dump(args, file)
    trainLoader, testLoader, loadedData = get_dataset_loaders(
        args["datasetPath"],
        args["batchSize"],
        args.get("gauss_in", True) or no_gauss,
        is_speech,
        args.get("nlp_10", False),
        is_nejm
    )
    ctc_criterion = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    if args["optimStyle"] == "META": 
        # args["nBatch"]
        nBatch = 3950
        from neuraltrain.optimizers import LightningOptimizer
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
            total_steps=args["nBatch"], # epochs * len(train_loader),
        )
        optimizer = optimizer_assets["optimizer"]
        scheduler = optimizer_assets["lr_scheduler"]["scheduler"] 
    elif args["optimStyle"] == "CEBRA": 
        # args["nBatch"]
        nBatch = 20000
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
            total_iters=nBatch # args["nBatch"],
        )
    so_far_batch = load_checkpoint(f'{checkpoint_address}_dont_load_shit',model, optimizer, scheduler)
    print(so_far_batch)
    inf_losses = 0
    
    testLoss = []
    testCER = []

    curr_train_loss = 0.0 
    curr_train_ctc_loss = 0.0 
    curr_train_contrastive_loss = 0.0 
    curr_train_count = 0 
    dummy_epoch = 0 

    train_iter = iter(trainLoader)
    print('len(trainLoader)', len(trainLoader))
    for batch in trange(nBatch): # args["nBatch"]
        
        model.train()
        try:
            X, y, X_len, y_len, dayIdx = next(train_iter)
        except StopIteration:
            train_iter = iter(trainLoader)
            X, y, X_len, y_len, dayIdx = next(train_iter)

        X, y, X_len, y_len, dayIdx = (
            X.to(device),
            y.to(device),
            X_len.to(device),
            y_len.to(device),
            dayIdx.to(device),
        )
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

            # Input:  (B, T, F)
            # Output: logits for CTC of shape (T, B, C)
            pred, lengths, embeddings, emb_lengths = model(X, X_len)
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
            inf_losses += 1
            if inf_losses > 10:
                break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
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
        log_freq = 79 if args['optimStyle'] == "META" else 50 # trainloader lengths --> TODO: read from trainloader so its compatible with other datasets 
        if batch % log_freq == 0: 
            with torch.no_grad():
                model.eval()
                allLoss = []
                total_edit_distance = 0
                total_seq_length = 0
                for X, y, X_len, y_len, testDayIdx in testLoader:

                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                        X, y, X_len, y_len, testDayIdx = (
                            X.to(device),
                            y.to(device),
                            X_len.to(device),
                            y_len.to(device),
                            testDayIdx.to(device),
                        )
                        pred, lengths, _, _ = model(X, X_len)
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