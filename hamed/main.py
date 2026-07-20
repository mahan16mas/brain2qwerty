from hamed_models import MetaModel
from neuraltrain.optimizers import LightningOptimizer
from torch import nn
from tqdm import tqdm, trange
import torch
import os
import numpy as np
import pickle
from edit_distance import SequenceMatcher
from general_utils import save_checkpoint, load_checkpoint
from data_utils import get_dataset_loaders_nlp_21, get_dummy_loaders

def eval_model(model, test_loader, device='cuda'):
    ctc_criterion = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    error_and_lengths = []
    with torch.no_grad():
        model.eval()
        allLoss = []
        total_edit_distance = 0
        total_seq_length = 0
        for neuro_chunks, targets_padded, target_lengths, channel_positions, uids_tensor in tqdm(test_loader):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):

                neuro_chunks = neuro_chunks.to(device)
                targets_padded = targets_padded.to(device)
                target_lengths = target_lengths.to(device)
                channel_positions = channel_positions.to(device)
                uids_tensor = uids_tensor.to(device)
                subject_id = torch.zeros(len(neuro_chunks)).long().to(device)
                pred, lengths = model.forward(neuro_chunks, subject_id, channel_positions, uids_tensor)

                loss = ctc_criterion(
                    torch.permute(pred.log_softmax(2), [1, 0, 2]),
                    targets_padded,
                    lengths,
                    target_lengths,
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
                        targets_padded[iterIdx][0: target_lengths[iterIdx]].cpu().detach()
                    )
                    matcher = SequenceMatcher(
                        a=trueSeq.tolist(), b=decodedSeq.tolist()
                    )
                    distance = matcher.distance()
                    total_edit_distance += distance
                    total_seq_length += len(trueSeq)
                    error_and_lengths.append((distance, len(trueSeq)))

        avgDayLoss = np.sum(allLoss) / len(test_loader)
        cer = total_edit_distance / total_seq_length
        return cer, avgDayLoss, error_and_lengths

def fix_logits(logits):
    logits = torch.roll(torch.Tensor(logits), shifts=-6, dims=-1)
    logits[:, :, [26, 31]] = logits[:, :, [31, 26]]
    logits[:, :, [26, 27, 28, 29, 30]] = logits[:, :, [27, 28, 26, 30, 29]]
    return logits

def model_logits(model, test_loader, device='cuda', nlp=False):
    rnn_outputs = {"logits":[], "logitLengths":[], "trueSeqs":[]}
    with torch.no_grad():
        model.eval()

        for neuro_chunks, targets_padded, target_lengths, channel_positions, uids_tensor in tqdm(test_loader):

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                neuro_chunks = neuro_chunks.to(device)
                targets_padded = targets_padded.to(device)
                target_lengths = target_lengths.to(device)
                channel_positions = channel_positions.to(device)
                uids_tensor = uids_tensor.to(device)
                subject_id = torch.zeros(len(neuro_chunks)).long().to(device)
                pred, lengths = model.forward(neuro_chunks, subject_id, channel_positions, uids_tensor)

            pred = pred.float()
            if nlp:
                pred = fix_logits(pred)
            for iterIdx in range(pred.shape[0]):
                    trueSeq = np.array(targets_padded[iterIdx][0: target_lengths[iterIdx]].cpu().detach())

                    rnn_outputs["logits"].append(pred[iterIdx].cpu().detach().numpy().tolist())
                    rnn_outputs["logitLengths"].append(
                        lengths[iterIdx].cpu().detach().item()
                    )
                    rnn_outputs["trueSeqs"].append(trueSeq.tolist())


    return rnn_outputs

def train_model():
    out_dir='./debug'
    DS_DIR = rf"/mnt/data/hossein/Hossein_workspace/nips_cetra/mahan/CORP/CORP_data_release"
    batch_size = 8 
    epochs = 50 
    conv_dropout=0.5
    dropout_input=0.2
    gauss_in=False 
    seed = 42
    checkpoint_address = f"{out_dir}/checkpoint.pt"
    os.makedirs(out_dir, exist_ok=True)

    # train_loader, test_loader, _ = get_dataset_loaders_nlp_21(DS_DIR, batch_size, gauss_in)
    train_loader, test_loader, _ = get_dummy_loaders(DS_DIR, batch_size, gauss_in)

    torch.manual_seed(seed)
    np.random.seed(seed)
    inf_losses = 0
    device = "cpu" # torch.device("cuda")

    model = MetaModel(
        num_neurons=192,
        num_classes=32,
        conv_dropout=conv_dropout,
        dropout_input=dropout_input,
    ).to(device)
    print(model)
    
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
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
    so_far_batch = 0
    # so_far_batch = load_checkpoint(checkpoint_address, model, optimizer, scheduler)
    testCER, testLoss = [], []
    # epochs = min(epochs, 50)

    for epoch in range(epochs):
        if epoch < so_far_batch: continue
        if inf_losses > 10: break
        epoch_loss = 0
        n_items = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            model.train()
            neuro_chunks, targets_padded, target_lengths, channel_positions, uids_tensor = batch
            neuro_chunks = neuro_chunks.to(device)
            targets_padded = targets_padded.to(device)
            target_lengths = target_lengths.to(device)
            channel_positions = channel_positions.to(device)
            channel_positions = torch.randn_like(channel_positions)
            uids_tensor = uids_tensor.to(device)
            subject_id = torch.zeros(len(neuro_chunks)).long().to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                pred, lengths = model.forward(neuro_chunks, subject_id, channel_positions, uids_tensor)
                ctc_loss = criterion(
                    torch.permute(pred.log_softmax(2), [1, 0, 2]),
                    targets_padded,
                    lengths,
                    target_lengths,
                )
                ctc_loss = torch.sum(ctc_loss)
            epoch_loss += ctc_loss.item()
            n_items += len(targets_padded)
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
                    neuro_chunks, targets_padded, target_lengths, channel_positions, uids_tensor = batch
                    neuro_chunks = neuro_chunks.to(device)
                    targets_padded = targets_padded.to(device)
                    target_lengths = target_lengths.to(device)
                    channel_positions = channel_positions.to(device)
                    uids_tensor = uids_tensor.to(device)
                    subject_id = torch.zeros( len(neuro_chunks)).long().to(device)
                    pred, lengths = model.forward(neuro_chunks, subject_id, channel_positions, uids_tensor)

                    loss = criterion(
                        torch.permute(pred.log_softmax(2), [1, 0, 2]),
                        targets_padded,
                        lengths,
                        target_lengths,
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
                            targets_padded[iterIdx][0: target_lengths[iterIdx]].cpu().detach()
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

        if True:

            torch.save(model.state_dict(), out_dir + "/modelWeights")

            save_checkpoint(checkpoint_address, model, optimizer, scheduler, epoch)
        if epoch % 10 == 0:
            torch.save(model.state_dict(), out_dir + f"/modelWeights_{epoch}")

        testLoss.append(avgDayLoss)
        testCER.append(cer)

        tStats = {}
        tStats["testLoss"] = np.array(testLoss)
        tStats["testCER"] = np.array(testCER)

        with open(out_dir + "/trainingStats", "wb") as file:
            pickle.dump(tStats, file)


if __name__=="__main__":
    train_model()
