import torch
from utils.dataset import charset
from utils.dataset import DummyHandwritingDataset, HandwritingDataset, BrainToTextDataset, SpeechDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from edit_distance import SequenceMatcher
from utils.data_10_loader import get_input as get_10_input
from utils.data_loader import get_input
from typing import Tuple, List
import pickle
import os
CHUNK_SIZE = 4

def ctc_collate(batch: list[tuple[torch.Tensor, str, int]]):
    xs, ys, ds = zip(*batch)


    all_chunks = []
    uids = []

    for uid, x in enumerate(xs):
        T, feat_dim = x.shape
        num_chunks = (T + CHUNK_SIZE - 1) // CHUNK_SIZE

        for c in range(num_chunks):
            start = c * CHUNK_SIZE
            end = start + CHUNK_SIZE

            if end <= T:
                chunk = x[start:end]
            else:
                chunk = x[T - CHUNK_SIZE : T]

            all_chunks.append(chunk)
            uids.append(uid)

    neuro_chunks = torch.stack(all_chunks)

    B_sentences = len(xs)
    target_seqs = [torch.tensor(charset.text_to_int(y), dtype=torch.long) for y in ys]
    target_lengths = torch.tensor([t.numel() for t in target_seqs], dtype=torch.long)
    max_target_len = max(target_lengths) if len(target_lengths) > 0 else 0
    targets_padded = torch.zeros(B_sentences, max_target_len, dtype=torch.long)

    offset = 0
    for i, length in enumerate(target_lengths):
        targets_padded[i, :length] = torch.cat(target_seqs)[
            offset : offset + length
        ]
        offset += length

    uids_tensor = torch.tensor(uids, dtype=torch.long)
    neuro_chunks = neuro_chunks.permute(0, 2, 1)
    channel_positions = torch.zeros(neuro_chunks.shape[0], neuro_chunks.shape[1], 2)
    return neuro_chunks, targets_padded, target_lengths, channel_positions, uids_tensor


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
    target_lengths = torch.stack(target_lengths)

    all_chunks = []
    uids = []

    for uid, x in enumerate(xs):
        T, feat_dim = x.shape
        num_chunks = (T + CHUNK_SIZE - 1) // CHUNK_SIZE

        for c in range(num_chunks):
            start = c * CHUNK_SIZE
            end = start + CHUNK_SIZE

            if end <= T:
                chunk = x[start:end]
            else:
                chunk = x[T - CHUNK_SIZE: T]

            all_chunks.append(chunk)
            uids.append(uid)

    neuro_chunks = torch.stack(all_chunks)
    uids_tensor = torch.tensor(uids, dtype=torch.long)


    max_target_len = int(target_lengths.max())
    neuro_chunks = neuro_chunks.permute(0, 2, 1)
    channel_positions = torch.zeros(neuro_chunks.shape[0], neuro_chunks.shape[1], 2)
    targets_padded = torch.zeros(
        B,
        max_target_len,
        dtype=torch.long,
    )

    for i, y in enumerate(ys):
        L = y.shape[0]
        targets_padded[i, :L] = y



    return neuro_chunks, targets_padded, target_lengths, channel_positions, uids_tensor


def _padding(batch):
    X, y, X_lens, y_lens, days = zip(*batch)

    all_chunks = []
    uids = []

    for uid, x in enumerate(X):
        T, feat_dim = x.shape
        num_chunks = (T + CHUNK_SIZE - 1) // CHUNK_SIZE

        for c in range(num_chunks):
            start = c * CHUNK_SIZE
            end = start + CHUNK_SIZE

            if end <= T:
                chunk = x[start:end]
            else:
                chunk = x[T - CHUNK_SIZE: T]

            all_chunks.append(chunk)
            uids.append(uid)

    neuro_chunks = torch.stack(all_chunks)
    uids_tensor = torch.tensor(uids, dtype=torch.long)

    neuro_chunks = neuro_chunks.permute(0, 2, 1)
    channel_positions = torch.zeros(neuro_chunks.shape[0], neuro_chunks.shape[1], 2)

    y_padded = pad_sequence(y, batch_first=True, padding_value=0)

    return neuro_chunks, y_padded, torch.stack(y_lens), channel_positions, uids_tensor



def get_dataset_loaders_speech_nejm(dataset_name, batch_size, gauss_in=False):
    with open(dataset_name, 'rb') as f:
        dataset_pkl = pickle.load(f)

    train_file_set = dataset_pkl['train'][:23]
    val_file_paths = dataset_pkl['test']
    train_ds = BrainToTextDataset(train_file_set,  gauss=not gauss_in)
    valid_ds = BrainToTextDataset(val_file_paths, gauss=not gauss_in)
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
    train_input = get_10_input(dataset_name, norm=True, train=True, days=range(final_day), gauss=not gauss_in,
                               gauss_sigma=2.0)
    test_input_0 = get_10_input(dataset_name, norm=True, train=False, days=range(final_day), gauss=not gauss_in,
                              gauss_sigma=2.0, valid=True)
    test_input = get_10_input(dataset_name, norm=True, train=False, days=range(final_day, 10), gauss=not gauss_in,
                              gauss_sigma=2.0)
    test_input = test_input_0 + test_input
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


def merge_by_borders(data1, borders1, data2, borders2):
    ends1 = borders1[1:] + [len(data1)]
    ends2 = borders2[1:] + [len(data2)]

    merged = []

    for start1, end1, start2, end2 in zip(
        borders1, ends1, borders2, ends2
    ):
        merged.extend(data1[start1:end1])
        merged.extend(data2[start2:end2])

    return merged

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
    valid_input_0 = get_input(
        os.path.join(dataset_name, "seed_model_training_data/mat/"),
        norm=True,
        gauss=not gauss_in,
        train=False,
        valid=True,
        gauss_sigma=2.0
    )
    valid_input_1, borders_1 = get_input(
        os.path.join(dataset_name, "online_evaluation_data/no_recalibration/mat/"),
        norm=True,
        gauss=not gauss_in,
        train=False,
        gauss_sigma=2.0,
        return_borders=True
    )
    valid_input_2, borders_2 = get_input(
        os.path.join(dataset_name, "online_evaluation_data/recalibration/mat/"),
        norm=True,
        gauss=not gauss_in,
        train=False,
        gauss_sigma=2.0,
        return_borders=True
    )
    valid_input = merge_by_borders(valid_input_1, borders_1, valid_input_2, borders_2)
    assert len(valid_input) == len(valid_input_1) + len(valid_input_2)
    valid_input = valid_input_0 + valid_input
    
    # valid_set = DummyHandwritingDataset()
    # train_set = DummyHandwritingDataset()
    
    valid_set = HandwritingDataset(valid_input)
    train_set = HandwritingDataset(train_input)
    

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              collate_fn=ctc_collate,
                              num_workers=4, pin_memory=True, 
                              persistent_workers=True
                              )
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

