import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from edit_distance import SequenceMatcher
from data_10_loader import get_input as get_10_input
from data_loader import get_input
from typing import Tuple, List
import pickle
import os
CHUNK_SIZE = 4

import torch
from torch.utils.data import Dataset
from typing import List, Tuple
# from utils.augmentation import GaussianSmoothing

# class SpeechDataset(Dataset):
#     def __init__(self, data, transform=None, gauss=False,):
#         if gauss:
#             smoother = GaussianSmoothing(256, 20, 2.0, dim=1)
#         self.data = data
#         self.transform = transform
#         self.n_days = len(data)
#         self.n_trials = sum(len(d["sentenceDat"]) for d in data)

#         self.neural_feats = []
#         self.phone_seqs = []
#         self.neural_time_bins = []
#         self.phone_seq_lens = []
#         self.days = []

#         for day in trange(self.n_days):
#             day_data = data[day]
#             n_trials = len(day_data["sentenceDat"])
#             for trial in range(n_trials):
#                 neural = day_data["sentenceDat"][trial]
#                 neural = torch.as_tensor(neural, dtype=torch.float32)
#                 phones = day_data["phonemes"][trial]
#                 if gauss:
#                     with torch.no_grad():
#                         neural = smoother(neural.unsqueeze(0)).squeeze(0)

#                 self.neural_feats.append(neural)
#                 self.phone_seqs.append(torch.as_tensor(phones, dtype=torch.int32))
#                 self.neural_time_bins.append(neural.shape[0])
#                 self.phone_seq_lens.append(data[day]["phoneLens"][trial])
#                 self.days.append(day)

#     def __len__(self):
#         return self.n_trials

#     def __getitem__(self, idx):
#         neural_feats = self.neural_feats[idx]
#         phone_seq = self.phone_seqs[idx]
#         neural_time_bin = self.neural_time_bins[idx]
#         phone_seq_len = self.phone_seq_lens[idx]
#         day = self.days[idx]

#         if self.transform:
#             neural_feats = self.transform(neural_feats)

#         return (
#             neural_feats,                               # already float32
#             phone_seq,                                  # already int32
#             torch.tensor(neural_time_bin, dtype=torch.int32),
#             torch.tensor(phone_seq_len, dtype=torch.int32),
#             torch.tensor(day, dtype=torch.int64),
#         )


CHARS = [
    '>', ',', '?', '~', "'",
    'a', 'b', 'c', 'd', 'e', 'f', 'g',
    'h', 'i', 'j', 'k', 'l', 'm', 'n',
    'o', 'p', 'q', 'r', 's', 't',
    'u', 'v', 'w', 'x', 'y', 'z',
]
BLANK_TOKEN = "<BLANK>"


class Charset:
    def __init__(self, symbols: List[str]):
        # index 0 reserved for CTC blank
        self.idx2sym = [BLANK_TOKEN] + symbols
        self.sym2idx = {s: i + 1 for i, s in enumerate(symbols)}
        self.sym2idx[BLANK_TOKEN] = 0

    @property
    def num_classes(self) -> int:
        return len(self.idx2sym)

    def text_to_int(self, text: str) -> List[int]:
        return [self.sym2idx[ch] for ch in text if ch in self.sym2idx]

    def int_to_text(self, ids: List[int]) -> str:
        return "".join(self.idx2sym[i] for i in ids if i != 0)


charset = Charset(CHARS)


class HandwritingDataset(Dataset):
    """
    items: List of (features, transcript, session_id)
    """
    def __init__(self, items: List[Tuple[torch.FloatTensor, str, int]], gauss=False):
        super().__init__()
        self.items = items
        if gauss:
            smoother = GaussianSmoothing(192, 20, 2.0, dim=1)
            for i in range(len(items)):
                x, y, z = self.items[i]
                x = smoother(x.unsqueeze(0)).squeeze(0)
                self.items[i] = (x, y, z)

        self.charset = charset


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        x, y, d = self.items[idx]

        assert isinstance(x, torch.Tensor) and x.dtype == torch.float32 and x.dim() == 2
        assert isinstance(y, str)
        assert isinstance(d, int)

        return x, y, d



class DummyHandwritingDataset(Dataset):
    """
    items: List of (features, transcript, session_id)
    """
    def __init__(self, ):
        super().__init__()
        # items: List[Tuple[torch.FloatTensor, str, int]], 

        # self.items = items
        # if gauss:
        #     smoother = GaussianSmoothing(192, 20, 2.0, dim=1)
        #     for i in range(len(items)):
        #         x, y, z = self.items[i]
        #         x = smoother(x.unsqueeze(0)).squeeze(0)
        #         self.items[i] = (x, y, z)

        self.charset = charset


    def __len__(self):
        return 100

    def __getitem__(self, idx):
        # x, y, d = self.items[idx]
        a = int(10*torch.randn(1))
        if a % 2 == 0: 
            x = torch.randn((1000, 192))
            y = "stop>smoking"
            d = 1
            # torch.Size([1229, 192]) stop>smoking 1
        else: 
            x = torch.randn((2500, 192))
            y = "keep>smoking>till>you>die"
            d = 2
            # torch.Size([1229, 192]) stop>smoking 1
            
        assert isinstance(x, torch.Tensor) and x.dtype == torch.float32 and x.dim() == 2
        assert isinstance(y, str)
        assert isinstance(d, int)

        return x, y, d


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



# def get_dataset_loaders_speech_nejm(dataset_name, batch_size, gauss_in=False):
#     with open(dataset_name, 'rb') as f:
#         dataset_pkl = pickle.load(f)

#     train_file_set = dataset_pkl['train'][:23]
#     val_file_paths = dataset_pkl['test']
#     train_ds = BrainToTextDataset(train_file_set,  gauss=not gauss_in)
#     valid_ds = BrainToTextDataset(val_file_paths, gauss=not gauss_in)
#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
#                               num_workers=4, pin_memory=True, collate_fn=ctc_collate_nejm,
#                               persistent_workers=True)

#     test_loader = DataLoader(
#         valid_ds,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=0,
#         pin_memory=True,
#         collate_fn=ctc_collate_nejm,
#     )
#     return train_loader, test_loader, None


# def get_dataset_loaders_speech(
#         datasetName,
#         batchSize,
#         gauss_in=False
# ):
#     with open(datasetName, "rb") as handle:
#         loadedData = pickle.load(handle)

#     train_ds = SpeechDataset(loadedData["train"], transform=None, gauss=not gauss_in)
#     test_ds = SpeechDataset(loadedData["test"], gauss=not gauss_in)

#     train_loader = DataLoader(train_ds, batch_size=batchSize, shuffle=True,
#                               num_workers=4, pin_memory=True, collate_fn=_padding,
#                               persistent_workers=True)

#     test_loader = DataLoader(
#         test_ds,
#         batch_size=batchSize,
#         shuffle=False,
#         num_workers=0,
#         pin_memory=True,
#         collate_fn=_padding,
#     )

#     return train_loader, test_loader, loadedData


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

def get_dummy_loaders(
        dataset_name,
        batch_size,
        gauss_in=True
):
    train_set = DummyHandwritingDataset()

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              pin_memory=False, collate_fn=ctc_collate,
                              persistent_workers=False)
    
    return train_loader, train_loader, None



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


