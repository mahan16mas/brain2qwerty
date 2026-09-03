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
            dataset = _WholeTrialTensorDataset(neuro=self.neuro, segments=segments)
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


def main():
    print(NUM_CLASSES_WITH_BLANK)
    # debug_config() -> single timeline, fast iteration/smoke tests
    # experiment_config() -> full dataset
    cfg = experiment_config() # debug_config()

    train_loader, test_loader = build_wholetrial_dataloaders(cfg["data"])

    for neuro, neuro_len, target, target_len, meta in train_loader:
        print("neuro:", neuro.shape)        # (B, n_channels, T_max)
        print("neuro_len:", neuro_len)       # (B,)
        print("target:", target.shape)       # (B, L_max)
        print("target_len:", target_len)     # (B,)
        # print("meta:", meta)                 # (B, 3) -> subject_id, session, trial_id
        print("first sentence:", decode_target(target[0, : target_len[0]]))
        # print("second sentence:", decode_target(target[1, : target_len[1]]))
        # print("thids sentence:", decode_target(target[2, : target_len[2]]))
        # print(target[0])
        # print(target_len[0])

        # print(target[1])
        # print(target_len[1])
                
        # break
        print()

if __name__ == "__main__":
    main()