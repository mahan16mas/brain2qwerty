# # Standalone dataloader inspector — no training, just builds the loaders and
# # prints what's actually inside each batch. Run this once BEFORE changing the
# # segmenter (type=="Keystroke", fixed duration) and once AFTER (type=="Sentence",
# # duration=None) to diff the two.
# #
# # Uses debug_config() (single timeline) so it's fast to iterate on.

# import studies  # noqa: F401  (registers Pinet2024Meg / Pinet2024Eeg)

# from brain2qwerty_v1.config.xp_config import debug_config
# from brain2qwerty_v1.main import Data

# N_BATCHES_TO_PRINT = 3


# def describe_batch(i, batch):
#     print(f"\n--- batch {i} ---")
#     for name, tensor in batch.data.items():
#         print(f"  data[{name!r}].shape = {tuple(tensor.shape)}  dtype={tensor.dtype}")

#     uids = [seg.trigger.extra.get("sentence_UID") for seg in batch.segments]
#     unique_uids = sorted(set(uids))
#     print(f"  n_segments_in_batch = {len(batch.segments)}")
#     print(f"  n_unique_sentence_UIDs_in_batch = {len(unique_uids)}")

#     # per-segment start/duration — this is the key thing to diff before/after
#     starts = [seg.start for seg in batch.segments]
#     durations = [seg.duration for seg in batch.segments]
#     print(f"  segment durations: min={min(durations):.3f}s max={max(durations):.3f}s "
#           f"(all equal? {len(set(durations)) == 1})")
#     print(f"  first 3 segments: "
#           f"{[(round(s, 3), round(d, 3)) for s, d in zip(starts[:3], durations[:3])]}")


# def main():
#     cfg = debug_config()
#     data = Data(**cfg["data"])
#     loaders = data.build()

#     for split, loader in loaders.items():
#         dataset = loader.dataset
#         print(f"\n=== split={split} ===")
#         print(f"total segments in dataset: {len(dataset.segments)}")

#         all_durations = [seg.duration for seg in dataset.segments]
#         print(f"segment duration range across WHOLE split: "
#               f"min={min(all_durations):.3f}s max={max(all_durations):.3f}s "
#               f"mean={sum(all_durations)/len(all_durations):.3f}s")

#         for i, batch in enumerate(loader):
#             if i >= N_BATCHES_TO_PRINT:
#                 break
#             describe_batch(i, batch)


# if __name__ == "__main__":
#     main()

# Trial-level accumulation debug script.
#
# Does NOT touch the `neuro` (MegExtractor) config or the segmenter — uses the
# pipeline exactly as it is today (keystroke-anchored, 0.5s @ 50Hz chunks).
# Instead, it iterates the WHOLE loader (not just a few batches, since a
# sentence's keystrokes can straddle two batches — see SentenceGroupedDistri-
# butedSampler) and accumulates every chunk under its sentence_UID, so you can
# see what a "reconstructed trial" looks like from the existing per-keystroke
# windows: how many keystrokes, how much the 0.5s windows overlap each other,
# and the true trial span vs. the sum of individual chunk durations.

from collections import defaultdict

import studies  # noqa: F401  (registers Pinet2024Meg / Pinet2024Eeg)

from brain2qwerty_v1.config.xp_config import debug_config
from brain2qwerty_v1.main import Data
from brain2qwerty_v1.utils import CHAR_INDEX

SPLIT = "train"
MAX_TRIALS_TO_PRINT = 10  # set None to print all


def main():
    cfg = debug_config()
    data = Data(**cfg["data"])
    loaders = data.build()
    loader = loaders[SPLIT]

    # uid -> list of (start, duration, neuro_chunk_tensor)
    per_trial = defaultdict(list)

    n_batches = 0
    for batch in loader:
        n_batches += 1
        neuro = batch.data["neuro"]  # (batch, n_channels, n_timepoints)
        for i, seg in enumerate(batch.segments):
            uid = seg.trigger.extra.get("sentence_UID")
            per_trial[uid].append((seg.start, seg.duration, neuro[i]))

        feature = batch.data["feature"]
        decoded = [CHAR_INDEX[i.item()] for i in feature[0]]
        print(feature)
        print(decoded)

        exit()
    print(f"Iterated {n_batches} batches, {len(per_trial)} unique sentence_UIDs found.\n")

    uids = sorted(per_trial.keys())
    if MAX_TRIALS_TO_PRINT is not None:
        uids = uids[:MAX_TRIALS_TO_PRINT]

    for uid in uids:
        chunks = sorted(per_trial[uid], key=lambda c: c[0])  # sort by start time
        starts = [c[0] for c in chunks]
        durations = [c[1] for c in chunks]
        n_keystrokes = len(chunks)

        trial_start = starts[0]
        trial_end = max(s + d for s, d in zip(starts, durations))
        true_span = trial_end - trial_start

        sum_chunk_durations = sum(durations)
        # rough overlap estimate: how much longer the sum of chunks is than
        # the actual span they cover (only meaningful if chunks are contiguous
        # in time, which they roughly are here since sorted by start)
        overlap_estimate = sum_chunk_durations - true_span

        chunk_shape = tuple(chunks[0][2].shape)

        print(f"UID={uid}")
        print(f"  n_keystrokes           = {n_keystrokes}")
        print(f"  trial_start..trial_end = {trial_start:.3f}s .. {trial_end:.3f}s "
              f"(true span = {true_span:.3f}s)")
        print(f"  sum(chunk durations)   = {sum_chunk_durations:.3f}s "
              f"(vs true span -> overlap_estimate = {overlap_estimate:.3f}s)")
        print(f"  inter-keystroke gaps (s): "
              f"{[round(starts[i+1]-starts[i], 3) for i in range(min(5, len(starts)-1))]} ...")
        print(f"  per-chunk shape        = {chunk_shape}  (n_channels, n_timepoints)")
        print()


if __name__ == "__main__":
    main()