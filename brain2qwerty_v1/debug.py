# Standalone dataloader inspector — no training, just builds the loaders and
# prints what's actually inside each batch. Run this once BEFORE changing the
# segmenter (type=="Keystroke", fixed duration) and once AFTER (type=="Sentence",
# duration=None) to diff the two.
#
# Uses debug_config() (single timeline) so it's fast to iterate on.

import studies  # noqa: F401  (registers Pinet2024Meg / Pinet2024Eeg)

from brain2qwerty_v1.config.xp_config import debug_config
from brain2qwerty_v1.main import Data

N_BATCHES_TO_PRINT = 3


def describe_batch(i, batch):
    print(f"\n--- batch {i} ---")
    for name, tensor in batch.data.items():
        print(f"  data[{name!r}].shape = {tuple(tensor.shape)}  dtype={tensor.dtype}")

    uids = [seg.trigger.extra.get("sentence_UID") for seg in batch.segments]
    unique_uids = sorted(set(uids))
    print(f"  n_segments_in_batch = {len(batch.segments)}")
    print(f"  n_unique_sentence_UIDs_in_batch = {len(unique_uids)}")

    # per-segment start/duration — this is the key thing to diff before/after
    starts = [seg.start for seg in batch.segments]
    durations = [seg.duration for seg in batch.segments]
    print(f"  segment durations: min={min(durations):.3f}s max={max(durations):.3f}s "
          f"(all equal? {len(set(durations)) == 1})")
    print(f"  first 3 segments: "
          f"{[(round(s, 3), round(d, 3)) for s, d in zip(starts[:3], durations[:3])]}")


def main():
    cfg = debug_config()
    data = Data(**cfg["data"])
    loaders = data.build()

    for split, loader in loaders.items():
        dataset = loader.dataset
        print(f"\n=== split={split} ===")
        print(f"total segments in dataset: {len(dataset.segments)}")

        all_durations = [seg.duration for seg in dataset.segments]
        print(f"segment duration range across WHOLE split: "
              f"min={min(all_durations):.3f}s max={max(all_durations):.3f}s "
              f"mean={sum(all_durations)/len(all_durations):.3f}s")

        for i, batch in enumerate(loader):
            if i >= N_BATCHES_TO_PRINT:
                break
            describe_batch(i, batch)


if __name__ == "__main__":
    main()