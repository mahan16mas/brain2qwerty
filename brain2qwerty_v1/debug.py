# Trial-level accumulation debug script.
#
# Does NOT touch the `neuro` (MegExtractor) config or the segmenter — uses the
# pipeline exactly as it is today (keystroke-anchored, 0.5s @ 50Hz chunks).
# Instead, it iterates the WHOLE loader (not just a few batches, since a
# sentence's keystrokes can straddle two batches — see SentenceGroupedDistri-
# butedSampler) and accumulates every chunk under its sentence_UID, so you can
# see what a "reconstructed trial" looks like from the existing per-keystroke
# windows: how many keystrokes actually made it into the dataloader (vs. how
# many the sentence really has, since remove_incomplete_segments=True can drop
# edge keystrokes), the true trial span, and how much the 0.5s windows overlap.

from collections import defaultdict

import studies  # noqa: F401  (registers Pinet2024Meg / Pinet2024Eeg)

from brain2qwerty_v1.config.xp_config import debug_config
from brain2qwerty_v1.main import Data

SPLIT = "train"
MAX_TRIALS_TO_PRINT = 10  # set None to print all


def main():
    cfg = debug_config()
    data = Data(**cfg["data"])
    loaders = data.build()
    loader = loaders[SPLIT]

    # uid -> list of (window_start, duration, trigger_time, neuro_chunk_tensor, extra)
    per_trial = defaultdict(list)

    n_batches = 0
    for batch in loader:
        n_batches += 1
        neuro = batch.data["neuro"]  # (batch, n_channels, n_timepoints)
        for i, seg in enumerate(batch.segments):
            extra = seg.trigger.extra
            uid = extra.get("sentence_UID")
            per_trial[uid].append(
                (seg.start, seg.duration, seg.trigger.start, neuro[i], extra)
            )

    print(f"Iterated {n_batches} batches, {len(per_trial)} unique sentence_UIDs found.\n")

    uids = sorted(per_trial.keys())
    if MAX_TRIALS_TO_PRINT is not None:
        uids = uids[:MAX_TRIALS_TO_PRINT]

    for uid in uids:
        # sort by the actual keystroke trigger time (not the window start,
        # which is offset by config["start"]=-0.2s)
        chunks = sorted(per_trial[uid], key=lambda c: c[2])
        window_starts = [c[0] for c in chunks]
        durations = [c[1] for c in chunks]
        trigger_times = [c[2] for c in chunks]  # actual keystroke timesteps
        extra0 = chunks[0][4]

        n_keystrokes_in_dataloader = len(chunks)

        sentence = extra0.get("sentence", extra0.get("text", "<not found>"))
        sentence_typed = extra0.get("sentence_typed", "<not found>")
        n_keystrokes_expected = (
            len(sentence_typed) if isinstance(sentence_typed, str) else None
        )

        first_keystroke_timestep = trigger_times[0]
        last_keystroke_timestep = trigger_times[-1]
        true_span = last_keystroke_timestep - first_keystroke_timestep

        sum_chunk_durations = sum(durations)
        overlap_estimate = sum_chunk_durations - true_span

        chunk_shape = tuple(chunks[0][3].shape)

        print(f"UID={uid}")
        print(f"  sentence (ground truth) = {sentence!r}")
        print(f"  sentence_typed          = {sentence_typed!r}")
        print(f"  n_keystrokes: in dataloader={n_keystrokes_in_dataloader}  "
              f"expected(from sentence_typed)={n_keystrokes_expected}  "
              f"dropped={None if n_keystrokes_expected is None else n_keystrokes_expected - n_keystrokes_in_dataloader}")
        print(f"  first keystroke timestep (s, in-recording) = {first_keystroke_timestep:.3f}")
        print(f"  last  keystroke timestep (s, in-recording) = {last_keystroke_timestep:.3f}")
        print(f"  true span (last - first)                   = {true_span:.3f}s")
        print(f"  sum(chunk durations) = {sum_chunk_durations:.3f}s "
              f"(overlap_estimate = {overlap_estimate:.3f}s)")
        print(f"  inter-keystroke gaps (s, first 5): "
              f"{[round(trigger_times[i+1]-trigger_times[i], 3) for i in range(min(5, len(trigger_times)-1))]}")
        print(f"  per-chunk shape = {chunk_shape}  (n_channels, n_timepoints)")
        print()


if __name__ == "__main__":
    main()