import wandb
import math

api = wandb.Api()

entity = "Trust-Pose-NLP"
project = "NeuroNLP"

runs = api.runs(f"{entity}/{project}")

for run in runs:
    print(f"\nProcessing {run.name} ({run.id})")

    # Skip experiments that we've already processed
    if "best_test_cer" in run.summary:
        print(
            f"  already processed, skipping "
            f"(best_test_cer = {run.summary['best_test_cer']})"
        )
        continue

    best_cer = math.inf
    found_test_cer = False

    # Look through the complete history.
    # If test_cer was never logged, no valid values will be found.
    for row in run.scan_history(keys=["test_cer"]):
        cer = row.get("test_cer")

        if cer is None:
            continue

        found_test_cer = True

        try:
            cer = float(cer)

            if not math.isnan(cer):
                best_cer = min(best_cer, cer)

        except (TypeError, ValueError):
            print(f"  warning: invalid test_cer value: {cer}")

    # test_cer was not logged for this experiment
    if not found_test_cer:
        print("  test_cer not found, skipping")
        continue

    # test_cer existed, but contained no usable numerical values
    if best_cer == math.inf:
        print("  test_cer exists but has no valid numerical values, skipping")
        continue

    print(f"  best test CER = {best_cer}")

    # Add the result to the old run's summary
    run.summary["best_test_cer"] = best_cer
    run.summary.update()

    print("  summary updated")

exit()

import torch

CHUNK_SIZE = 6
STRIDE = 4

# Fake sequence:
# timestep 0 -> [0, 0]
# timestep 1 -> [1, 1]
# ...
# timestep 19 -> [19, 19]
T = 20
feat_dim = 2

x = torch.arange(T).unsqueeze(1).repeat(1, feat_dim).float()
print(x)
# Pretend this is xs from your collate function
xs = (x,)

all_chunks = []
uids = []

for uid, x in enumerate(xs):
    T, feat_dim = x.shape

    # Normal sliding windows
    starts = list(range(0, T - CHUNK_SIZE + 1, STRIDE))

    # Make sure the end of the recording is included
    last_start = T - CHUNK_SIZE

    if last_start >= 0 and (not starts or starts[-1] != last_start):
        starts.append(last_start)

    for start in starts:
        chunk = x[start : start + CHUNK_SIZE]

        all_chunks.append(chunk)
        uids.append(uid)

# Stack them
chunks = torch.stack(all_chunks)

print("Original x shape:", x.shape)
print("Chunks shape:", chunks.shape)
print("Start positions:", starts)
print()

for i, chunk in enumerate(chunks):
    # Only print the first feature because both features contain
    # the same timestep number
    print(f"Chunk {i}: {chunk[:, :].tolist()}")