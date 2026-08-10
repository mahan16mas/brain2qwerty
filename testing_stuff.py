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