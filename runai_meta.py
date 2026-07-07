import subprocess
import sys
import time

IMAGE = 'registry.rcp.epfl.ch/upmwmathis-mirzaei/robust-cebra:v1.2'
GPU = 1
CPU = 2
MEMORY = '64Gi'
NODE_POOLS = 'h200'
LARGE_SHM = '--large-shm'
PVC_HOME = 'home:/home/mirzaei'
PVC_SCRATCH = 'upmwmathis-scratch:/data'
PROJECT = 'upmwmathis-mirzaei'
CMD_PREFIX = (
    "cd /data/hossein/mm_project/brain2qwerty"
)
PYTHON_PATH = "/data/hossein/mm_project/brain2qwerty/.venv/bin/python"

def run_command(cmd, quiet=False):
    """Run a shell command and print output."""
    if not quiet:
        print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if not quiet:
        if result.returncode != 0:
            print(f"Error (code {result.returncode}): {result.stderr}", file=sys.stderr)
        else:
            print(result.stdout)
    return result.returncode

exports = (
            "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mirzaei/lm-cebra/.venv/lib/python3.10/site-packages/nvidia/cu13/lib && "
            "export TORCHINDUCTOR_CACHE_DIR=/data/hossein/mm_project/tmp/torch_cache"
        )
from itertools import product
from collections import defaultdict
datasets = [0, 1, 2, 3]
dropouts = [(0.1, 0.1), (0.0, 0.0), (0.5, 0.2)]
epochs = [40, 50, 300]
all_hypers = product(datasets, dropouts, epochs)
out_dirs = defaultdict(list)
all_runs = {}
for dataset_num, (conv_dropout, input_dropout), epoch in all_hypers:
    dataset_name = 'speech'
    dataset_dir = '/data/hossein/data/speech/speech_data_raw_all_in_test.pkl'
    batch_size = 16
    speech = True
    nlp10 = False
    nejm = False
    if dataset_num == 1:
        speech = False
        dataset_name = 'nlp21'
        dataset_dir = '/data/hossein/mm_project/CORP_data_release'
    if dataset_num == 2:
        speech = False
        nlp10 = True
        dataset_name = 'nlp10'
        dataset_dir = '"/data/hossein/mm_project/old_nlp_data/data.npz"'
    if dataset_num == 3:
        dataset_dir = "/data/hossein/mm_project/speech_gru_cebra/data/nejm_dataset.pkl"
        speech = True
        nejm = True
        dataset_name = 'nejm'
    name = f"{dataset_name}-meta-{epoch}-{str(f'{conv_dropout}{input_dropout}').replace('.', '')}"
    out_dirs[dataset_num].append(name)
    continue
    args = (
        f"eval_model.py {'--nlp_10' if nlp10 else ''} {'--is_speech' if speech else ''} "
        f"--dataset_path {dataset_dir} {'--is_nejm' if nejm else ''} --out_dir {name} "
        # f"--epochs {epoch} --dropout_input {input_dropout} --conv_dropout {conv_dropout} "
    )
    all_runs[name] = args
print(out_dirs)

num_sumbissions = 0
for job_name, file_name in all_runs.items():
    gpu_name = 'h100'
    log_file = f"{job_name}.log"
    inner_cmd = f"{CMD_PREFIX} && {exports} && {PYTHON_PATH} {file_name} > {log_file} 2>&1"
    bash_command = f"bash -lc '{inner_cmd}'"
    del_cmd = f"runai delete job {job_name}"
    run_command(del_cmd, True)

    time.sleep(1)


    submit_cmd = (
                f"runai submit --name {job_name} "
                f"--image {IMAGE} "
                f"--gpu {GPU} --cpu {CPU} --memory {MEMORY} "
                f"--node-pools {gpu_name} "
                f"{LARGE_SHM} "
                f"--pvc {PVC_HOME} --pvc {PVC_SCRATCH} "
                f"-e HOME=/home/mirzaei -e PYTHONUNBUFFERED=1 "
                f"-p {PROJECT} "
                f"--run-as-user "
                f"--command -- {bash_command}"
            )
    run_command(submit_cmd)
    print(f"Submitted job: {job_name}")
    num_sumbissions += 1


print(f"submitted {num_sumbissions} jobs")