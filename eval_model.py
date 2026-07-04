from meta_model import MetaModel
import pickle, os
import torch
import argparse
from train_meta1 import get_dataset_loaders, eval_model, model_logits

def dataset_name(is_speech, is_nejm, nlp10):
    if is_speech:
        return "speech" if not is_nejm else "nejm"
    if nlp10: return "nlp10"
    return "nlp21"

parser = argparse.ArgumentParser(description="Eval Neural Decoder")
parser.add_argument('--out_dir', type=str, default='default',
                    help="Defaults to modelName if not provided")
parser.add_argument('--dataset_path', type=str, default='/data/hossein/data/speech/speech_data_raw.npz')
parser.add_argument('--is_speech', action='store_true', help='training on speech dataset')
parser.add_argument('--nlp_10', action='store_true', help='nlp 10 instead of 21')
parser.add_argument('--is_nejm', action='store_true', help='nejm speech')
parser.add_argument('--batch_size', type=int, default=8)
run_args = parser.parse_args()


device = torch.device("cuda")
is_speech, is_nejm, nlp10 = run_args.is_speech, run_args.is_nejm, run_args.nlp_10
model = MetaModel(
        num_neurons=192 if not is_speech else (512 if is_nejm else 256),
        num_classes=(41 if is_speech else 32),
    ).to(device)
model.load_state_dict(torch.load(run_args.out_dir + "/modelWeights",weights_only=False), )

train_loader, test_loader, _ = get_dataset_loaders(run_args.dataset_path, run_args.batch_size, False, is_speech, nlp10,
                                                   is_nejm, )

ds_name = dataset_name(is_speech, is_nejm, nlp10)
cer, _, raw_outputs = eval_model(model, test_loader, device)
with open(run_args.out_dir + "/evalStats", "wb") as f:
    pickle.dump(raw_outputs, f)
rnn_outputs = model_logits(model, test_loader, device, not is_speech)
os.makedirs(f"/data/hossein/mm_project/speech_gru_cebra/meta_{ds_name}", exist_ok=True)
with open(f"/data/hossein/mm_project/speech_gru_cebra/meta_{ds_name}/logits", "wb") as f:
    pickle.dump(rnn_outputs, f)

print(cer)