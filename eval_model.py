from meta_model import MetaModel
import pickle, os
import torch
import argparse
from train_meta1 import get_dataset_loaders, eval_model

parser = argparse.ArgumentParser(description="Eval Neural Decoder")
parser.add_argument('--out_dir', type=str, default='default',
                    help="Defaults to modelName if not provided")
parser.add_argument('--dataset_path', type=str, default='/data/hossein/data/speech/speech_data_raw.npz')
parser.add_argument('--is_speech', action='store_true', help='training on speech dataset')
parser.add_argument('--nlp_10', action='store_true', help='nlp 10 instead of 21')
parser.add_argument('--is_nejm', action='store_true', help='nejm speech')
parser.add_argument('--batch_size', type=int, default=16)
run_args = parser.parse_args()


device = torch.device("cuda")
is_speech, is_nejm, nlp10 = run_args.is_speech, run_args.is_nejm, run_args.nlp_10
model = MetaModel(
        num_neurons=192 if not is_speech else (512 if is_nejm else 256),
        num_classes=(41 if is_speech else 32),
    ).to(device)
model.load_state_dict(torch.load(run_args.out_dir + "/modelWeights",))

train_loader, test_loader, _ = get_dataset_loaders(run_args.dataset_path, run_args.batch_size, False, is_speech, nlp10,
                                                   is_nejm, )

cer, _, _ = eval_model(model, test_loader, device)
print(cer)