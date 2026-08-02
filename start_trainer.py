from train_meta1 import train_model
import argparse

parser = argparse.ArgumentParser(description="Train Neural Decoder")

# Strings
parser.add_argument('--out_dir', type=str, default='default',
                    help="Defaults to modelName if not provided")
parser.add_argument('--dataset_path', type=str, default='/mnt/data/hossein/Hossein_workspace/nips_cetra/mahan/CORP/CORP_data_release')

# Booleans (Actions are inverse to their defaults)
parser.add_argument('--is_speech', action='store_true', help='training on speech dataset')
parser.add_argument('--nlp_10', action='store_true', help='nlp 10 instead of 21')
parser.add_argument('--is_nejm', action='store_true', help='nejm speech')

"""
parser.add_argument('--speech_data_dir', type=str, default='/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/neuronlp/data/speech_old/save_data_speech_mahan.pkl')
parser.add_argument('--nlp_10_data_dir', type=str, default="/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/neuronlp/data/nlp10/data.pkl")
parser.add_argument('--nlp_21_data_dir', type=str, default='/mnt/data/hossein/Hossein_workspace/nips_cetra/mahan/CORP/CORP_data_release')
parser.add_argument('--nejm_dataset', type=str, default="/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/neuronlp/data/nejm/nejm_dataset.pkl")
parser.add_argument('--adv_norm', type=str, default='linf')
"""
# Integers
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--epochs', type=int, default=300)

parser.add_argument('--conv_dropout', type=float, default=0.5)
parser.add_argument('--dropout_input', type=float, default=0.2)

parser.add_argument('--do_wandb', action='store_true', help='log w wandb')

parsed_args = parser.parse_args()

# Convert namespace to dictionary
args_dict = vars(parsed_args)
train_model(args_dict)

"""
CUDA_VISIBLE_DEVICES=1 python start_trainer.py --out_dir 'nlp21_default_300' --dataset_path "/mnt/data/hossein/Hossein_workspace/nips_cetra/mahan/CORP/CORP_data_release" --batch_size 8 --epochs 300 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb
"""