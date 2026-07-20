from train_meta1 import train_model
import argparse

parser = argparse.ArgumentParser(description="Train Neural Decoder")

# Strings
parser.add_argument('--out_dir', type=str, default='default',
                    help="Defaults to modelName if not provided")
parser.add_argument('--dataset_path', type=str, default=rf"/data/hossein/mm_project/CORP_data_release")
# A5000
# DATASET_DIR = "/mnt/data/hossein/Hossein_workspace/nips_cetra/mahan/CORP/CORP_data_release"
# LOCAL
# DATASET_DIR = rf'D:\Pose\NeuroNLP\data\CORP_data_release'
# cluster 
# /data/hossein/mm_project/CORP_data_release
 
# Booleans (Actions are inverse to their defaults)
parser.add_argument('--is_speech', action='store_true', help='training on speech dataset')
parser.add_argument('--nlp_10', action='store_true', help='nlp 10 instead of 21')
parser.add_argument('--is_nejm', action='store_true', help='nejm speech')
# Integers
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--epochs', type=int, default=300)

parser.add_argument('--conv_dropout', type=float, default=0.5)
parser.add_argument('--dropout_input', type=float, default=0.2)

parsed_args = parser.parse_args()

# Convert namespace to dictionary
args_dict = vars(parsed_args)
train_model(args_dict)
