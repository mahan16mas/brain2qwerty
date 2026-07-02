from train_meta1 import train_model
import argparse

parser = argparse.ArgumentParser(description="Train Neural Decoder")

# Strings
parser.add_argument('--out_dir', type=str, default='default',
                    help="Defaults to modelName if not provided")
parser.add_argument('--dataset_path', type=str, default='/data/hossein/data/speech/speech_data_raw.npz')

# Booleans (Actions are inverse to their defaults)
parser.add_argument('--is_speech', action='store_true', help='training on speech dataset')
parser.add_argument('--nlp_10', action='store_true', help='nlp 10 instead of 21')
parser.add_argument('--is_nejm', action='store_true', help='nejm speech')
# Integers
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--seed', type=int, default=0)

parsed_args = parser.parse_args()

# Convert namespace to dictionary
args_dict = vars(parsed_args)
train_model(args_dict)
