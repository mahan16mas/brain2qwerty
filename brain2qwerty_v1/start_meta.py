from .meta_main import train_model
import argparse

parser = argparse.ArgumentParser(description="Train Neural Decoder")

# Strings
parser.add_argument('--out_dir', type=str, default='default',
                    help="Defaults to modelName if not provided")

# Integers
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--epochs', type=int, default=300)

parser.add_argument('--conv_dropout', type=float, default=0.5)
parser.add_argument('--dropout_input', type=float, default=0.2)

parser.add_argument('--do_wandb', action='store_true', help='log w wandb')


parser.add_argument('--time_agg_out', type=str, help="time_aggregation method/layer used in convolutional patch encoder pooling part", default="att", choices=['gap', 'linear', 'att'])
parser.add_argument('--cnn_hidden', type=int, default=2048)
parser.add_argument('--cnn_depth', type=int, default=8)
parser.add_argument('--cnn_initial_linear', type=int, default=512)
parser.add_argument('--transformer_depth', type=int, default=4, help="Only used when you have passed --cebra_patch_encoder")
parser.add_argument('--transformer_head', type=int, default=2, help="Only used when you have passed --cebra_patch_encoder")

parsed_args = parser.parse_args()

# Convert namespace to dictionary
args_dict = vars(parsed_args)
train_model(args_dict)

'''
python -m brain2qwerty_v1.start_meta --out_dir "Meta-NoMerger-NoSubSpecificLayer-MEG-OwnCode_w_CTC-bs16" --do_wandb 
'''