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
parser.add_argument('--use_jude', action='store_true', help='use jude speech')

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

parser.add_argument('--use_mahan_model_params', action='store_true', help='use mahan params for model configs instead of meta defualts')
parser.add_argument('--time_agg_out', type=str, help="time_aggregation method/layer used in convolutional patch encoder pooling part", default="att", choices=['gap', 'linear', 'att'])
parser.add_argument('--cnn_only', action='store_true', help='use Conv output directly for CTC')
parser.add_argument('--cnn_hidden', type=int, default=2048)
parser.add_argument('--cnn_depth', type=int, default=8)
parser.add_argument('--cnn_initial_linear', type=int, default=512)

parser.add_argument('--use_rnn_decoder', action='store_true', help='use ConvRNN model from ablation_model.py')
parser.add_argument('--rnn_hidden', type=int, default=2048, help="Only used when you have passed --use_rnn_decoder: hidden dim of RNN decoder in ConvRNN model")
parser.add_argument('--rnn_layers', type=int, default=5, help="Only used when you have passed --use_rnn_decoder: num layers of RNN decoder in ConvRNN model")
parser.add_argument('--bidir', action='store_true', help="Only used when you have passed --use_rnn_decoder: bidir RNN decoder in ConvRNN model")
parser.add_argument('--rnn_dr', type=float, default=0.4, help="Only used when you have passed --use_rnn_decoder: dropout RNN decoder in ConvRNN model")

parser.add_argument('--cebra_patch_encoder', action='store_true', help='use CEBRA backbone for meta model patch encoder')
# initial_layer_size=512, # hardcoded for now 
parser.add_argument('--cebra_hidden_dim', type=int, default=256, help="Only used when you have passed --cebra_patch_encoder")
parser.add_argument('--cebra_out_dim', type=int, default=64, help="Only used when you have passed --cebra_patch_encoder")
parser.add_argument('--cebra_model_name', type=str, help="Only used when you have passed --cebra_patch_encoder", default="att", choices=['Offset5Model', 'Offset36Dropoutv2'])
parser.add_argument('--cebra_pad_mode', type=str, help="Only used when you have passed --cebra_patch_encoder", default="att", choices=['replicate', 'reflect'])
parser.add_argument('--transformer_depth', type=int, default=4, help="Only used when you have passed --cebra_patch_encoder")
parser.add_argument('--transformer_head', type=int, default=2, help="Only used when you have passed --cebra_patch_encoder")

parser.add_argument('--cebra_rnn', action='store_true', help='use CEBRARNN model')
# for cebra dim, use the `cebra_hidden_dim`, `cebra_out_dim`, `cebra_model_name`, `cebra_pad_mode`, and also add time_agg customization in it later
# for rnn arch, use rnn_hidden, rnn_layers, bidir, rnn_dr

parser.add_argument('--add_noise', action='store_true', help='add noise when training')
parser.add_argument('--gauss_in', action='store_true', help='gaussian smoothing after noise augmentation')
parser.add_argument('--whiteNoiseSD', type=float, default=0.8)
parser.add_argument('--constantOffsetSD', type=float, default=0.2)

parser.add_argument('--no_smoothing', action='store_true', help='no smoothing at all - this can not be called with --ad_noise')

parser.add_argument('--chunk_size', type=int, default=4, help="")
parser.add_argument('--chunk_stride', type=int, default=4, help="")


parsed_args = parser.parse_args()

# Convert namespace to dictionary
args_dict = vars(parsed_args)
train_model(args_dict)

"""
# [X]
CUDA_VISIBLE_DEVICES=1 python start_trainer.py --out_dir 'nlp21_meta_default_50' --dataset_path "/mnt/data/hossein/Hossein_workspace/nips_cetra/mahan/CORP/CORP_data_release" --batch_size 16 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb 
# [X]
CUDA_VISIBLE_DEVICES=1 python start_trainer.py --out_dir 'nlp21_meta_default_50_bs8' --dataset_path "/mnt/data/hossein/Hossein_workspace/nips_cetra/mahan/CORP/CORP_data_release" --batch_size 8 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb 
# [X]
CUDA_VISIBLE_DEVICES=0 python start_trainer.py --out_dir 'nlp21_meta_mahanHyperArch_50_bs16' --dataset_path "/mnt/data/hossein/Hossein_workspace/nips_cetra/mahan/CORP/CORP_data_release" --batch_size 16 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --use_mahan_model_params


# [X]
CUDA_VISIBLE_DEVICES=1 python start_trainer.py --out_dir 'nlp21_meta_default_50_bs8_cnn_only' --dataset_path "/mnt/data/hossein/Hossein_workspace/nips_cetra/mahan/CORP/CORP_data_release" --batch_size 8 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --cnn_only

# [X]
python start_trainer.py --out_dir 'nlp21_meta_default_300' --batch_size 64 --epochs 300 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb  --dataset_path "/data/hossein/mm_project/CORP_data_release"



# [X]
CUDA_VISIBLE_DEVICES=1 python start_trainer.py --use_rnn_decoder --out_dir 'nlp21_meta_convRNN_default_50' --dataset_path "/mnt/data/hossein/Hossein_workspace/nips_cetra/mahan/CORP/CORP_data_release" --batch_size 16 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb 

# [R]
# this is kinda extention on ConvOnly model
CUDA_VISIBLE_DEVICES=1 python start_trainer.py --use_rnn_decoder --out_dir 'nlp21_meta_convRNN_shallow_RNN_default_50_bs8' --dataset_path "/mnt/data/hossein/Hossein_workspace/nips_cetra/mahan/CORP/CORP_data_release" --batch_size 8 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --rnn_hidden 2048 --rnn_layers 1


### CEBRA-Transformer: 
# [X]
CUDA_VISIBLE_DEVICES=0 python start_trainer.py --cebra_patch_encoder --out_dir 'nlp21_meta_CEBRATRANSFORMER_defaultCEBRA(h-256_o-64_replicate_tr-d-4_tr-h-2)_bs16' --dataset_path "/mnt/data/hossein/Hossein_workspace/nips_cetra/mahan/CORP/CORP_data_release" --batch_size 16 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --cebra_model_name "Offset5Model" --cebra_hidden_dim 256 --cebra_out_dim 64  --cebra_pad_mode 'replicate' --transformer_depth 4 --transformer_head 2

# [X]
CUDA_VISIBLE_DEVICES=1 python start_trainer.py --cebra_patch_encoder --out_dir 'nlp21_meta_CEBRATRANSFORMER_defaultCEBRA(h-1024_o-256_reflect_tr-d-4_tr-h-2)_bs16' --dataset_path "/mnt/data/hossein/Hossein_workspace/nips_cetra/mahan/CORP/CORP_data_release" --batch_size 16 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --cebra_model_name "Offset5Model" --cebra_hidden_dim 1024 --cebra_out_dim 256  --cebra_pad_mode 'replicate' --transformer_depth 4 --transformer_head 2


# [   ]
CUDA_VISIBLE_DEVICES=1 python start_trainer.py --cebra_patch_encoder --out_dir 'nlp21_meta_CEBRATRANSFORMER_defaultCEBRA(h-2048_o-2048_reflect_tr-d-4_tr-h-2)_bs16' --dataset_path "/mnt/data/hossein/Hossein_workspace/nips_cetra/mahan/CORP/CORP_data_release" --batch_size 16 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --cebra_model_name "Offset5Model" --cebra_hidden_dim 2048 --cebra_out_dim 2048  --cebra_pad_mode 'replicate' --transformer_depth 4 --transformer_head 2

#####################

# [X]
CUDA_VISIBLE_DEVICES=1 python start_trainer.py --out_dir 'nlp21_meta_default_50_time-agg-gap' --dataset_path "/mnt/data/hossein/Hossein_workspace/nips_cetra/mahan/CORP/CORP_data_release" --batch_size 16 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --time_agg_out 'gap' 

# [ ]
CUDA_VISIBLE_DEVICES=1 python start_trainer.py --out_dir 'nlp21_meta_default_50_time-agg-linear' --dataset_path "/mnt/data/hossein/Hossein_workspace/nips_cetra/mahan/CORP/CORP_data_release" --batch_size 16 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --time_agg_out 'linear' 

# [X]
CUDA_VISIBLE_DEVICES=0 python start_trainer.py --out_dir 'nlp21_meta_default_50_cnn-hidden-1024' --dataset_path "/mnt/data/hossein/Hossein_workspace/nips_cetra/mahan/CORP/CORP_data_release" --batch_size 16 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --cnn_hidden 1024

# [ ] dont run it, because 1024 flopped ----> why?
CUDA_VISIBLE_DEVICES=0 python start_trainer.py --out_dir 'nlp21_meta_default_50_cnn-hidden-512' --dataset_path "/mnt/data/hossein/Hossein_workspace/nips_cetra/mahan/CORP/CORP_data_release" --batch_size 16 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --cnn_hidden 512
"""