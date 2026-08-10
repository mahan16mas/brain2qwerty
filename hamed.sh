# [X]
# python start_trainer.py --cebra_patch_encoder --out_dir 'nlp21_meta_CEBRATRANSFORMER_defaultCEBRA(h-2048_o-2048_reflect_tr-d-4_tr-h-2)_bs16' --dataset_path "/data/hossein/mm_project/CORP_data_release" --batch_size 16 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --cebra_model_name "Offset5Model" --cebra_hidden_dim 2048 --cebra_out_dim 2048  --cebra_pad_mode 'replicate' --transformer_depth 4 --transformer_head 2

# python start_trainer.py --cebra_patch_encoder --out_dir 'nlp21_meta_CEBRATRANSFORMER_defaultCEBRA(h-2048_o-512_reflect_tr-d-4_tr-h-2)_bs16' --dataset_path "/data/hossein/mm_project/CORP_data_release" --batch_size 16 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --cebra_model_name "Offset5Model" --cebra_hidden_dim 2048 --cebra_out_dim 512  --cebra_pad_mode 'replicate' --transformer_depth 4 --transformer_head 2

# CUDA_VISIBLE_DEVICES=0 python start_trainer.py --cebra_rnn --out_dir "nlp21_meta_CEBRARNN(h-1024_o-1024_reflect_rnn-dim-1024_rnn-layers-5_bidir)-bs8-mahanLR" --dataset_path /mnt/data/hossein/Hossein_workspace/nips_cetra/mahan/CORP/CORP_data_release --batch_size 8 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --cebra_model_name Offset5Model --cebra_hidden_dim 1024 --cebra_out_dim 1024 --cebra_pad_mode replicate --rnn_hidden 1024 --rnn_layers 5 --bidir

# CUDA_VISIBLE_DEVICES=0 python start_trainer.py --out_dir "nlp21_meta_default_50_w_noise(0.8-0.2)-noSmoothing" --dataset_path /mnt/data/hossein/Hossein_workspace/nips_cetra/mahan/CORP/CORP_data_release --batch_size 16 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --add_noise --whiteNoiseSD 0.8 --constantOffsetSD 0.2 

### GOOD ARGS
# python start_trainer.py --cebra_patch_encoder --out_dir 'nlp21_meta_CEBRATRANSFORMER_defaultCEBRA(h-1024_o-1024_reflect_tr-d-4_tr-h-2)_bs16' --dataset_path "/data/hossein/mm_project/CORP_data_release" --batch_size 16 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --cebra_model_name "Offset5Model" --cebra_hidden_dim 1024 --cebra_out_dim 1024  --cebra_pad_mode 'replicate' --transformer_depth 4 --transformer_head 2


# baseline on h200 
# python start_trainer.py --out_dir nlp21_meta_default_50_h200 --dataset_path "/data/hossein/mm_project/CORP_data_release" --batch_size 16 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb
# python start_trainer.py --out_dir nlp21_meta_default_50_linear_h200 --dataset_path "/data/hossein/mm_project/CORP_data_release" --batch_size 16 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --time_agg_out linear

# run it w.no initial layer
# python start_trainer.py --cebra_patch_encoder --out_dir "nlp21_meta_CEBRATRANSFORMER_defaultCEBRA(h-1024_o-1024_reflect_tr-d-4_tr-h-2)_bs16_no_inital_layer" --dataset_path /data/hossein/mm_project/CORP_data_release --batch_size 16 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --cebra_model_name Offset5Model --cebra_hidden_dim 1024 --cebra_out_dim 1024 --cebra_pad_mode replicate --transformer_depth 4 --transformer_head 2
# python start_trainer.py --cebra_patch_encoder --out_dir "nlp21_meta_CEBRATRANSFORMER_defaultCEBRA(h-512_o-512_reflect_tr-d-4_tr-h-2)_bs16_no_inital_layer" --dataset_path /data/hossein/mm_project/CORP_data_release --batch_size 16 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --cebra_model_name Offset5Model --cebra_hidden_dim 512 --cebra_out_dim 512 --cebra_pad_mode replicate --transformer_depth 4 --transformer_head 2

# CEBRA RNN 
# python start_trainer.py --cebra_rnn --out_dir 'nlp21_meta_CEBRARNN(h-1024_o-1024_reflect_rnn-dim-1024_rnn-layers-5)_bs16' --dataset_path "/data/hossein/mm_project/CORP_data_release" --batch_size 16 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --cebra_model_name "Offset5Model" --cebra_hidden_dim 1024 --cebra_out_dim 1024  --cebra_pad_mode 'replicate' --rnn_hidden 1024 --rnn_layers 5

# python start_trainer.py --cebra_rnn --out_dir 'nlp21_meta_CEBRARNN(h-1024_o-1024_reflect_rnn-dim-1024_rnn-layers-1)_bs16' --dataset_path "/data/hossein/mm_project/CORP_data_release" --batch_size 16 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --cebra_model_name "Offset5Model" --cebra_hidden_dim 1024 --cebra_out_dim 1024  --cebra_pad_mode 'replicate' --rnn_hidden 1024 --rnn_layers 1

# python start_trainer.py --cebra_rnn --out_dir 'nlp21_meta_CEBRARNN(h-1024_o-1024_reflect_rnn-dim-512_rnn-layers-5)_bs16' --dataset_path "/mnt/data/hossein/Hossein_workspace/nips_cetra/mahan/CORP/CORP_data_release" --batch_size 16 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --cebra_model_name "Offset5Model" --cebra_hidden_dim 1024 --cebra_out_dim 1024  --cebra_pad_mode 'replicate' --rnn_hidden 512 --rnn_layers 5

# python start_trainer.py --cebra_rnn --out_dir 'nlp21_meta_CEBRARNN(h-512_o-512_reflect_rnn-dim-512_rnn-layers-1)_bs16' --dataset_path "/mnt/data/hossein/Hossein_workspace/nips_cetra/mahan/CORP/CORP_data_release" --batch_size 16 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --cebra_model_name "Offset5Model" --cebra_hidden_dim 512 --cebra_out_dim 512  --cebra_pad_mode 'replicate' --rnn_hidden 512 --rnn_layers 1


# CUDA_VISIBLE_DEVICES=1 python start_trainer.py --cebra_rnn --out_dir 'nlp21_meta_CEBRARNN(h-1024_o-1024_reflect_rnn-dim-1024_rnn-layers-5_bidir)_bs8_300' --dataset_path "/mnt/data/hossein/Hossein_workspace/nips_cetra/mahan/CORP/CORP_data_release" --batch_size 8 --epochs 300 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --cebra_model_name "Offset5Model" --cebra_hidden_dim 1024 --cebra_out_dim 1024  --cebra_pad_mode 'replicate' --rnn_hidden 1024 --rnn_layers 5 --bidir

# CUDA_VISIBLE_DEVICES=0 python start_trainer.py --cebra_rnn --out_dir 'nlp21_meta_CEBRA-NO-RNN(h-1024_o-1024_reflect)_bs16' --dataset_path "/mnt/data/hossein/Hossein_workspace/nips_cetra/mahan/CORP/CORP_data_release" --batch_size 16 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --cebra_model_name "Offset5Model" --cebra_hidden_dim 1024 --cebra_out_dim 1024  --cebra_pad_mode 'replicate' --rnn_hidden 1024

# CUDA_VISIBLE_DEVICES=0 python start_trainer.py --cebra_rnn --out_dir 'nlp21_meta_CEBRARNN(h-1024_o-1024_reflect_rnn-dim-1024_rnn-layers-5_bidir)_bs16' --dataset_path "/mnt/data/hossein/Hossein_workspace/nips_cetra/mahan/CORP/CORP_data_release" --batch_size 8 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --cebra_model_name "Offset5Model" --cebra_hidden_dim 1024 --cebra_out_dim 1024  --cebra_pad_mode 'replicate' --rnn_hidden 1024 --rnn_layers 5 --bidir --rnn_dr 0.0

# CUDA_VISIBLE_DEVICES=0 python start_trainer.py --cebra_rnn --out_dir 'nlp21_meta_CEBRA-LSTM(h-1024_o-1024_reflect_rnn-dim-1024_rnn-layers-5_bidir)_bs16' --dataset_path "/mnt/data/hossein/Hossein_workspace/nips_cetra/mahan/CORP/CORP_data_release" --batch_size 16 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --cebra_model_name "Offset5Model" --cebra_hidden_dim 1024 --cebra_out_dim 1024  --cebra_pad_mode 'replicate' --rnn_hidden 1024 --rnn_layers 5 --bidir

# CUDA_VISIBLE_DEVICES=1 python start_trainer.py --cebra_rnn --out_dir 'nlp21_meta_CEBRA-LSTM(h-1024_o-1024_reflect_rnn-dim-1024_rnn-layers-8_bidir)_bs16' --dataset_path "/mnt/data/hossein/Hossein_workspace/nips_cetra/mahan/CORP/CORP_data_release" --batch_size 16 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --cebra_model_name "Offset5Model" --cebra_hidden_dim 1024 --cebra_out_dim 1024  --cebra_pad_mode 'replicate' --rnn_hidden 1024 --rnn_layers 8 --bidir

###################################################### 

### Default meta with chunk size of 25 
# python start_trainer.py --out_dir "nlp21_meta_default_50_chunk25(kernel25stride4-overlap)" --dataset_path /data/hossein/mm_project/CORP_data_release --batch_size 16 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb

# finding good noise
python start_trainer.py --out_dir "nlp21_meta_default_50_w_noise(0.4-0.1)-noInsideModelSmoothing" --dataset_path /data/hossein/mm_project/CORP_data_release --batch_size 16 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --add_noise --whiteNoiseSD 0.4 --constantOffsetSD 0.1


python start_trainer.py --out_dir "nlp21_meta_default_50_w_noise(0.2-0.05)-noInsideModelSmoothing" --dataset_path /data/hossein/mm_project/CORP_data_release --batch_size 16 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --add_noise --whiteNoiseSD 0.2 --constantOffsetSD 0.05


python start_trainer.py --out_dir "nlp21_meta_default_50_w_noise(0.1-0.025)-noInsideModelSmoothing" --dataset_path /data/hossein/mm_project/CORP_data_release --batch_size 16 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --add_noise --whiteNoiseSD 0.1 --constantOffsetSD 0.025