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
# python start_trainer.py --out_dir "nlp21_meta_default_50_w_noise(0.8-0.2)_(chunk25_stride4)" --dataset_path "/data/hossein/mm_project/CORP_data_release" --batch_size 16 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --add_noise --whiteNoiseSD 0.8 --constantOffsetSD 0.2 --chunk_size 25 --chunk_stride 4

# python start_trainer.py --out_dir "nlp21_meta_default_50_w_noise(0.8-0.2)_(chunk4_stride4)" --dataset_path "/data/hossein/mm_project/CORP_data_release" --batch_size 16 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --add_noise --whiteNoiseSD 0.8 --constantOffsetSD 0.2 --chunk_size 4 --chunk_stride 4

# python start_trainer.py --out_dir "nlp21_meta_default_50_w_noise(0.1-0.025)_(chunk4_stride4)" --dataset_path /data/hossein/mm_project/CORP_data_release --batch_size 16 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --add_noise --whiteNoiseSD 0.1 --constantOffsetSD 0.025 --chunk_size 4 --chunk_stride 4

# python start_trainer.py --use_rnn_decoder --out_dir "nlp21_meta_convRNN_default_50__w_noise(0.8-0.2)_(chunk4_stride4)" --dataset_path /data/hossein/mm_project/CORP_data_release --batch_size 16 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --add_noise --whiteNoiseSD 0.8 --constantOffsetSD 0.2 --chunk_size 4 --chunk_stride 4 --do_wandb

# python start_trainer.py --use_rnn_decoder --out_dir "nlp21_meta_convRNN_default_50_(chunk25_stride4)" --dataset_path /data/hossein/mm_project/CORP_data_release --batch_size 16 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --chunk_size 25 --chunk_stride 4 --do_wandb

# python start_trainer.py --use_rnn_decoder --out_dir "nlp21_meta_convRNN_default_50_(chunk4_stride4)_w_noise(0.8-0.2)" --dataset_path /data/hossein/mm_project/CORP_data_release --batch_size 16 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --chunk_size 4 --chunk_stride 4 --do_wandb --add_noise --whiteNoiseSD 0.8 --constantOffsetSD 0.2 

# python start_trainer.py --use_rnn_decoder --out_dir "nlp21_meta_convRNN_default_50_(chunk4_stride4)_w_noise(0.1-0.125)" --dataset_path /data/hossein/mm_project/CORP_data_release --batch_size 16 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --chunk_size 4 --chunk_stride 4 --do_wandb --add_noise --whiteNoiseSD 0.1 --constantOffsetSD 0.125 


############# making META lighter 
# CUDA_VISIBLE_DEVICES=0 python start_trainer.py --out_dir "nlp21_meta-ShallowCNN(d=4,2048)-Tran(4,2)_50_bs8_chunk25.4" --dataset_path /mnt/data/hossein/Hossein_workspace/nips_cetra/mahan/CORP/CORP_data_release --batch_size 8 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --cnn_hidden 2048 --transformer_depth 4 --transformer_head 2 --chunk_size 25 --chunk_stride 4

# CUDA_VISIBLE_DEVICES=0 python start_trainer.py --out_dir "nlp21_meta-ShallowCNN(d=4,2048)-Tran(2,1)_50_bs8_chunk25.4" --dataset_path /mnt/data/hossein/Hossein_workspace/nips_cetra/mahan/CORP/CORP_data_release --batch_size 8 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --cnn_hidden 2048 --transformer_depth 2 --transformer_head 1 --chunk_size 25 --chunk_stride 4

# CUDA_VISIBLE_DEVICES=1 python start_trainer.py --out_dir "nlp21_meta-ShallowCNN(d=2,1024)-Tran(4,2)_50_bs8_chunk25.4" --dataset_path /mnt/data/hossein/Hossein_workspace/nips_cetra/mahan/CORP/CORP_data_release --batch_size 8 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --cnn_hidden 1024 --transformer_depth 4 --transformer_head 2 --chunk_size 25 --chunk_stride 4 --cnn_depth 2 

# CUDA_VISIBLE_DEVICES=1 python start_trainer.py --out_dir "nlp21_meta-ShallowCNN(d=2,2048)-Tran(2,1)_50_bs8_chunk25.4" --dataset_path /mnt/data/hossein/Hossein_workspace/nips_cetra/mahan/CORP/CORP_data_release --batch_size 8 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --cnn_hidden 1024 --transformer_depth 2 --transformer_head 1 --chunk_size 25 --chunk_stride 4 --cnn_depth 2 

# CUDA_VISIBLE_DEVICES=1 python start_trainer.py --out_dir "nlp21_meta-ShallowCNN(d=2,512)-Tran(2,1)_50_bs8_chunk25.4" --dataset_path /mnt/data/hossein/Hossein_workspace/nips_cetra/mahan/CORP/CORP_data_release --batch_size 8 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --cnn_hidden 512 --transformer_depth 2 --transformer_head 1 --chunk_size 25 --chunk_stride 4 --cnn_depth 2 


# python start_trainer.py --out_dir "nlp21_meta-ShallowCNN(d=8,2048)-Tran(2,1)_50_bs8_chunk25.4" --dataset_path /data/hossein/mm_project/CORP_data_release --batch_size 8 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --cnn_hidden 2048 --transformer_depth 2 --transformer_head 1 --chunk_size 25 --chunk_stride 4 --cnn_depth 8 

# python start_trainer.py --out_dir "nlp21_meta-ShallowCNN(d=8,1024)-Tran(2,1)_50_bs8_chunk25.4" --dataset_path /data/hossein/mm_project/CORP_data_release --batch_size 8 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --cnn_hidden 1024 --transformer_depth 2 --transformer_head 1 --chunk_size 25 --chunk_stride 4 --cnn_depth 8 

# python start_trainer.py --out_dir "nlp21_meta-NoCNN(d=0,2048)-Tran(2,1)_50_bs8_chunk25.4_GAP" --dataset_path /data/hossein/mm_project/CORP_data_release --batch_size 8 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --cnn_hidden 2048 --transformer_depth 2 --transformer_head 1 --chunk_size 25 --chunk_stride 4 --cnn_depth 0 --cnn_initial_linear 2048 --time_agg_out 'gap'

# python start_trainer.py --out_dir "nlp21_meta-NoCNN(d=0,2048)-Tran(4,2)_50_bs8_chunk25.4_GAP" --dataset_path /data/hossein/mm_project/CORP_data_release --batch_size 8 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --cnn_hidden 2048 --transformer_depth 4 --transformer_head 2 --chunk_size 25 --chunk_stride 4 --cnn_depth 0 --cnn_initial_linear 2048 --time_agg_out 'gap'

# python start_trainer.py --out_dir "nlp21_meta-NoCNN(d=0,2048)-Tran(2,1)_50_bs8_chunk4.4" --dataset_path /data/hossein/mm_project/CORP_data_release --batch_size 8 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --cnn_hidden 2048 --transformer_depth 2 --transformer_head 1 --chunk_size 4 --chunk_stride 4 --cnn_depth 0 --cnn_initial_linear 2048

# python start_trainer.py --out_dir "nlp21_meta-NoCNN(d=0,2048)-Tran(4,2)_50_bs8_chunk4.4" --dataset_path /data/hossein/mm_project/CORP_data_release --batch_size 8 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --cnn_hidden 2048 --transformer_depth 4 --transformer_head 2 --chunk_size 4 --chunk_stride 4 --cnn_depth 0 --cnn_initial_linear 2048


# python start_trainer.py --out_dir "nlp21_meta-NoCNN(d=0,2048)-Tran(2,1)_50_bs8_chunk4.4_GAP" --dataset_path /data/hossein/mm_project/CORP_data_release --batch_size 8 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --cnn_hidden 2048 --transformer_depth 2 --transformer_head 1 --chunk_size 4 --chunk_stride 4 --cnn_depth 0 --cnn_initial_linear 2048 --time_agg_out 'gap'

# python start_trainer.py --out_dir "nlp21_meta-NoCNN(d=0,2048)-Tran(4,2)_50_bs8_chunk4.4_GAP" --dataset_path /data/hossein/mm_project/CORP_data_release --batch_size 8 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --cnn_hidden 2048 --transformer_depth 4 --transformer_head 2 --chunk_size 4 --chunk_stride 4 --cnn_depth 0 --cnn_initial_linear 2048 --time_agg_out 'gap'



# A 
# 2
# python start_trainer_cebra.py --datasetPath /data/hossein/mm_project/CORP_data_release --offset 4  --gru --bidir --batchSize 8 --random_offset --hidden 1024 --dropout 0.4 --layers 5  --kernel 32 --stride 4 --seed 5 --do_wandb --no_contrastive --no_noise --cebra_unfolder --ceb_hidden 256 --ceb_out 64 --nBatch 20000 --out_dir METECEBRA_A-LargeConv-Unfold_CEBRA_32_4-opt_META --convStyle LARGE --unfolding CEBRA_32_4 --optimStyle META
# # 4 
# python start_trainer_cebra.py --datasetPath /data/hossein/mm_project/CORP_data_release --offset 4  --gru --bidir --batchSize 8 --random_offset --hidden 1024 --dropout 0.4 --layers 5  --kernel 32 --stride 4 --seed 5 --do_wandb --no_contrastive --no_noise --cebra_unfolder --ceb_hidden 256 --ceb_out 64 --nBatch 20000 --out_dir METECEBRA_A-CEBRAConv-Unfold_CEBRA_32_4-opt_META --convStyle CEBRA --unfolding CEBRA_32_4 --optimStyle META 
# # 6
# python start_trainer_cebra.py --datasetPath /data/hossein/mm_project/CORP_data_release --offset 4  --gru --bidir --batchSize 8 --random_offset --hidden 1024 --dropout 0.4 --layers 5  --kernel 32 --stride 4 --seed 5 --do_wandb --no_contrastive --no_noise --cebra_unfolder --ceb_hidden 256 --ceb_out 64 --nBatch 20000 --out_dir METECEBRA_A-LargeConv-Unfold_CEBRA_4_4-opt_META --convStyle LARGE --unfolding CEBRA_4_4 --optimStyle META 
# # 8
# python start_trainer_cebra.py --datasetPath /data/hossein/mm_project/CORP_data_release --offset 4  --gru --bidir --batchSize 8 --random_offset --hidden 1024 --dropout 0.4 --layers 5  --kernel 32 --stride 4 --seed 5 --do_wandb --no_contrastive --no_noise --cebra_unfolder --ceb_hidden 256 --ceb_out 64 --nBatch 20000 --out_dir METECEBRA_A-CEBRAConv-Unfold_CEBRA_4_4-opt_META --convStyle CEBRA --unfolding CEBRA_4_4 --optimStyle META 
# # 10
# python start_trainer_cebra.py --datasetPath /data/hossein/mm_project/CORP_data_release --offset 4  --gru --bidir --batchSize 8 --random_offset --hidden 1024 --dropout 0.4 --layers 5  --kernel 32 --stride 4 --seed 5 --do_wandb --no_contrastive --no_noise --cebra_unfolder --ceb_hidden 256 --ceb_out 64 --nBatch 20000 --out_dir METECEBRA_A-LargeConv-Unfold_AVGPOOL_4_4-opt_META --convStyle LARGE --unfolding AVGPOOL_4_4 --optimStyle META 
# # 1
# python start_trainer_cebra.py --datasetPath /data/hossein/mm_project/CORP_data_release --offset 4  --gru --bidir --batchSize 8 --random_offset --hidden 1024 --dropout 0.4 --layers 5  --kernel 32 --stride 4 --seed 5 --do_wandb --no_contrastive --no_noise --cebra_unfolder --ceb_hidden 256 --ceb_out 64 --nBatch 20000 --out_dir METECEBRA_A-LargeConv-Unfold_CEBRA_32_4-opt_CEBRA --convStyle LARGE --unfolding CEBRA_32_4 --optimStyle CEBRA 
# # 3
# python start_trainer_cebra.py --datasetPath /data/hossein/mm_project/CORP_data_release --offset 4  --gru --bidir --batchSize 8 --random_offset --hidden 1024 --dropout 0.4 --layers 5  --kernel 32 --stride 4 --seed 5 --do_wandb --no_contrastive --no_noise --cebra_unfolder --ceb_hidden 256 --ceb_out 64 --nBatch 20000 --out_dir METECEBRA_A-CEBRAConv-Unfold_CEBRA_32_4-opt_CEBRA --convStyle CEBRA --unfolding CEBRA_32_4 --optimStyle CEBRA 
# # 5
# python start_trainer_cebra.py --datasetPath /data/hossein/mm_project/CORP_data_release --offset 4  --gru --bidir --batchSize 8 --random_offset --hidden 1024 --dropout 0.4 --layers 5  --kernel 32 --stride 4 --seed 5 --do_wandb --no_contrastive --no_noise --cebra_unfolder --ceb_hidden 256 --ceb_out 64 --nBatch 20000 --out_dir METECEBRA_A-LargeConv-Unfold_CEBRA_4_4-opt_CEBRA --convStyle LARGE --unfolding CEBRA_4_4 --optimStyle CEBRA 
# # 7
# python start_trainer_cebra.py --datasetPath /data/hossein/mm_project/CORP_data_release --offset 4  --gru --bidir --batchSize 8 --random_offset --hidden 1024 --dropout 0.4 --layers 5  --kernel 32 --stride 4 --seed 5 --do_wandb --no_contrastive --no_noise --cebra_unfolder --ceb_hidden 256 --ceb_out 64 --nBatch 20000 --out_dir METECEBRA_A-CEBRAConv-Unfold_CEBRA_4_4-opt_CEBRA --convStyle CEBRA --unfolding CEBRA_4_4 --optimStyle CEBRA 

# # 9
# python start_trainer_cebra.py --datasetPath /data/hossein/mm_project/CORP_data_release --offset 4  --gru --bidir --batchSize 8 --random_offset --hidden 1024 --dropout 0.4 --layers 5  --kernel 32 --stride 4 --seed 5 --do_wandb --no_contrastive --no_noise --cebra_unfolder --ceb_hidden 256 --ceb_out 64 --nBatch 20000 --out_dir METECEBRA_A-LargeConv-Unfold_AVGPOOL_4_4-opt_CEBRA --convStyle LARGE --unfolding AVGPOOL_4_4 --optimStyle CEBRA 
# # 11
# python start_trainer_cebra.py --datasetPath /data/hossein/mm_project/CORP_data_release --offset 4  --gru --bidir --batchSize 8 --random_offset --hidden 1024 --dropout 0.4 --layers 5  --kernel 32 --stride 4 --seed 5 --do_wandb --no_contrastive --no_noise --cebra_unfolder --ceb_hidden 256 --ceb_out 64 --nBatch 20000 --out_dir METECEBRA_A-LargeConv-Unfold_AVGPOOL_25_4-opt_CEBRA --convStyle LARGE --unfolding AVGPOOL_25_4 --optimStyle CEBRA 
# # 12
# python start_trainer_cebra.py --datasetPath /data/hossein/mm_project/CORP_data_release --offset 4  --gru --bidir --batchSize 8 --random_offset --hidden 1024 --dropout 0.4 --layers 5  --kernel 32 --stride 4 --seed 5 --do_wandb --no_contrastive --no_noise --cebra_unfolder --ceb_hidden 256 --ceb_out 64 --nBatch 20000 --out_dir METECEBRA_A-LargeConv-Unfold_AVGPOOL_25_4-opt_META --convStyle LARGE --unfolding AVGPOOL_25_4 --optimStyle META 
# # 13
# python start_trainer_cebra.py --datasetPath /data/hossein/mm_project/CORP_data_release --offset 4  --gru --bidir --batchSize 8 --random_offset --hidden 1024 --dropout 0.4 --layers 5  --kernel 32 --stride 4 --seed 5 --do_wandb --no_contrastive --no_noise --cebra_unfolder --ceb_hidden 256 --ceb_out 64 --nBatch 20000 --out_dir METECEBRA_A-LargeConv-Unfold_KERNEL_4_4-opt_CEBRA --convStyle LARGE --unfolding KERNEL_4_4 --optimStyle CEBRA 
# # 14
# python start_trainer_cebra.py --datasetPath /data/hossein/mm_project/CORP_data_release --offset 4  --gru --bidir --batchSize 8 --random_offset --hidden 1024 --dropout 0.4 --layers 5  --kernel 32 --stride 4 --seed 5 --do_wandb --no_contrastive --no_noise --cebra_unfolder --ceb_hidden 256 --ceb_out 64 --nBatch 20000 --out_dir METECEBRA_A-LargeConv-Unfold_KERNEL_4_4-opt_META --convStyle LARGE --unfolding KERNEL_4_4 --optimStyle META 


# python start_trainer_cebra.py --datasetPath /data/hossein/mm_project/CORP_data_release --offset 4  --gru --bidir --batchSize 8 --random_offset --hidden 1024 --dropout 0.4 --layers 5  --kernel 32 --stride 4 --seed 5 --do_wandb --no_contrastive --no_noise --cebra_unfolder --ceb_hidden 256 --ceb_out 64 --nBatch 20000 --out_dir METECEBRA_A-LargeConv-Unfold_CEBRA_4_4-opt_META-LONGER --convStyle LARGE --unfolding CEBRA_4_4 --optimStyle META 
# python start_trainer_cebra.py --datasetPath /data/hossein/mm_project/CORP_data_release --offset 4  --gru --bidir --batchSize 8 --random_offset --hidden 1024 --dropout 0.4 --layers 5  --kernel 32 --stride 4 --seed 5 --do_wandb --no_contrastive --no_noise --cebra_unfolder --ceb_hidden 256 --ceb_out 64 --nBatch 20000 --out_dir METECEBRA_A-LargeConv-Unfold_AVGPOOL_4_4-opt_META-LONGER --convStyle LARGE --unfolding AVGPOOL_4_4 --optimStyle META 
# python start_trainer_cebra.py --datasetPath /data/hossein/mm_project/CORP_data_release --offset 4  --gru --bidir --batchSize 8 --random_offset --hidden 1024 --dropout 0.4 --layers 5  --kernel 32 --stride 4 --seed 5 --do_wandb --no_contrastive --no_noise --cebra_unfolder --ceb_hidden 256 --ceb_out 64 --nBatch 20000 --out_dir METECEBRA_A-LargeConv-Unfold_KERNEL_4_4-opt_META-LONG --convStyle LARGE --unfolding KERNEL_4_4 --optimStyle META 
# python start_trainer_cebra.py --datasetPath /data/hossein/mm_project/CORP_data_release --offset 4  --gru --bidir --batchSize 8 --random_offset --hidden 1024 --dropout 0.4 --layers 5  --kernel 32 --stride 4 --seed 5 --do_wandb --no_contrastive --no_noise --cebra_unfolder --ceb_hidden 256 --ceb_out 64 --nBatch 20000 --out_dir METECEBRA_A-LargeConv-Unfold_AVGPOOL_25_4-opt_META-LONG --convStyle LARGE --unfolding AVGPOOL_25_4 --optimStyle META 


# Running Speech with kernel 25 and stride 4 
# /data/hossein/data/speech/speech_data_raw_all_in_test.pkl
python start_trainer.py --out_dir "speech_meta_default_50_chunk25(kernel25stride4-overlap)_bs8" --dataset_path /data/hossein/data/speech/speech_data_raw_all_in_test.pkl --batch_size 9 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --chunk_size 25 --chunk_stride 4
