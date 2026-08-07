# [X]
# python start_trainer.py --cebra_patch_encoder --out_dir 'nlp21_meta_CEBRATRANSFORMER_defaultCEBRA(h-2048_o-2048_reflect_tr-d-4_tr-h-2)_bs16' --dataset_path "/data/hossein/mm_project/CORP_data_release" --batch_size 16 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --cebra_model_name "Offset5Model" --cebra_hidden_dim 2048 --cebra_out_dim 2048  --cebra_pad_mode 'replicate' --transformer_depth 4 --transformer_head 2

# python start_trainer.py --cebra_patch_encoder --out_dir 'nlp21_meta_CEBRATRANSFORMER_defaultCEBRA(h-2048_o-512_reflect_tr-d-4_tr-h-2)_bs16' --dataset_path "/data/hossein/mm_project/CORP_data_release" --batch_size 16 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --cebra_model_name "Offset5Model" --cebra_hidden_dim 2048 --cebra_out_dim 512  --cebra_pad_mode 'replicate' --transformer_depth 4 --transformer_head 2


# python start_trainer.py --cebra_patch_encoder --out_dir 'nlp21_meta_CEBRATRANSFORMER_defaultCEBRA(h-1024_o-1024_reflect_tr-d-4_tr-h-2)_bs16' --dataset_path "/data/hossein/mm_project/CORP_data_release" --batch_size 16 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --cebra_model_name "Offset5Model" --cebra_hidden_dim 1024 --cebra_out_dim 1024  --cebra_pad_mode 'replicate' --transformer_depth 4 --transformer_head 2


# baseline on h200 
python start_trainer.py --out_dir nlp21_meta_default_50_h200 --dataset_path "/data/hossein/mm_project/CORP_data_release" --batch_size 16 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb

python start_trainer.py --out_dir nlp21_meta_default_50_linear_h200 --dataset_path "/data/hossein/mm_project/CORP_data_release" --batch_size 16 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --time_agg_out linear



