### 
# --chunk_size 4 --chunk_stride 4 
CUDA_VISIBLE_DEVICES=1 python start_trainer.py --out_dir "nlp21_meta_default_50_chunk25(kernel25stride4-overlap)" --dataset_path "/data/hossein/mm_project/CORP_data_release" --batch_size 16 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --chunk_size 25 --chunk_stride 4

CUDA_VISIBLE_DEVICES=1 python start_trainer.py --out_dir "nlp21_meta_default_50_chunk25(kernel32stride4-overlap)" --dataset_path "/mnt/data/hossein/Hossein_workspace/nips_cetra/mahan/CORP/CORP_data_release" --batch_size 16 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --chunk_size 32 --chunk_stride 4

CUDA_VISIBLE_DEVICES=1 python start_trainer.py --out_dir "nlp21_meta_default_50_chunk25(no overlap)" --dataset_path "/mnt/data/hossein/Hossein_workspace/nips_cetra/mahan/CORP/CORP_data_release" --batch_size 16 --epochs 50 --conv_dropout 0.5 --dropout_input 0.2 --do_wandb --chunk_size 25 --chunk_stride 25