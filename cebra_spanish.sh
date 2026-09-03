python -m brain2qwerty_v1.start_cebra.py --out_dir small-cebra-default-CebUnfold_True --no_noise --bidir --cebra_unfolder --gru --offset 4 --batchSize 64  --random_offset --hidden 1024 --dropout 0.4 --layers 5 --nBatch 12000 --kernel 25 --stride 4 --seed 5 --do_wandb --gradClipValue 10.0

python -m brain2qwerty_v1.start_cebra.py --out_dir small-cebra-default-CebUnfold_False --no_noise --bidir --gru --offset 4 --batchSize 64  --random_offset --hidden 1024 --dropout 0.4 --layers 5 --nBatch 12000 --kernel 25 --stride 4 --seed 5 --do_wandb --gradClipValue 10.0 


python -m brain2qwerty_v1.start_cebra.py --out_dir small-cebra-default-CebUnfold_True_wSmooth --no_noise --bidir --cebra_unfolder --gru --offset 4 --batchSize 64  --random_offset --hidden 1024 --dropout 0.4 --layers 5 --nBatch 12000 --kernel 25 --stride 4 --seed 5 --do_wandb --gradClipValue 10.0 --gauss_in

python -m brain2qwerty_v1.start_cebra.py --out_dir small-cebra-default-CebUnfold_False_wSmooth --no_noise --bidir --gru --offset 4 --batchSize 64  --random_offset --hidden 1024 --dropout 0.4 --layers 5 --nBatch 12000 --kernel 25 --stride 4 --seed 5 --do_wandb --gradClipValue 10.0 --gauss_in


