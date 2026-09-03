# python -m brain2qwerty_v1.start_cebra --out_dir small-cebra-default-CebUnfold_True --no_noise --bidir --cebra_unfolder --gru --offset 4 --batchSize 64  --random_offset --hidden 1024 --dropout 0.4 --layers 5 --nBatch 20000 --kernel 25 --stride 4 --seed 5 --do_wandb --gradClipValue 10.0

# python -m brain2qwerty_v1.start_cebra --out_dir small-cebra-default-CebUnfold_False --no_noise --bidir --gru --offset 4 --batchSize 64  --random_offset --hidden 1024 --dropout 0.4 --layers 5 --nBatch 20000 --kernel 25 --stride 4 --seed 5 --do_wandb --gradClipValue 10.0 


# python -m brain2qwerty_v1.start_cebra --out_dir small-cebra-default-CebUnfold_True_wSmooth --no_noise --bidir --cebra_unfolder --gru --offset 4 --batchSize 64  --random_offset --hidden 1024 --dropout 0.4 --layers 5 --nBatch 20000 --kernel 25 --stride 4 --seed 5 --do_wandb --gradClipValue 10.0 --gauss_in

# python -m brain2qwerty_v1.start_cebra --out_dir small-cebra-default-CebUnfold_False_wSmooth --no_noise --bidir --gru --offset 4 --batchSize 64  --random_offset --hidden 1024 --dropout 0.4 --layers 5 --nBatch 20000 --kernel 25 --stride 4 --seed 5 --do_wandb --gradClipValue 10.0 --gauss_in


### Adding noise? 
# python -m brain2qwerty_v1.start_cebra --out_dir SpanishBCBL-small-cebra-default-CebUnfold_True-wNoise_0.8_0.2 --bidir --cebra_unfolder --gru --offset 4 --batchSize 64  --random_offset --hidden 1024 --dropout 0.4 --layers 5 --nBatch 10000 --kernel 25 --stride 4 --seed 5 --do_wandb --gradClipValue 10.0 --whiteNoiseSD 0.8 --constantOffsetSD 0.2 

# python -m brain2qwerty_v1.start_cebra --out_dir SpanishBCBL-small-cebra-default-CebUnfold_True-wNoise_0.8_0.2-GaussIn --bidir --cebra_unfolder --gru --offset 4 --batchSize 64  --random_offset --hidden 1024 --dropout 0.4 --layers 5 --nBatch 10000 --kernel 25 --stride 4 --seed 5 --do_wandb --gradClipValue 10.0 --whiteNoiseSD 0.8 --constantOffsetSD 0.2 --gauss_in

python -m brain2qwerty_v1.start_cebra --out_dir SpanishBCBL-small-cebra-default-CebUnfold_True-wNoise_1.2_0.6-GaussIn --bidir --cebra_unfolder --gru --offset 4 --batchSize 64  --random_offset --hidden 1024 --dropout 0.4 --layers 5 --nBatch 12000 --kernel 25 --stride 4 --seed 5 --do_wandb --gradClipValue 10.0 --whiteNoiseSD 1.2 --constantOffsetSD 0.6 --gauss_in

python -m brain2qwerty_v1.start_cebra --out_dir SpanishBCBL-small-cebra-default-CebUnfold_True-wNoise_0.5_0.0 --bidir --cebra_unfolder --gru --offset 4 --batchSize 64  --random_offset --hidden 1024 --dropout 0.4 --layers 5 --nBatch 12000 --kernel 25 --stride 4 --seed 5 --do_wandb --gradClipValue 10.0 --whiteNoiseSD 0.5 --constantOffsetSD 0.0

python -m brain2qwerty_v1.start_cebra --out_dir SpanishBCBL-small-cebra-default-CebUnfold_True-wNoise_0.5_0.2 --bidir --cebra_unfolder --gru --offset 4 --batchSize 64  --random_offset --hidden 1024 --dropout 0.4 --layers 5 --nBatch 12000 --kernel 25 --stride 4 --seed 5 --do_wandb --gradClipValue 10.0 --whiteNoiseSD 0.5 --constantOffsetSD 0.2 

python -m brain2qwerty_v1.start_cebra --out_dir SpanishBCBL-small-cebra-default-CebUnfold_True-wNoise_0.5_0.2-GaussIn --bidir --cebra_unfolder --gru --offset 4 --batchSize 64  --random_offset --hidden 1024 --dropout 0.4 --layers 5 --nBatch 12000 --kernel 25 --stride 4 --seed 5 --do_wandb --gradClipValue 10.0 --whiteNoiseSD 0.5 --constantOffsetSD 0.2 --gauss_in

python -m brain2qwerty_v1.start_cebra --out_dir SpanishBCBL-small-cebra-default-CebUnfold_True-wNoise_0.1_0.05-GaussIn --bidir --cebra_unfolder --gru --offset 4 --batchSize 64  --random_offset --hidden 1024 --dropout 0.4 --layers 5 --nBatch 12000 --kernel 25 --stride 4 --seed 5 --do_wandb --gradClipValue 10.0 --whiteNoiseSD 0.1 --constantOffsetSD 0.05 --gauss_in

python -m brain2qwerty_v1.start_cebra --out_dir SpanishBCBL-small-cebra-default-CebUnfold_True-wNoise_0.1_0.05 --bidir --cebra_unfolder --gru --offset 4 --batchSize 64  --random_offset --hidden 1024 --dropout 0.4 --layers 5 --nBatch 12000 --kernel 25 --stride 4 --seed 5 --do_wandb --gradClipValue 10.0 --whiteNoiseSD 0.1 --constantOffsetSD 0.05

