`start_trainer.py  --is_speech --dataset_path /data/hossein/data/speech/speech_data_raw_all_in_test.pkl  --out_dir speech-meta --epochs 40 --dropout_input 0.2 --conv_dropout 0.0 
eval_model.py --out_dir speech-meta --dataset_path /data/hossein/data/speech/speech_data_raw_all_in_test.pkl  --is_speech  --conv_zero
start_trainer.py   --dataset_path /data/hossein/mm_project/CORP_data_release  --out_dir nlp21-meta --epochs 40 --dropout_input 0.2 --conv_dropout 0.0 
eval_model.py --out_dir nlp21-meta --dataset_path /data/hossein/mm_project/CORP_data_release    --conv_zero
start_trainer.py --nlp_10  --dataset_path "/data/hossein/mm_project/old_nlp_data/data.npz"  --out_dir nlp10-meta --epochs 40 --dropout_input 0.2 --conv_dropout 0.0 
eval_model.py --out_dir nlp10-meta --dataset_path "/data/hossein/mm_project/old_nlp_data/data.npz" --nlp_10   --conv_zero
start_trainer.py  --is_speech --dataset_path /data/hossein/mm_project/speech_gru_cebra/data/nejm_dataset.pkl --is_nejm --out_dir nejm-meta --epochs 40 --dropout_input 0.2 --conv_dropout 0.0 
eval_model.py --out_dir nejm-meta --dataset_path /data/hossein/mm_project/speech_gru_cebra/data/nejm_dataset.pkl  --is_speech --is_nejm --conv_zero
`
