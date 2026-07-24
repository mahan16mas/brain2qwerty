import os
import pandas as pd
import pickle
import numpy as np

seeds = range(5)
datasets = ["nlp21", "nlp10", "speech", "nejm", ]
ds_names = ["NLP21", "NLP10", "SPEECH24", "SPEECH45"]
borders = [99, 40, 400, 565]

for seed in seeds:
    rows = []
    filename = f"meta_seed{seed}.csv"
    for dataset, ds_name, border in zip(datasets, ds_names, borders):
        is_nlp = "NLP" in ds_name
        dataset_name = dataset
        filename = f"{dataset_name}_seed{seed}.csv"


        def get_mean(err_dist):
            mean = sum([s[0] for s in err_dist]) / sum([s[1] for s in err_dist])
            return mean

        err_dist_path = f"{dataset}-smallx{f'-{seed}' if seed != 0 else ''}/evalStats"
        err_dist = pickle.load(open(err_dist_path, "rb"))
        seen_cer = get_mean(err_dist[:border])
        unseen_cer = get_mean(err_dist[border:])
        mean_cer = get_mean(err_dist)
        rows.append({"experiment_setup":f"NoLM-{'C' if is_nlp else 'P'}ER", "dataset_name":ds_name, "model_name":"META",
                         "seen_test_day_average":seen_cer, "unseen_test_day_average":unseen_cer, "mean_test_day_average":mean_cer,})




    df_new = pd.DataFrame(rows)

    df_new.to_csv(filename, index=False)