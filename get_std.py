import os
import pandas as pd
import pickle
import numpy as np

datasets = ["nejm", "nlp10", "nlp21", "speech"]
for dataset in datasets:
    dataset_name = dataset
    filename = f"{dataset_name}_pre_lm.csv"


    def get_mean(file_path):
        err_dist = pickle.load(open(file_path, 'rb'))
        mean = sum([s[0] for s in err_dist]) / sum([s[1] for s in err_dist])
        return mean

    name_to_folders = {
        "META":[f"{dataset}-smallx"] + [f"{dataset_name}-smallx-{seed}" for seed in range(1, 5)]
    }

    data = []
    for name, folders in name_to_folders.items():
        scores = []
        for folder in folders:
            file = f"{folder}/{'' if 'Multi' not in name else dataset + '_'}evalStats"
            scores.append(get_mean(file))
        mean = np.mean(scores)
        std = np.std(scores)
        data.append({"model":name, "mean":mean, "std":std})



    df_new = pd.DataFrame(data)

    if os.path.exists(filename):
        df_old = pd.read_csv(filename)
        df_final = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_final = df_new

    df_final.to_csv(filename, index=False)