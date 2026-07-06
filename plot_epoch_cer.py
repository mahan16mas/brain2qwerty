import matplotlib.pyplot as plt
import pickle
import os

datasets = ["nejm", "speech", "nlp10", "nlp21"]
for dataset in datasets:
    folder = f"{dataset}-meta-4"
    with open(f"{folder}/trainingStats", "rb") as file:
        ts = pickle.load(file)
    cer = ts["testCER"]
    plt.plot(cer, label=dataset)
    plt.legend()
    plt.savefig(f"{dataset}.png")
    plt.close()
