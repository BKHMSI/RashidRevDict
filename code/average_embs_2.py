import os
import numpy as np
from glob import glob 
from utils import read_json, write_json

files = glob("outputs/task1/*/revdict-preds-combined-cos.json")

all_sgns = []
all_electra = []

for filepath in files:
    modelname = filepath.split("/")[2]
    if modelname not in ["camelbert-msa", "marbertv2"]:
        continue
    print(f"Model: {modelname}")
    data = read_json(filepath)
    electra = []
    sgns = []
    for row in data:
        electra += [np.array(row["electra"])]
        sgns += [np.array(row["sgns"])]
    all_electra += [np.array(electra)]
    all_sgns += [np.array(sgns)]

electra = np.array(all_electra).mean(axis=0)
sgns = np.array(all_sgns).mean(axis=0)

for i in range(len(data)):
    data[i]["electra"] = list(electra[i])
    data[i]["sgns"] = list(sgns[i])

# python code/score.py --submission_path outputs/task1/ensemble_cos_final.json --reference_files_dir data/ar.dev.json
write_json(f"outputs/task1/ensemble_cos_final.json", data)