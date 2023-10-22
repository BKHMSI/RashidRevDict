import os
import numpy as np
from glob import glob 
from utils import read_json, write_json
from score import eval_revdict_2
from prettytable import PrettyTable
import itertools
from tqdm import tqdm 
import pandas as pd

# electra_files = sorted(glob("outputs/task1/devset/*-electra/results.json"))
# sgns_files = sorted(glob("outputs/task1/devset/*-sgns/results.json"))

electra_files = sorted(glob("outputs/task1/devset/*-electra/results.json"))
sgns_files = sorted(glob("outputs/task1/devset/*-sgns/results.json"))

model_names = [os.path.dirname(file).split("/")[-1].replace("-electra", "") for file in electra_files]

assert len(electra_files) == len(sgns_files)

electra_embeddings = np.zeros((len(electra_files), 6400, 256))
sgns_embeddings = np.zeros((len(sgns_files), 6400, 300))

for i, (e_file, s_file) in enumerate(zip(electra_files, sgns_files)):
    e_data = read_json(e_file)
    s_data = read_json(s_file)

    assert len(e_data) == 6400
    assert len(s_data) == 6400

    for row_idx, e_row in enumerate(e_data):
        electra_embeddings[i][row_idx] = e_row["electra"]

    for row_idx, s_row in enumerate(s_data):
        sgns_embeddings[i][row_idx] = s_row["sgns"]

indices = np.arange(len(sgns_files))

best_electra = 0
best_electra_subset = ()

best_sgns = 0
best_sgns_subset = ()

latex_table = []
metrics = ["MSE", "Cos", "Rank"]

for L in tqdm(range(len(indices) + 1)):
    for subset in itertools.combinations(indices, L):
        if len(subset) == 0: continue

        subset = np.array(subset)

        electra_embedding_reduced = electra_embeddings[subset].mean(0)
        sgns_embeddings_reduced = sgns_embeddings[subset].mean(0)

        for i, row in enumerate(e_data):
            row["electra"] = list(electra_embedding_reduced[i])
            row["sgns"] = list(sgns_embeddings_reduced[i])

        submission_file = "outputs/task1/training/ensemble.json"
        write_json(submission_file, e_data)
        scores = eval_revdict_2(submission_file, "data/ar.dev.json", "outputs/task1/training/scores.txt")

        table = PrettyTable(field_names=["Model", "MSE", "Cos", "Rank"])
        table.add_row(["Electra", scores[0], scores[2], scores[4]])
        table.add_row(["SGNS", scores[1], scores[3], scores[5]])

        latex_table += [{
            "Models": ','.join([model_names[k] for k in subset]),
            "Electra (MSE)": f"{scores[0]:.4f}",
            "Electra (Cos)": f"{scores[2]:.4f}",
            "Electra (Rank)": f"{scores[4]:.4f}",
            "SGNS (MSE)": f"{scores[1]:.4f}",
            "SGNS (Cos)": f"{scores[3]:.4f}",
            "SGNS (Rank)": f"{scores[5]:.4f}",
        }]


        mean_score_electra = np.mean([1-scores[0], scores[2], 1-scores[4]])
        if mean_score_electra > best_electra:
            best_electra = mean_score_electra
            best_electra_subset = subset

        mean_score_sgns = np.mean([1-scores[1], scores[3], 1-scores[5]])
        if mean_score_sgns > best_sgns:
            best_sgns = mean_score_sgns
            best_sgns_subset = subset

        print()
        print(f"Models {[model_names[j] for j in subset]}")
        print(table)
        print()

print(f"Electra: {best_electra} | {[model_names[j] for j in best_electra_subset]}")
print(f"SGNS: {best_sgns_subset} | {[model_names[j] for j in best_sgns_subset]}")

df = pd.DataFrame(latex_table)
# df = df.groupby(["Models", "Embedding", "Metric", "Score"]).count().reset_index()
df.to_latex("outputs/task1/table.tex", index=False)