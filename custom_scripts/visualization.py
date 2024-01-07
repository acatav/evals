import os
import json
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def extract_data_from_filename(filename):
    top_k_match = re.search(r'top_k=(\d+)', filename)
    top_n_match = re.search(r'top_n=(\d+)', filename)

    top_k = int(top_k_match.group(1)) if top_k_match else None
    reranker = "cohere" if "cohere" in filename else "W/O reranker"

    if reranker != 'W/O reranker':
        top_k = int(top_n_match.group(1)) if top_n_match else top_k

    return top_k, reranker


def calculate_score(data):
    counts = data['final_report']
    total_counts = sum(counts.get(f'counts/{l}', 0) for l in ["A", "B", "C", "D", "E"])
    score = (counts.get('counts/A', 0) + counts.get('counts/B', 0)) / total_counts if total_counts else 0
    return score


def process_files(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith('.jsonl'):
            top_k, reranker = extract_data_from_filename(filename)
            with open(os.path.join(directory, filename), 'r') as file:
                for l in file:
                    if "final_report" not in l:
                        continue
                    else:
                        break
                file_data = json.loads(l)
                score = calculate_score(file_data)
                data.append({'top_k': top_k, 'reranker': reranker, 'score': score})
    return pd.DataFrame(data)


df = process_files('../results_nq')
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='top_k', y='score', hue='reranker', linestyle='--', marker='o')
plt.title('Faithfulness vs top_k - NQ')
plt.xlabel('Top K')
plt.ylabel('Faithfulness')
plt.show()
