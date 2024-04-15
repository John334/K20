import json
from typing import List
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

with open('train/train_info.json', 'r') as f:
    train_info = json.load(f) 
    
feature_cols:List[str] = train_info['feature_cols']

feature_cols = train_info['feature_cols']
target_cols = train_info['target_cols']

df = pd.read_parquet('data/parq/results/result.parquet', engine='fastparquet')

plt.figure(figsize=(20, len(target_cols)))

num_cols = 3
num_rows = len(target_cols) // num_cols + len(target_cols) % num_cols

for i, target in enumerate(target_cols):
    plt.subplot(num_rows, num_cols, i+1)

    sns.lineplot(data=df[[target, *[f'{target}_{model_name}_PRED' for model_name in train_info['models_infos'].keys()] ]] )
    plt.title(f'{target} vs. {target}_PRED')

plt.tight_layout() 
plt.show()