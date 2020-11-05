import pandas as pd
import numpy as np
import os
from file_loader import get_dataframes

dataframes = get_dataframes()

# For each dataframe - generating 10 other dfs

for name in dataframes:
    df = pd.read_csv(name, header=None)
    dataset_name = name.split('/')[-1].split('.')[0]
    for i in range(10):
        columns = np.random.choice(df.columns, int(len(df.columns) * 0.7), replace=False)
        samples = np.random.choice(df.index, int(len(df.index) * 0.7,), replace=False)
        new_df = df[columns]
        new_df = new_df.iloc[samples]
        new_df.to_csv(os.path.join(os.getcwd(), 'Generated', f'{dataset_name}_{i}.csv'), index=None)