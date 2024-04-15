
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

kd4_df = pd.read_parquet('data/parq/src/kd4_mapped.parquet', engine='fastparquet')
kd19_df = pd.read_parquet('data/parq/src/kd19_mapped.parquet', engine='fastparquet')

df = pd.DataFrame({
    'kd4': kd4_df.drop(['DATE'], axis=1).mean(),
    'kd19': kd19_df.drop(['DATE'], axis=1).mean()
}).reset_index().rename({'index': 'Frac'}, axis=1).melt(id_vars='Frac', var_name='Conv', value_name='Value')

sns.barplot(x='Frac', y='Value', hue='Conv', data=df)
plt.show()