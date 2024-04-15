
import numpy as np
import pandas as pd
import glob

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

kd_types = {'EVENT_ATTRIB_ID': int, 'EVENT_ID': int,'ATTRIB_CODE': str,'DOUBLE_VALUE': float,'STRING_VALUE': str, 'EVENT_DATE': str}

kd4_df = pd.concat([pd.read_excel(path, dtype = kd_types ) for path in glob.glob("data/xlsx/src/kd4/*")], axis=0) 

kd4_df['EVENT_DATE'] = pd.to_datetime(kd4_df['EVENT_DATE'])
kd4_df = kd4_df.rename({'EVENT_DATE': 'DATE'}, axis=1)

kd4_df.loc[(kd4_df['ATTRIB_CODE'] == 'CAMERA_NO'), 'STRING_VALUE'] = kd4_df.loc[(kd4_df['ATTRIB_CODE'] == 'CAMERA_NO'), 'STRING_VALUE'].apply(lambda x: f'KD4_{x}')

kd19_df = pd.concat([pd.read_excel(path, dtype = kd_types ) for path in glob.glob("data/xlsx/src/kd19/*")], axis=0) 
kd19_df['EVENT_DATE'] = pd.to_datetime(kd19_df['EVENT_DATE'])
kd19_df = kd19_df.rename({'EVENT_DATE': 'DATE'}, axis=1)

kd19_df.loc[(kd19_df['ATTRIB_CODE'] == 'CAMERA_NO'), 'STRING_VALUE'] = kd19_df.loc[(kd19_df['ATTRIB_CODE'] == 'CAMERA_NO'), 'STRING_VALUE'].apply(lambda x: f'{x}_0')

target_df = pd.concat([pd.read_excel(path) for path in glob.glob("data/xlsx/src/quality/*")], axis=0) 
target_df['Date'] = pd.to_datetime(target_df['Date'], dayfirst=True)

target_df.columns = (target_df.columns
                              .str.replace('(?<=[a-z])(?=[A-Z])', '_', regex=True)
                              .str.upper()
                              .str.removesuffix(', %'))

#target_numeric_columns = target_df.select_dtypes(include=np.number).columns.tolist()
#target_df[target_numeric_columns] = IterativeImputer().fit_transform(target_df[target_numeric_columns])

kd4_df.to_parquet('data/parq/src/kd4.parquet', engine='fastparquet', index=False)
kd19_df.to_parquet('data/parq/src/kd19.parquet', engine='fastparquet', index=False)
target_df.to_parquet('data/parq/src/target.parquet', engine='fastparquet', index=False)

kd4_df.to_csv('data/csv/src/kd4.csv', index=False, header=False)
kd19_df.to_csv('data/csv/src/kd19.csv', index=False, header=False)