import numpy as np
import pandas as pd

def filter_and_map_kd_data(df: pd.DataFrame):
    df = df.copy()
    df = df.pivot_table(index=['EVENT_ID', 'DATE'], columns='ATTRIB_CODE', values=['DOUBLE_VALUE','STRING_VALUE'], aggfunc='first')
    df = df.reset_index()
    
    df = df[[('EVENT_ID', ''), ('DATE', ''), 
             ('DOUBLE_VALUE', 'GRAN_0_10'), 
             ('DOUBLE_VALUE', 'GRAN_10_25'), 
             ('DOUBLE_VALUE', 'GRAN_25_40'),
             ('DOUBLE_VALUE', 'GRAN_40_60'),
             ('DOUBLE_VALUE', 'GRAN_60_80'),
             ('DOUBLE_VALUE', 'GRAN_80_999'), 
             ('STRING_VALUE', 'CAMERA_NO')]]
    
    df.columns = ['EVENT_ID', 'DATE', 'GRAN_0_10', 'GRAN_10_25', 'GRAN_25_40', 'GRAN_40_60', 'GRAN_60_80', 'GRAN_80_999' , 'CAMERA_NO']

    uniq_camers = sorted(pd.unique(df['CAMERA_NO']), reverse=True)
    cameras_index = {cam: i for i, cam in enumerate(uniq_camers)}

    df['EVENT_ID'] = df.apply(lambda x: x['EVENT_ID'] - cameras_index[x['CAMERA_NO']], axis=1)

    df = df.groupby('EVENT_ID').filter(lambda x: len(x) == len(uniq_camers))
    df = df.groupby('EVENT_ID').agg({'GRAN_0_10': 'sum', 'GRAN_10_25': 'sum', 'GRAN_25_40': 'sum', 'GRAN_40_60': 'sum', 'GRAN_60_80': 'sum', 'GRAN_80_999': 'sum', 'DATE': 'mean'})
    df = df.reset_index()

    df = df.drop(['EVENT_ID'], axis=1)

    gran_cols = ['GRAN_0_10', 'GRAN_10_25', 'GRAN_25_40', 'GRAN_40_60', 'GRAN_60_80', 'GRAN_80_999']


    sum_df = df[gran_cols].sum(axis=1)

    df = df[sum_df > 30]

    for col in gran_cols:
        df[col] = (df[col] / sum_df)

    df = df[(df[gran_cols] <= 0.7).all(axis=1)]
    df = df.drop(['GRAN_80_999'], axis=1)

    df['GRAN_40_80'] = df[['GRAN_40_60', 'GRAN_60_80']].sum(axis=1)
    df = df.drop(['GRAN_40_60', 'GRAN_60_80'], axis=1)

    return df

kd4_df = filter_and_map_kd_data(pd.read_parquet('data/parq/src/kd4.parquet', engine='fastparquet'))
kd19_df = filter_and_map_kd_data(pd.read_parquet('data/parq/src/kd19.parquet', engine='fastparquet'))

kd4_df.to_parquet('data/parq/src/kd4_mapped.parquet', engine='fastparquet', index=False)
kd19_df.to_parquet('data/parq/src/kd19_mapped.parquet', engine='fastparquet', index=False)
