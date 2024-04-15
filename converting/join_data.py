import numpy as np
import pandas as pd

kd4_df = pd.read_parquet("data/parq/src/kd4_mapped.parquet", engine="fastparquet")
kd19_df = pd.read_parquet("data/parq/src/kd19_mapped.parquet", engine="fastparquet")
target_df = pd.read_parquet("data/parq/src/target.parquet", engine="fastparquet")

time_delta = pd.Timedelta(minutes=11, seconds=30)

kd4_df["DATE"] = kd4_df["DATE"].dt.round(time_delta)
kd19_df["DATE"] = kd19_df["DATE"].dt.round(time_delta)

kd4_df = kd4_df.sort_values(by=["DATE"], ascending=True)
kd19_df = kd19_df.sort_values(by=["DATE"], ascending=True)

merged_df = pd.merge_asof(
    kd4_df,
    kd19_df,
    on="DATE",
    tolerance=pd.Timedelta(0),
    allow_exact_matches=True,
    suffixes=("_kd4", "_kd19"),
)
merged_df = merged_df.dropna()

feature_df = pd.DataFrame()
feature_df["DATE"] = merged_df["DATE"]
feature_cols = ["GRAN_0_10", "GRAN_10_25", "GRAN_25_40", "GRAN_40_80"]
for col in feature_cols:
    feature_df[col] = merged_df[f"{col}_kd4"] - merged_df[f"{col}_kd19"]

print(feature_df)
# feature_df = feature_df.loc[((feature_df[feature_cols] > 0) == (feature_df[feature_cols].mean() > 0)).all(axis=1)]

feature_df["DATE"] -= pd.Timedelta(hours=4)
feature_df["DATE"] = feature_df["DATE"].dt.floor(pd.Timedelta(days=1))
feature_df["RND_BY_DAY"] = np.random.randint(0, 1, feature_df.shape[0])

feature_df: pd.DataFrame = feature_df.groupby(["DATE", "RND_BY_DAY"]).agg(
    {
        "GRAN_0_10": "mean",
        "GRAN_10_25": "mean",
        "GRAN_25_40": "mean",
        "GRAN_40_80": "mean",
    }
)
feature_df = feature_df.reset_index()
feature_df = feature_df.drop(["RND_BY_DAY"], axis=1)

feature_df = feature_df.sort_values(by=["DATE"], ascending=True)
target_df = target_df.sort_values(by=["DATE"], ascending=True)

df = pd.merge_asof(
    feature_df,
    target_df,
    on="DATE",
    tolerance=pd.Timedelta(0),
    allow_exact_matches=True,
)
df = df.set_index("DATE")
df = df.dropna()

df.to_parquet("data/parq/kd_dataset.parquet", engine="fastparquet")
df.to_excel("data/xlsx/kd_dataset.xlsx")
