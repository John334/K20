import json
import pickle
import numpy as np
import pandas as pd


feature_cols = ["GRAN_0_10", "GRAN_10_25", "GRAN_25_40", "GRAN_40_80"]

target_cols = ["М10", "М40"]
models_names = ["LinearRegression", "CatBoost", "XGB", "Stack"]

models_infos = {
    target: {
        model_name: pickle.load(open(f"models/{target}_{model_name}_model.pkl", "rb"))
        for model_name in models_names
    }
    for target in target_cols
}


def predict(row: pd.Series):
    return {
        target: {
            model_name: model.predict(pd.DataFrame(row).T).item()
            for model_name, model in target_models_info.items()
        }
        for target, target_models_info in models_infos.items()
    }


def filter_and_map_kd_data(df: pd.DataFrame):
    df = df.copy()
    df = df.pivot_table(
        index=["EVENT_ID", "DATE"],
        columns="ATTRIB_CODE",
        values=["DOUBLE_VALUE", "STRING_VALUE"],
        aggfunc="first",
    )
    df = df.reset_index()

    df = df[
        [
            ("EVENT_ID", ""),
            ("DATE", ""),
            ("DOUBLE_VALUE", "GRAN_0_10"),
            ("DOUBLE_VALUE", "GRAN_10_25"),
            ("DOUBLE_VALUE", "GRAN_25_40"),
            ("DOUBLE_VALUE", "GRAN_40_60"),
            ("DOUBLE_VALUE", "GRAN_60_80"),
            ("DOUBLE_VALUE", "GRAN_80_999"),
            ("STRING_VALUE", "CAMERA_NO"),
        ]
    ]

    df.columns = [
        "EVENT_ID",
        "DATE",
        "GRAN_0_10",
        "GRAN_10_25",
        "GRAN_25_40",
        "GRAN_40_60",
        "GRAN_60_80",
        "GRAN_80_999",
        "CAMERA_NO",
    ]

    uniq_camers = sorted(pd.unique(df["CAMERA_NO"]), reverse=True)
    cameras_index = {cam: i for i, cam in enumerate(uniq_camers)}

    df["EVENT_ID"] = df.apply(
        lambda x: x["EVENT_ID"] - cameras_index[x["CAMERA_NO"]], axis=1
    )

    df = df.groupby("EVENT_ID").filter(lambda x: len(x) == len(uniq_camers))
    df = df.groupby("EVENT_ID").agg(
        {
            "GRAN_0_10": "sum",
            "GRAN_10_25": "sum",
            "GRAN_25_40": "sum",
            "GRAN_40_60": "sum",
            "GRAN_60_80": "sum",
            "GRAN_80_999": "sum",
            "DATE": "mean",
        }
    )
    df = df.reset_index()

    df = df.drop(["EVENT_ID"], axis=1)

    gran_cols = [
        "GRAN_0_10",
        "GRAN_10_25",
        "GRAN_25_40",
        "GRAN_40_60",
        "GRAN_60_80",
        "GRAN_80_999",
    ]

    sum_df = df[gran_cols].sum(axis=1)

    df = df[sum_df > 30]

    for col in gran_cols:
        df[col] = df[col] / sum_df

    df = df[(df[gran_cols] <= 0.7).all(axis=1)]
    df = df.drop(["GRAN_80_999"], axis=1)

    df["GRAN_40_80"] = df[["GRAN_40_60", "GRAN_60_80"]].sum(axis=1)
    df = df.drop(["GRAN_40_60", "GRAN_60_80"], axis=1)

    df = df.mean().to_frame().T
    df[feature_cols] = df[feature_cols].astype(float)

    return df


kd_types = {
    "EVENT_ATTRIB_ID": int,
    "EVENT_ID": int,
    "ATTRIB_CODE": str,
    "STRING_VALUE": str,
    "DOUBLE_VALUE": float,
    "EVENT_DATE": str,
}

kd4_start_np = pd.read_csv("data/csv/src/kd4.csv", header=None).to_numpy()
kd19_start_np = pd.read_csv("data/csv/src/kd19.csv", header=None).to_numpy()

kd4_start_df = pd.DataFrame(data=kd4_start_np, columns=kd_types.keys())
kd19_start_df = pd.DataFrame(data=kd19_start_np, columns=kd_types.keys())

kd4_start_df = kd4_start_df.astype(kd_types)
kd19_start_df = kd19_start_df.astype(kd_types)

kd4_start_df["EVENT_DATE"] = pd.to_datetime(kd4_start_df["EVENT_DATE"])
kd4_start_df = kd4_start_df.rename({"EVENT_DATE": "DATE"}, axis=1)

kd19_start_df["EVENT_DATE"] = pd.to_datetime(kd19_start_df["EVENT_DATE"])
kd19_start_df = kd19_start_df.rename({"EVENT_DATE": "DATE"}, axis=1)

kd4_df = filter_and_map_kd_data(kd4_start_df)
kd19_df = filter_and_map_kd_data(kd19_start_df)

if kd4_df.empty:
    raise Exception("KD4 is empty")

if kd19_df.empty:
    raise Exception("KD19 is empty")

feature_df: pd.Series = (kd4_df - kd19_df).squeeze()

result = predict(feature_df[feature_cols].astype(float))

p1 = result["М10"]["LinearRegression"]
p2 = result["М10"]["CatBoost"]
p3 = result["М10"]["XGB"]
p4 = result["М40"]["LinearRegression"]
p5 = result["М40"]["CatBoost"]
p6 = result["М40"]["XGB"]
print(p1, p2, p3, p4, p5, p6)
print(result)
