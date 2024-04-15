from datetime import datetime, timedelta
import json
import pickle
import time
import pandas as pd
import schedule
import cx_Oracle
import os
from dotenv import load_dotenv

d = r"D:\instantclient_21_6"
cx_Oracle.init_oracle_client(lib_dir=d)

load_dotenv()

feature_cols = ["GRAN_0_10", "GRAN_10_25", "GRAN_25_40", "GRAN_40_80"]

target_cols = ["М10", "М40"]
models_names = ["LinearRegression", "CatBoost", "XGB", "Stack"]

models_infos = {
    target: {
        model_name: pickle.load(open(f"models/{target}_{model_name}_model.pkl", "rb"))
        for model_name in ["LinearRegression", "CatBoost", "XGB", "Stack"]
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


def filter_and_map_kd_data(df: pd.DataFrame, kd_name: str):
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

    if df.empty:
        raise Exception(f"{kd_name} is empty")

    df = df.mean().to_frame().T
    df[feature_cols] = df[feature_cols].astype(float)

    return df


def get_connection():
    return cx_Oracle.connect(os.getenv("CONNECTION_STRING"))


kd_types = {
    "EVENT_ATTRIB_ID": int,
    "EVENT_ID": int,
    "ATTRIB_CODE": str,
    "STRING_VALUE": str,
    "DOUBLE_VALUE": float,
    "EVENT_DATE": str,
}


def get_data_from_oracle(query, params):
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        output_cursor = cursor.var(cx_Oracle.CURSOR)
        cursor.execute(query, cur=output_cursor, **params)
        result = output_cursor.getvalue()

        df = pd.DataFrame(result.fetchall(), columns=kd_types.keys())
        df = df.astype(kd_types)

        df["EVENT_DATE"] = pd.to_datetime(df["EVENT_DATE"])
        df = df.rename({"EVENT_DATE": "DATE"}, axis=1)
        return df
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_kd4_data_from_oracle(params):
    query_kd4 = "BEGIN :cur := kadp_cv_model.PKG_GRANULO_KD.FNC_SELECT_EVENT_BETWEEN_DATES_KD4(:pBeginDate, :pEndDate); END;"
    return get_data_from_oracle(query_kd4, params)


def get_kd19_data_from_oracle(params):
    query_kd19 = "BEGIN :cur := kadp_cv_model.PKG_GRANULO_K.FNC_SELECT_EVENT_BETWEEN_DATES_K19(:pBeginDate, :pEndDate); END;"
    return get_data_from_oracle(query_kd19, params)


def write_data_to_oracle(result):
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        query = "BEGIN kadp_cv_model.PKG_GRANULO_KD.ins_strength(:p1, :p2, :p3, :p4, :p5, :p6, :p7, :p8); END;"
        cursor.execute(
            query,
            p1=result["М10"]["LinearRegression"],
            p2=result["М10"]["CatBoost"],
            p3=result["М10"]["XGB"],
            p4=result["М40"]["LinearRegression"],
            p5=result["М40"]["CatBoost"],
            p6=result["М40"]["XGB"],
            p7=result["М10"]["Stack"],
            p8=result["М40"]["Stack"],
        )
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


n_minutes = 15
kd_delta = timedelta(minutes=11, seconds=30)


def func():
    try:
        params_kd4 = {
            "pBeginDate": (datetime.now() - timedelta(minutes=n_minutes)),
            "pEndDate": datetime.now(),
        }
        params_kd19 = {
            "pBeginDate": (params_kd4["pBeginDate"] - kd_delta),
            "pEndDate": params_kd4["pEndDate"] - kd_delta,
        }

        for params in [params_kd4, params_kd19]:
            for param in ["pBeginDate", "pEndDate"]:
                params[param] = params[param].strftime("%d-%m-%Y %H:%M:%S")

        print("start get queries")
        kd4_start_df = get_kd4_data_from_oracle(params_kd4)
        kd19_start_df = get_kd19_data_from_oracle(params_kd19)
        print("end get queries")

        print("start filter")
        kd4_df = filter_and_map_kd_data(kd4_start_df, "KD4")
        kd19_df = filter_and_map_kd_data(kd19_start_df, "KD4")
        print("end filter")

        feature_df: pd.Series = (kd4_df - kd19_df).squeeze()

        print("start inference")
        result = predict(feature_df[feature_cols].astype(float))
        print("end inference")

        print(result)

        print("start set query")
        write_data_to_oracle(result)
        print("end set query")

    except Exception as e:
        print(f"Exception: {e}")


schedule.every(n_minutes).seconds.do(func)

while True:
    schedule.run_pending()
    time.sleep(1)
