import pandas as pd
import numpy as np
import ast
import os
from tqdm import tqdm


def parse_literal(x):
    if pd.isna(x):
        return np.nan

    if isinstance(x, (dict, list)):
        return x

    if isinstance(x, str):
        x = x.strip()

        if x.startswith(("{", "[")):
            try:
                return ast.literal_eval(x)
            except:
                return x

    return x


def extract_cpu_mem(req):
    req = parse_literal(req)

    cpu, mem = np.nan, np.nan

    if isinstance(req, dict):
        cpu = float(req.get("cpus", np.nan))
        mem = float(req.get("memory", np.nan))

    return cpu, mem


def usage_cpu(x):
    x = parse_literal(x)

    if isinstance(x, dict):
        return float(x.get("cpus", np.nan))

    return np.nan


def preprocess_data(input_file="borg_traces_data.csv"):

    print("\n========== DATA PREPROCESSING ==========")

    # Load original dataset
    df = pd.read_csv(input_file)

    print("Loaded:", df.shape)

    tqdm.pandas()

    # Remove unnecessary column
    df.drop(
        columns=["Unnamed: 0"],
        inplace=True,
        errors="ignore"
    )

    # Convert numeric columns
    df["priority"] = pd.to_numeric(
        df["priority"],
        errors="coerce"
    ).fillna(0).astype("int32")

    df["scheduling_class"] = pd.to_numeric(
        df["scheduling_class"],
        errors="coerce"
    ).fillna(0).astype("int32")

    df["cluster"] = pd.to_numeric(
        df["cluster"],
        errors="coerce"
    ).fillna(0).astype("int16")

    df["failed"] = df["failed"].fillna(0).astype("int8")

    # Extract CPU and memory requests
    cpu_vals, mem_vals = zip(
        *df["resource_request"].progress_apply(extract_cpu_mem)
    )

    df["req_cpu"] = cpu_vals
    df["req_memory"] = mem_vals

    df["req_cpu"] = pd.to_numeric(
        df["req_cpu"],
        errors="coerce"
    ).fillna(0)

    df["req_memory"] = pd.to_numeric(
        df["req_memory"],
        errors="coerce"
    ).fillna(0)

    # Extract CPU usage
    df["avg_cpu_usage"] = (
        df["average_usage"]
        .progress_apply(usage_cpu)
        .fillna(0)
    )

    df["max_cpu_usage"] = (
        df["maximum_usage"]
        .progress_apply(usage_cpu)
        .fillna(0)
    )

    # Convert categorical columns
    df["event"] = df["event"].astype(str)
    df["machine_id"] = df["machine_id"].astype("category")

    # Select important features
    important_features = [
        "priority",
        "scheduling_class",
        "req_cpu",
        "req_memory",
        "avg_cpu_usage",
        "max_cpu_usage",
        "cluster",
        "machine_id",
        "event",
        "failed"
    ]

    df_clean = df[important_features].copy()

    # Generate Edge/Cloud label
    df_clean["label_edge_cloud"] = 0

    edge_condition = (
        (df_clean["req_cpu"] <= 0.02) &
        (df_clean["req_memory"] <= 0.01) &
        (df_clean["avg_cpu_usage"] <= 0.01) &
        (df_clean["priority"] >= 200)
    )

    df_clean.loc[
        edge_condition,
        "label_edge_cloud"
    ] = 1

    # Label names
    label_map = {
        0: "cloud",
        1: "edge"
    }

    df_clean["label_name"] = (
        df_clean["label_edge_cloud"].map(label_map)
    )

    # Create output folder
    os.makedirs("preprocessing", exist_ok=True)

    # Save processed workload
    output_file = "preprocessing/processed_workload.csv"

    df_clean.to_csv(
        output_file,
        index=False
    )

    print("\nLabel distribution:")
    print(df_clean["label_edge_cloud"].value_counts())

    print("\nProcessed dataset:", df_clean.shape)
    print("Saved:", output_file)

    print("\n========== PREPROCESSING COMPLETE ==========\n")

    return df_clean


if __name__ == "__main__":
    preprocess_data()