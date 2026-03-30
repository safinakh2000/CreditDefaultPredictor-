import pandas as pd
import yaml

def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)

def load_and_preprocess():
    cfg = load_config()
    df = pd.read_csv(cfg["paths"]["data"])

    # Drop irrelevant columns
    df = df.drop(columns=cfg["preprocessing"]["drop_columns"], errors="ignore")

    # Encode categorical features if needed
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop(columns=[cfg["preprocessing"]["target"]])
    y = df[cfg["preprocessing"]["target"]]

    return X, y

if __name__ == "__main__":
    X, y = load_and_preprocess()
    print("Preprocess OK — rows:", len(X))
