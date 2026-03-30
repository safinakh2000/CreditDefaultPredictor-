import joblib
import yaml
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from preprocess import load_and_preprocess
from utils import save_metadata

def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)

def train():
    cfg = load_config()

    print("✅ Loading & preprocessing data...")
    X, y = load_and_preprocess()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("✅ Training model...")
    params = cfg["model"]["params"]
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"✅ Model Accuracy: {accuracy:.4f}")

    print("✅ Saving model...")
    model_path = cfg["paths"]["model_registry"]
    metadata_path = cfg["paths"]["model_metadata"]

    joblib.dump(model, model_path)

    metadata = {
        "model_type": cfg["model"]["type"],
        "params": params,
        "trained_at": str(datetime.now()),
        "accuracy": accuracy
    }

    save_metadata(metadata_path, metadata)

    print("✅ Model saved successfully.")

if __name__ == "__main__":
    train()
