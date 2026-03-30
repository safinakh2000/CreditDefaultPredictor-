import yaml
import joblib
from preprocess import load_and_preprocess

def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)

def evaluate():
    cfg = load_config()
    X, y = load_and_preprocess()

    print("✅ Loading model...")
    model = joblib.load(cfg["paths"]["model_registry"])

    accuracy = model.score(X, y)
    print(f"✅ Full Dataset Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    evaluate()
