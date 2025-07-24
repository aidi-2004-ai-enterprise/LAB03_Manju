import pandas as pd
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import logging

# === Set up logging ===
logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

def main():
    # === Load dataset ===
    logger.info("Loading penguins dataset...")
    df = sns.load_dataset("penguins")
    df.dropna(inplace=True)
    logger.info(f"Dataset loaded: {len(df)} rows")

    # === Encode target ===
    le = LabelEncoder()
    df["species"] = le.fit_transform(df["species"])

    # === One-hot encode categorical features ===
    df = pd.get_dummies(df, columns=["sex", "island"])
    logger.info(f"Columns after encoding: {list(df.columns)}")

    # === Split dataset ===
    X = df.drop("species", axis=1)
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    logger.info(f"Training set size: {X_train.shape}")
    logger.info(f"Test set size: {X_test.shape}")

    # === Train XGBoost classifier ===
    logger.info("Training XGBoost classifier...")
    model = XGBClassifier(
        max_depth=3,
        n_estimators=100,
        use_label_encoder=False,
        eval_metric="mlogloss"
    )
    model.fit(X_train, y_train)

    # === Evaluate model ===
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_f1 = f1_score(y_train, train_pred, average='macro')
    test_f1 = f1_score(y_test, test_pred, average='macro')
    logger.info(f"Train F1 Score: {train_f1:.4f}")
    logger.info(f"Test F1 Score: {test_f1:.4f}")

    # === Save the model ===
    output_path = "app/data/model.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save_model(output_path)
    logger.info(f"Model saved to: {output_path}")
    logger.info("Training pipeline completed successfully!")

if __name__ == "__main__":
    main()
