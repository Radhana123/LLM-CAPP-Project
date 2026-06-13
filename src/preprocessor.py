import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

class CAPPPreprocessor:

    def __init__(self):
        self.material_encoder = LabelEncoder()
        self.label_encoder    = LabelEncoder()
        self.all_features = [
            "Hole", "Slot", "Thread",
            "Pocket", "Groove", "Chamfer"
        ]
        self.tolerance_map = {
            "0.005mm": 5, "0.01mm": 4, "0.02mm": 3,
            "0.05mm": 2, "0.1mm": 1
        }

    def load_data(self, path="data/parts_dataset.csv"):
        df = pd.read_csv(path)
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    def encode_features(self, df):
        for feat in self.all_features:
            df[f"feat_{feat}"] = df["features"].apply(
                lambda x: 1 if feat in str(x).split(";") else 0
            )
        return df

    def encode_material(self, df):
        df["material_encoded"] = self.material_encoder.fit_transform(df["material"])
        print("Material encoding:")
        for i, mat in enumerate(self.material_encoder.classes_):
            print(f"  {mat} = {i}")
        return df

    def encode_tolerance(self, df):
        df["tolerance_score"] = df["tolerance"].map(self.tolerance_map)
        return df

    def encode_label(self, df):
        df["label"] = self.label_encoder.fit_transform(df["operations"])
        print(f"\nUnique operation sequences: {len(self.label_encoder.classes_)}")
        return df

    def get_X_y(self, df):
        feature_cols = (
            ["material_encoded", "tolerance_score", "batch_size"] +
            [f"feat_{f}" for f in self.all_features]
        )
        X = df[feature_cols].values
        y = df["label"].values
        return X, y, feature_cols

    def preprocess(self, path="data/parts_dataset.csv"):
        df = self.load_data(path)
        df = self.encode_material(df)
        df = self.encode_tolerance(df)
        df = self.encode_features(df)
        df = self.encode_label(df)

        os.makedirs("data", exist_ok=True)
        df.to_csv("data/processed_dataset.csv", index=False)
        print("\nProcessed data saved: data/processed_dataset.csv")

        X, y, cols = self.get_X_y(df)
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print(f"Feature columns: {cols}")
        return X, y, df


if __name__ == "__main__":
    prep = CAPPPreprocessor()
    X, y, df = prep.preprocess()