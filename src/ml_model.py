# ML Model — CAPP System
# Random Forest Classifier Implementation

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessor import CAPPPreprocessor


class CAPPModel:

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
        )
        self.preprocessor = CAPPPreprocessor()

    def train(self, X_train, y_train):
        print("Model training ho raha hai...")
        self.model.fit(X_train, y_train)
        print("Training complete!")

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {acc*100:.2f}%")
        print("\nDetailed Report:")
        print(classification_report(y_test, y_pred))
        return acc

    def predict(self, X):
        return self.model.predict(X)

    def feature_importance(self, feature_names):
        importances = self.model.feature_importances_
        pairs = sorted(zip(feature_names, importances), key=lambda x: -x[1])
        print("\nFeature Importance:")
        for name, imp in pairs:
            bar = "█" * int(imp * 50)
            print(f"  {name:25} {bar} {imp:.3f}")

    def save(self, path="data/capp_model.pkl"):
        os.makedirs("data", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"\nModel saved: {path}")


if __name__ == "__main__":
    prep = CAPPPreprocessor()
    X, y, df = prep.preprocess()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTrain samples: {len(X_train)}")
    print(f"Test samples:  {len(X_test)}")

    model = CAPPModel()
    model.train(X_train, y_train)

    _, _, feature_cols = prep.get_X_y(df)
    acc = model.evaluate(X_test, y_test)
    model.feature_importance(feature_cols)

    model.save()

    print("\n" + "="*50)
        # Random new part test
    print("\nNEW PART PREDICTION TEST:")
    new_part = {
        "material":   "Titanium",   # training mein tha
        "features":   ["Pocket"],
        "tolerance":  "0.005mm",
        "batch_size": 75,
    }

    # Manually encode karo
    mat_classes = list(prep.material_encoder.classes_)
    mat_encoded = mat_classes.index(new_part["material"])
    tol_score   = prep.tolerance_map[new_part["tolerance"]]
    feat_vec    = [1 if f in new_part["features"] else 0 for f in prep.all_features]

    X_new = np.array([[mat_encoded, tol_score, new_part["batch_size"]] + feat_vec])
    pred  = model.predict(X_new)
    pred_label = prep.label_encoder.inverse_transform(pred)[0]

    print(f"  Material:   {new_part['material']}")
    print(f"  Features:   {new_part['features']}")
    print(f"  Tolerance:  {new_part['tolerance']}")
    print(f"  Predicted Operations: {pred_label}")

    print(f"FINAL ACCURACY: {acc*100:.2f}%")
    print("="*50)