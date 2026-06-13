import random
import pandas as pd
import os

random.seed(42)  # same results har baar

MATERIALS  = ["Aluminum", "Steel", "Titanium", "Brass", "Plastic"]
FEATURES   = ["Hole", "Slot", "Thread", "Pocket", "Groove", "Chamfer"]
TOLERANCES = ["0.005mm", "0.01mm", "0.02mm", "0.05mm", "0.1mm"]

OPERATION_MAP = {
    "Aluminum": ["Facing", "Drilling", "Milling", "Inspection"],
    "Steel":    ["Facing", "Center_Drilling", "Drilling", "Reaming", "Inspection"],
    "Titanium": ["Facing", "Drilling", "Grinding", "Inspection"],
    "Brass":    ["Turning", "Threading", "Chamfering", "Inspection"],
    "Plastic":  ["Milling", "Drilling", "Deburring", "Inspection"],
}

def generate_dataset(n=100):
    rows = []
    for i in range(n):
        material   = random.choice(MATERIALS)
        features   = random.sample(FEATURES, random.randint(1, 3))
        tolerance  = random.choice(TOLERANCES)
        batch_size = random.randint(10, 1000)
        ops = OPERATION_MAP[material].copy()
        # 30% chance ki ek optional operation add ho
        if random.random() > 0.7 and "Reaming" not in ops:
            if "Drilling" in ops:
                idx = ops.index("Inspection")
                ops.insert(idx, "Deburring")
        operations = ops
        rows.append({
            "part_id":    f"P{i+1:03d}",
            "material":   material,
            "features":   ";".join(features),
            "tolerance":  tolerance,
            "batch_size": batch_size,
            "operations": ";".join(operations),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_dataset(100)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/parts_dataset.csv", index=False)
    print(f"Dataset saved: {len(df)} rows")
    print(df.head(5).to_string())
    print("\nMaterial distribution:")
    print(df["material"].value_counts())