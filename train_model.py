
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv("fruit.csv")

features = ["weight", "width", "height", "color_score"]
X, y = df[features], df["fruit_name"]

Xtr, Xte, ytr, yte = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(Xtr, ytr)

print(classification_report(yte, model.predict(Xte)))

vals = [
    float(input(f"Enter {col}: "))
    for col in features
]

new_fruit = pd.DataFrame([vals], columns=features)
print("Predicted fruit:", model.predict(new_fruit)[0])
