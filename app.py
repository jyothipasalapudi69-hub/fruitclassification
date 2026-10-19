import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title("🍎 Fruit Classification")
st.write("Predict a fruit using its characteristics")

df = pd.read_csv("fruit.csv")

features = ["weight", "width", "height", "color_score"]
X = df[features]
y = df["fruit_name"]

model = RandomForestClassifier(
    n_estimators=100, random_state=42
)
model.fit(X, y)

st.subheader("Enter Fruit Details")

weight = st.number_input("Weight", min_value=1.0, value=238.0)
width = st.number_input("Width", min_value=1.0, value=8.6)
height = st.number_input("Height", min_value=1.0, value=7.6)
color_score = st.number_input(
    "Color Score", min_value=0.0, max_value=1.0, value=0.41
)

if st.button("Predict Fruit"):
    values = pd.DataFrame(
        [[weight, width, height, color_score]],
        columns=features
    )
    prediction = model.predict(values)[0]
    st.success(f"Predicted Fruit: {prediction.title()}")