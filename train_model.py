import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# Column names
columns = [
    "Pregnancies","Glucose","BloodPressure","SkinThickness",
    "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"
]

# Load dataset
df = pd.read_csv("diabetes.csv", names=columns)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(probability=True))
])

# Train
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model/svm_model.pkl")

print("✅ Model trained & saved!")