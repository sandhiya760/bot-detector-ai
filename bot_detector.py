import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 🔹 Step 1: Load the advanced dataset
df = pd.read_csv('advanced_bot_data.csv')

# 🔍 Optional: Inspect the dataset
print("📊 Dataset Check:")
print("Rows in dataset:", len(df))
print(df.head())

# 🔹 Step 2: Separate features and target
X = df.drop('is_human', axis=1)
y = df['is_human']

# 🔹 Step 3: Split data (stratified to keep human/bot balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 🔹 Step 4: Define and train MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# 🔹 Step 5: Evaluate the model
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("📊 Classification Report:\n", classification_report(y_test, y_pred))

# 🔹 Step 6: Save the trained model
joblib.dump(model, 'bot_model.pkl')
print("✅ Model saved as 'bot_model.pkl'")

