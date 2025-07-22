# machine-learning-model
# heart_disease_prediction.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_excel("heart_disease_dataset.xlsx")

# Step A: Data Understanding & Preprocessing
print("Initial shape:", df.shape)
df.drop_duplicates(inplace=True)
print("Shape after removing duplicates:", df.shape)
df.fillna(df.mean(), inplace=True)

# Convert categorical (if needed)
# Already numeric in this case

# Feature Scaling
scaler = StandardScaler()
X = df.drop("target", axis=1)
y = df["target"]
X_scaled = scaler.fit_transform(X)

# Step B: EDA
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

sns.countplot(x='target', data=df)
plt.title("Target Class Distribution")
plt.show()

# Step C: Feature Selection
cor_matrix = df.corr()
top_features = cor_matrix["target"].abs().sort_values(ascending=False)[1:6].index.tolist()
print("Top Features:", top_features)

# Step D: Model Training & Evaluation
X_train, X_test, y_train, y_test = train_test_split(df[top_features], y, test_size=0.3, random_state=42)

# Model 1: Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Model 2: Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Evaluation Function
def evaluate_model(name, y_true, y_pred):
    print(f"\n{name} Evaluation:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d')
    plt.title(f"{name} Confusion Matrix")
    plt.show()

evaluate_model("Logistic Regression", y_test, y_pred_lr)
evaluate_model("Random Forest", y_test, y_pred_rf)

# Bonus: Hyperparameter tuning
# params = {'n_estimators': [10, 50, 100]}
# grid = GridSearchCV(RandomForestClassifier(), params, cv=3)
# grid.fit(X_train, y_train)
# print("Best parameters:", grid.best_params_)
