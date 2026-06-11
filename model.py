import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("titanic.csv")

# Select columns
df = df[['Pclass', 'Sex', 'Age', 'Fare', 'Survived']]

# Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Convert categorical values
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Features and target
X = df[['Pclass', 'Sex', 'Age', 'Fare']]
y = df['Survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Graph 1 - Survival Distribution
plt.figure()
sns.countplot(x='Survived', data=df)
plt.title("Passenger Survival Distribution")
plt.show()

# Graph 2 - Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()