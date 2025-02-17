import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.discriminant_analysis as skl_da
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
file_path = "training_data_vt2025.csv"
df = pd.read_csv(file_path)

# Remove spaces from column names
df.columns = df.columns.str.strip()

# Convert target variable to numeric values
df["increase_stock"] = df["increase_stock"].map({"low_bike_demand": 0, "high_bike_demand": 1})

# Drop missing values
df = df.dropna()

# Select numerical and categorical features
num_features = ["temp", "humidity", "windspeed", "cloudcover"]
cat_features = ["hour_of_day", "day_of_week", "month", "weekday", "summertime"]

# Ensure selected features exist in dataset
num_features = [f for f in num_features if f in df.columns]
cat_features = [f for f in cat_features if f in df.columns]

# One-Hot Encode categorical features
encoder = OneHotEncoder(drop="first", sparse=False)
X_cat_encoded = encoder.fit_transform(df[cat_features])

# Combine numerical and categorical features
X = np.concatenate([df[num_features].values, X_cat_encoded], axis=1)
y = df["increase_stock"]

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train[:, :len(num_features)] = scaler.fit_transform(X_train[:, :len(num_features)])
X_test[:, :len(num_features)] = scaler.transform(X_test[:, :len(num_features)])

# Train LDA model
lda = skl_da.LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# Make predictions
y_pred_lda = lda.predict(X_test)

# Compute accuracy
accuracy_lda = accuracy_score(y_test, y_pred_lda)
print(f"LDA Accuracy: {accuracy_lda:.4f}")

# Compute confusion matrix
conf_matrix_lda = confusion_matrix(y_test, y_pred_lda)
print("\nConfusion Matrix:")
print(pd.DataFrame(conf_matrix_lda, index=["Actual Low", "Actual High"], columns=["Predicted Low", "Predicted High"]))

# Plot confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix_lda, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Low Demand", "High Demand"], 
            yticklabels=["Low Demand", "High Demand"])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("LDA Confusion Matrix")
plt.savefig("lda_confusion_matrix.png")  # Save plot
plt.show()
