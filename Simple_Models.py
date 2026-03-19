import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import for: Logistic Regression, SVM, Random Forest, XGBoost
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from Trainer import train_and_evaluate_model

save_path="Results_Plots"
# Load dataset
# df = preprocess_lab_data(base_path="Labeled", window_size=128, step_size=16, save_path=None)
df = pd.read_csv("Processed_Data\lab_W256_S128.csv")
Y = df['Task']
X = df.drop(columns=['Task'])
# Encode the labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)
# print the label mapping
print("📚 Label mapping:")
for i, label in enumerate(le.classes_):
    print(f"{i}: {label}")
# Split the data into training and testing sets making sure to keep the same ratio of classes in both sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=42)
# convert to numpy arrays
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

# Initialize models
logistic_model = LogisticRegression(max_iter=1000, random_state=42)
svm_model = SVC(random_state=42)
rf_model = RandomForestClassifier(random_state=42)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

logistic_accuracy = train_and_evaluate_model(logistic_model, X_train, y_train, X_test, y_test, evaluate=True, save_path=save_path, remap=False)
svm_accuracy = train_and_evaluate_model(svm_model, X_train, y_train, X_test, y_test, evaluate=True, save_path=save_path, remap=False)
rf_accuracy = train_and_evaluate_model(rf_model, X_train, y_train, X_test, y_test, evaluate=True, save_path=save_path, remap=False)
xgb_accuracy = train_and_evaluate_model(xgb_model, X_train, y_train, X_test, y_test, evaluate=True, save_path=save_path, remap=False)

print(f"\n📊 Final Results:")
print(f"Logistic Regression → Accuracy: {logistic_accuracy:.2f}")
print(f"SVM                 → Accuracy: {svm_accuracy:.2f}")
print(f"Random Forest       → Accuracy: {rf_accuracy:.2f}")
print(f"XGBoost             → Accuracy: {xgb_accuracy:.2f}")


# Plot accuracies
accuracies = [logistic_accuracy, svm_accuracy, rf_accuracy, xgb_accuracy]
models = ["Logistic Regression", "SVM", "Random Forest", "XGBoost"]
plt.figure(figsize=(10, 6))
sns.barplot(x=models, y=accuracies)
# add the accuracy value on top of the bar
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.01, f"{acc:.2f}", ha='center', va='bottom')
plt.title("Model Accuracies")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Results_Plots/model_accuracies.png")
print("==========================")
print(f"Visualization saved as 'Results_Plots/model_accuracies.png'")
print("==========================\n\n")

