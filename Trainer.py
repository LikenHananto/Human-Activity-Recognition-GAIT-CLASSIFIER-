import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Function: train_and_evaluate_model
# This function trains a given model on the training data and evaluates it on the test data.
# It prints the classification report and confusion matrix, and saves the confusion matrix plot if a save path is provided.
# It returns the accuracy of the model on the test data.
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, evaluate=True, save_path=None, remap=True):
    # get model name
    model_name = model.__class__.__name__
    print("==========================")
    print(f"⚙️ Training {model_name}...")
    print("==========================")

    # adjustment for xgboost
    if model_name == "XGBClassifier" and remap:
        y_test = y_test - 1
        y_train = y_train - 1

    # Train the model
    model.fit(X_train, y_train)

    if not evaluate:
        return model
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print classification report
    print(classification_report(y_test, y_pred))
    
    # Print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"confusion_matrix_{model_name}.png"))
    plt.clf()

    return accuracy