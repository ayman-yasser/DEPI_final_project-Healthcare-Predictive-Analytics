import numpy as np
import pandas as pd
import seaborn as sns # type: ignore
import os
import argparse
from imblearn.over_sampling import SMOTE # type: ignore
import mlflow
import mlflow.sklearn
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, roc_curve, auc, classification_report
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from mlflow.models.signature import infer_signature
warnings.filterwarnings('ignore')

# --------------------- Data Preparation ----------------------------

# Read the Dataset
TRAIN_PATH = os.path.join(os.getcwd(), 'health_care_processed2.csv')
df = pd.read_csv(TRAIN_PATH)

x = df[['mental_health_score','exercise_per_week','age','bmi','liver_function','blood_sugar','hospital_stay_days','hospital_visits','medication','smoker','diabetes','diagnosis']]
y = df['treatment_outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=42, stratify=y
)

# --------------------- Handling Imbalance ----------------------------

# Calculate class weights
vals_count = 1 - (np.bincount(y_train) / len(y_train))
vals_count = vals_count / np.sum(vals_count)
dict_weights = {i: vals_count[i] for i in range(3)}  # Assuming 3 classes: 0, 1, 2

# Oversample using SMOTE
smote = SMOTE(sampling_strategy='auto')
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# --------------------- Modeling ----------------------------
def train_model(X_train, y_train, plot_name, max_depth: int, class_weight=None):
    mlflow.set_experiment('treatment_outcome')
    with mlflow.start_run():
        mlflow.set_tag('clf', 'decision_tree')

        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=45,
            class_weight=class_weight
        )
        clf.fit(X_train, y_train)
        y_pred_test = clf.predict(X_test)

        # Metrics
        f1_test = f1_score(y_test, y_pred_test, average='macro')
        acc_test = accuracy_score(y_test, y_pred_test)

        # Log parameters and metrics
        mlflow.log_params({'max_depth': max_depth})
        mlflow.log_metrics({'accuracy': acc_test, 'f1-score': f1_test})
        signature = infer_signature(X_train, clf.predict(X_train))
        input_example = X_train.iloc[0].values.reshape(1, -1)
        mlflow.sklearn.log_model(clf, f'{clf.__class__.__name__}/{plot_name}',
                                 input_example=input_example , signature=signature)

        # ---------------------- Confusion Matrix ----------------------
        plt.figure(figsize=(10, 6))
        sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True,
                    cbar=False, fmt='d', cmap='Blues')
        plt.title(f'{plot_name} - Confusion Matrix')
        plt.xticks(ticks=np.arange(3) + 0.5, labels=[0, 1, 2])
        plt.yticks(ticks=np.arange(3) + 0.5, labels=[0, 1, 2])
        mlflow.log_figure(plt.gcf(), f'{plot_name}_conf_matrix.png')
        plt.close()

        # ---------------------- Feature Importance ----------------------
        importances = clf.feature_importances_
        features = X_train.columns
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(12, 6))
        sns.barplot(x=importances[indices], y=features[indices])
        plt.title(f'{plot_name} - Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        mlflow.log_figure(plt.gcf(), f'{plot_name}_feature_importance.png')
        plt.close()

        # ---------------------- Classification Report ----------------------
        report = classification_report(y_test, y_pred_test, output_dict=True)
        df_report = pd.DataFrame(report).transpose()

        plt.figure(figsize=(10, 6))
        sns.heatmap(df_report.iloc[:-1, :-1], annot=True, cmap='YlGnBu')
        plt.title(f'{plot_name} - Classification Report Heatmap')
        mlflow.log_figure(plt.gcf(), f'{plot_name}_class_report.png')
        plt.close()

# ---------------- Main Function ----------------

def main( max_depth: int):
    # 1. Without handling imbalance
    train_model(X_train=X_train, y_train=y_train, plot_name='without-imbalance',
                 max_depth=max_depth, class_weight=None)

    # 2. With class weights
    train_model(X_train=X_train, y_train=y_train, plot_name='with-class-weights',
                 max_depth=max_depth, class_weight=dict_weights)

    # 3. With SMOTE oversampling
    train_model(X_train=X_train_resampled, y_train=y_train_resampled, plot_name='with-SMOTE',
                 max_depth=max_depth, class_weight=None)

# ---------------- Entry Point ----------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', '-n', type=int, default=100)
    parser.add_argument('--max_depth', '-d', type=int, default=10)
    args = parser.parse_args()
    
    main( max_depth=args.max_depth)