import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, roc_curve, auc
import warnings
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings('ignore')

# --------------------- Data Preparation ----------------------------

# Read the Dataset
TRAIN_PATH = os.path.join(os.getcwd(), 'health_care_processed2.csv')
df = pd.read_csv(TRAIN_PATH)

x = df[['mental_health_score','medication','exercise_per_week','age','bmi','liver_function','blood_sugar','smoker','diabetes','diagnosis','hospital_stay_days','hospital_visits']]
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
def train_model(X_train, y_train, plot_name, n_estimators: int, max_depth: int, class_weight=None):
    mlflow.set_experiment('treatment_outcome')
    with mlflow.start_run():
        mlflow.set_tag('clf', 'random_forest')

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
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
        mlflow.log_params({'n_estimators': n_estimators, 'max_depth': max_depth})
        mlflow.log_metrics({'accuracy': acc_test, 'f1-score': f1_test})

        input_example = X_train.iloc[0].values.reshape(1, -1)
        mlflow.sklearn.log_model(clf, f'{clf.__class__.__name__}/{plot_name}',
                                 input_example=input_example)

        # Confusion Matrix
        plt.figure(figsize=(10, 6))
        sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True,
                    cbar=False, fmt='.2f', cmap='Blues')
        plt.title(plot_name)
        plt.xticks(ticks=np.arange(3) + 0.5, labels=[0, 1, 2])
        plt.yticks(ticks=np.arange(3) + 0.5, labels=[0, 1, 2])
        mlflow.log_figure(plt.gcf(), f'{plot_name}_conf_matrix.png')
        plt.close()

        # ROC Curve (only for binary classification)
        if len(np.unique(y_test)) == 2:
            fpr, tpr, _ = roc_curve(y_test, y_pred_test, pos_label=1)
            roc_auc = auc(fpr, tpr)

            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                     label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            mlflow.log_figure(plt.gcf(), f'{plot_name}_roc_curve.png')
            plt.close()

# ---------------- Main Function ----------------

def main(n_estimators: int, max_depth: int):
    # 1. Without handling imbalance
    train_model(X_train=X_train, y_train=y_train, plot_name='without-imbalance',
                n_estimators=n_estimators, max_depth=max_depth, class_weight=None)

    # 2. With class weights
    train_model(X_train=X_train, y_train=y_train, plot_name='with-class-weights',
                n_estimators=n_estimators, max_depth=max_depth, class_weight=dict_weights)

    # 3. With SMOTE oversampling
    train_model(X_train=X_train_resampled, y_train=y_train_resampled, plot_name='with-SMOTE',
                n_estimators=n_estimators, max_depth=max_depth, class_weight=None)

# ---------------- Entry Point ----------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', '-n', type=int, default=100)
    parser.add_argument('--max_depth', '-d', type=int, default=10)
    args = parser.parse_args()

    main(n_estimators=args.n_estimators, max_depth=args.max_depth)