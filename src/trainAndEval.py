import os
import pandas
import warnings

warnings.filterwarnings("ignore")
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime as dt

from getData import read_params
import argparse
import joblib
import json

import mlflow
from urllib.parse import urlparse

from modelTrainingAndHyperTuning import hyperparameter_tuning

dt_now = dt.now()
experi_time = dt_now.strftime("%m/%d/%Y")
run_time = dt_now.strftime("%m/%d/%Y, %H:%M:%S")

# -------------------PREDICTION METRICS---------------------------


def predict_on_test_data(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred


def predict_prob(model, X_test):
    proba = model.predict_proba(X_test)
    return proba


def get_metrics(y_true, y_pred, y_probas):
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_score,
        recall_score,
        ConfusionMatrixDisplay,
    )
    import scikitplot as skplt
    from matplotlib import pyplot as plt

    acc_score = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="micro")
    recall = recall_score(y_true, y_pred, average="micro")
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.savefig('confusion_matrix.png')
    # cm_dis.figure_.savefig('confusion_matrix.png')
    
    skplt.metrics.plot_roc(y_true, y_probas)
    plt.savefig('ROC_Curves.png')
    
    return {
        "Accuracy": round(acc_score, 3),
        "Precision": round(prec, 3),
        "Recall": round(recall, 3),
    }



# -----------------------------------------------------------------------


def train_and_evaluate(config_path):
    config = read_params(config_path)
    pre_processed_data_path = config["data_source"]["local_data_source"][
        "pre_processed_data"
    ]
    test_size = config["base"]["test_size"]
    # model_dir = config["model_dir"]

    
    random_seed = config["base"]["random_seed"]

    df = pd.read_csv(pre_processed_data_path, sep=",")

    X = df.drop(columns=["sleep_disorder"])
    y = df["sleep_disorder"]

    # y.to_csv('for_streamlit_y.csv')
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )

    # ----------------ML FLOW-------------------

    ml_flow_config = config["ml_flow_config"]
    remote_server_uri = ml_flow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)

    mlflow.set_experiment(f"{ml_flow_config['experiment_name']} {experi_time}")

    with mlflow.start_run(
        run_name=f"{ml_flow_config['run_name']} {run_time}") as mlops_run:
        
        best_params = hyperparameter_tuning(X_train, y_train)  # Hyper Parameter Tuning
        # best_params = {'n_estimators': 101, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 30, 'bootstrap': False}

        bootstrap = best_params["bootstrap"]
        max_depth = best_params["max_depth"]
        max_features = best_params["max_features"]
        min_samples_leaf = best_params["min_samples_leaf"]
        min_samples_split = best_params["min_samples_split"]
        n_estimators = best_params["n_estimators"]

        model_tuned = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            max_depth=max_depth,
            bootstrap=bootstrap,
        )
        model_tuned.fit(X_train, y_train)

        y_pred = predict_on_test_data(model_tuned, X_test)
        y_probas = predict_prob(model_tuned, X_test)
        metrics = get_metrics(y_test, y_pred, y_probas)


        # {'accuracy': round(acc, 2), 'precision': round(prec, 2), 'recall': round(recall, 2), 'entropy': round(entropy, 2)}
        # -------------------------------------------------
        # Log Parameters and Metrics
        for param in best_params:
            mlflow.log_param(param, best_params[param])

        for key, val in metrics.items():
            mlflow.log_metric(key, val)

        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                model_tuned,
                "RF Classifier",
                registered_model_name=ml_flow_config["registered_model_name"],
            )

        else:
            mlflow.sklearn.load_model(model_tuned, "RF Classifier")

        if not config["metrics_path"]["roc_auc"] == None:
            mlflow.log_artifact(
                config["metrics_path"]["roc_auc"], "Roc curves")
            mlflow.log_artifact(
                config["metrics_path"]["confusion_mat"], "Confusion matrix")
            

        print("\nFinished Train & Eval and Logged ML Flow Registry\n")

    # -------------------------------------------------


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(parsed_args.config)

