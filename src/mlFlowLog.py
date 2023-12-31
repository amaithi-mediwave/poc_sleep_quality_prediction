from getData import read_params
import argparse 
import mlflow 
from mlflow.tracking import MlflowClient
import joblib 


def log_production_model(config_path):
    
    config = read_params(config_path)
    mlflow_config = config["ml_flow_config"]

    model_name = mlflow_config["registered_model_name"]

    remote_server_uri = mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)

    runs = mlflow.search_runs(search_all_experiments=True) # it will return a dataframe
    
    # runs.to_csv('runs.csv')
    
    high_accuracy = runs["metrics.Accuracy"].sort_values(ascending=False).iloc[0]
    high_accuracy_run_id = runs[runs["metrics.Accuracy"] == high_accuracy]["run_id"].iloc[0]

    client = MlflowClient()
    
    for mv in client.search_model_versions(f"name='{model_name}'"):
        mv = dict(mv)
        
        if mv["run_id"] == high_accuracy_run_id:
            current_version = mv["version"]
            logged_model = mv["source"]
            
            client.transition_model_version_stage(
                name=model_name,
                version=current_version,
                stage="Production",
                    )
        else:
            current_version = mv["version"]
            client.transition_model_version_stage(
                name=model_name,
                version=current_version,
                stage="Staging",
                    )
    loaded_model =mlflow.pyfunc.load_model(logged_model)
    model_path = mlflow_config['production_model_path']
    joblib.dump(loaded_model, model_path)
    print("\nFinished ML FLOW LOG PRODUCTION\n")
    


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    log_production_model(config_path=parsed_args.config)