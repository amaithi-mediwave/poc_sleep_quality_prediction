import pandas as pd


# set_config(transform_output="pandas")

def hyperparameter_tuning(x_train, y_train):
    
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.ensemble import RandomForestClassifier
    
    classifier = RandomForestClassifier()

    param_grid = {
        "bootstrap": [True],
        "max_depth": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        "max_features": ["sqrt", "log2"],
        "min_samples_leaf": [1, 2, 4],
        "min_samples_split": [2, 5, 8, 13],
        "n_estimators": [20, 30, 40, 50, 100, 200, 400, 600, 800, 1000],
    }

    hyper_tuner = RandomizedSearchCV(
        estimator=classifier,
        param_distributions=param_grid,
        n_jobs=-1,
        cv=5,
        n_iter=100,
        verbose=2,
        scoring="accuracy",
    )

    hyper_tuner.fit(x_train, y_train)

    print("\n Hyperparameter Tuning Initiated \n")

    print("Best Parameters: ", hyper_tuner.best_params_, "\n")

    best_params = hyper_tuner.best_params_

    print("\nFinished HYPERPARAMETER TUNING\n")

    return best_params
