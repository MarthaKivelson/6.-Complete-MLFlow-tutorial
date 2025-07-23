from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import os
import logging
from sklearn.metrics import accuracy_score
import yaml
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import GridSearchCV
import dagshub

#dagshub.init(repo_owner="MarthaKivelson", repo_name="Complete_MLFlow_tutorial", mlflow=True)

#mlflow.set_tracking_uri("https://dagshub.com/MarthaKivelson/6.-Complete-MLFlow-tutorial.mlflow/")

log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

logger = logging.getLogger("file2")
logger.setLevel("DEBUG")

consolehandler = logging.StreamHandler()
consolehandler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
consolehandler.setFormatter(formatter)

logger.addHandler(consolehandler)

data = load_wine()
x = data.data
y = data.target
#logger.debug(f"Input\n {x}\n, \n Output{y}")

rf = RandomForestClassifier(random_state=42)

with open("C:\\Users\\kputt\\Desktop\\study meterial\\MLOps\\6.-Complete-MLFlow-tutorial\\params_grid.yaml", "r") as f:
    params = yaml.safe_load(f)

gscv = GridSearchCV(estimator=rf, param_grid=params, cv=5, n_jobs=-1, verbose=2)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=2)

mlflow.set_experiment("Complete-ML-Flow-tutorial-hp")


with mlflow.start_run():

    gscv.fit(xtrain, ytrain)
    logger.debug("GSCV done")
    best_params = gscv.best_params_
    best_score = gscv.best_score_



    ypred = gscv.predict(xtest)

    acc = accuracy_score(ytest, ypred)
    #logger.debug("Accuracy score is", acc)

    cm = confusion_matrix(ytest, ypred)
    plt.figure(figsize=(10,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Confusion matrics")

    plt.savefig("confussion_matrix.png")



    mlflow.log_metric("Accuracy", acc)
    mlflow.log_param("Max_depth",params['max_depth'])
    mlflow.log_param("N_estimators",params['n_estimators'])
    mlflow.log_artifact(__file__)
    mlflow.log_artifact("confussion_matrix.png")
    joblib.dump(rf, "rf_model.pkl")
    mlflow.log_artifact("rf_model.pkl")

    #TAGS
    mlflow.set_tags({"Author":"me", "Proj":"Wine_dataset"})

    print(best_params, best_score)


