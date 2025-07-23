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


log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

logger = logging.getLogger("file1")
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

with open("C:\\Users\\kputt\\Desktop\\study meterial\\MLOps\\6.-Complete-MLFlow-tutorial\\params.yaml", "r") as f:
    params = yaml.safe_load(f)


print(params)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=2)

mlflow.set_experiment("Complete ML Flow tutorial")

with mlflow.start_run():
    rf = RandomForestClassifier(n_estimators=int(params["file1"]["n_estimators"]), max_depth=params["file1"]["max_depth"])
    rf.fit(xtrain, ytrain)
    #logger.debug("RF training done")

    ypred = rf.predict(xtest)

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
    mlflow.log_param("Max_depth",params["file1"]['max_depth'])
    mlflow.log_param("N_estimators",params["file1"]['n_estimators'])
    mlflow.log_artifact(__file__)
    mlflow.log_artifact("confussion_matrix.png")
    mlflow.sklearn.log_model(rf, "RFC")

    #TAGS
    mlflow.set_tags({"Author":"me", "Proj":"Wine_dataset"})

    print("Done")
    

    