from sklearn.datasets import load_breast_cancer     # dataset to be used
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier     # Decision Tree model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd                                 # used for displaying data in a table
import seaborn as sns                               # used for generating confusion matrix visuals
import matplotlib.pyplot as plt
import numpy as np

# PRE-PROCESSING
# load breast cancer data to be used
breast_cancer = load_breast_cancer()
X, y = breast_cancer.data, breast_cancer.target # features, target

# split dataset into 80% train, 20% test
# set random seed to 0 for reproducibility, ensure each model trains and tests with the exact same sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.20)

# MODEL TRAINING
# initialize models, varying the value of max_depth, set random seed to 0 for reproducibility
dt_none = DecisionTreeClassifier(random_state=0)
dt_1 = DecisionTreeClassifier(max_depth=1, random_state=0)
dt_3 = DecisionTreeClassifier(max_depth=3, random_state=0)
dt_5 = DecisionTreeClassifier(max_depth=5, random_state=0)
dt_10 = DecisionTreeClassifier(max_depth=10, random_state=0)
dt_25 = DecisionTreeClassifier(max_depth=25, random_state=0)

# train models with training dataset
dt_none.fit(X_train, y_train)
dt_1.fit(X_train, y_train)
dt_3.fit(X_train, y_train)
dt_5.fit(X_train, y_train)
dt_10.fit(X_train, y_train)
dt_25.fit(X_train, y_train)

# test models i.e. make predictions using features from testing dataset
dt_pred_none = dt_none.predict(X_test)
dt_pred_1 = dt_1.predict(X_test)
dt_pred_3 = dt_3.predict(X_test)
dt_pred_5 = dt_5.predict(X_test)
dt_pred_10 = dt_10.predict(X_test)
dt_pred_25 = dt_25.predict(X_test)

# EVALUATION
# calculate metrics for evaluation
def calc_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1

# store metric results in a dictionary
metrics = {
    'Max Depth = None': calc_metrics(y_test, dt_pred_none),
    'Max Depth = 1': calc_metrics(y_test, dt_pred_1),
    'Max Depth = 3': calc_metrics(y_test, dt_pred_3),
    'Max Depth = 5': calc_metrics(y_test, dt_pred_5),
    'Max Depth = 10': calc_metrics(y_test, dt_pred_10),
    'Max Depth = 25': calc_metrics(y_test, dt_pred_25)
    }

# display metric results in a pd.DataFrame
metrics_df = pd.DataFrame(metrics, index=['Accuracy', 'Precision', 'Recall', 'F1 Score']).T
print(metrics_df)

# generate and display confusion matrices for each model
def gen_confusion_matrix(cm, title):
    plt.figure()
    sns.heatmap(cm/np.sum(cm), fmt=".2%", annot=True, cmap="Blues", xticklabels=["Malignant", "Benign"], yticklabels=["Malignant", "Benign"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    # plt.savefig(f'./matrix/dt_study/confusion_matrix_{title}.png') # uncomment to download images
    # plt.show()  # uncomment to display images

gen_confusion_matrix(confusion_matrix(y_test, dt_pred_none), 'Max Depth = None')
gen_confusion_matrix(confusion_matrix(y_test, dt_pred_1), 'Max Depth = 1')
gen_confusion_matrix(confusion_matrix(y_test, dt_pred_3), 'Max Depth = 3')
gen_confusion_matrix(confusion_matrix(y_test, dt_pred_5), 'Max Depth = 5')
gen_confusion_matrix(confusion_matrix(y_test, dt_pred_10), 'Max Depth = 10')
gen_confusion_matrix(confusion_matrix(y_test, dt_pred_25), 'Max Depth = 25')
