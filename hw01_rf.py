from sklearn.datasets import load_breast_cancer     # dataset to be used
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # Random Forest model
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
# set random seed to 0 for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.20)

# MODEL TRAINING
# set random seed to 0 for reproducibility, isolating the effect of the hyperparameters
# initialize models, varying max depth
rf_depth_none = RandomForestClassifier(random_state=0) # default max_depth=None
rf_depth_1 = RandomForestClassifier(max_depth=1, random_state=0)
rf_depth_3 = RandomForestClassifier(max_depth=3, random_state=0)
rf_depth_5 = RandomForestClassifier(max_depth=5, random_state=0)
rf_depth_10 = RandomForestClassifier(max_depth=10, random_state=0)
rf_depth_25 = RandomForestClassifier(max_depth=25, random_state=0)

# varying min sample split
rf_min_2 = RandomForestClassifier(random_state=0) # default min_samples_split=2
rf_min_5 = RandomForestClassifier(min_samples_split=5, random_state=0)
rf_min_10 = RandomForestClassifier(min_samples_split=10, random_state=0)
rf_min_15 = RandomForestClassifier(min_samples_split=15, random_state=0)
rf_min_25 = RandomForestClassifier(min_samples_split=25, random_state=0)
rf_min_30 = RandomForestClassifier(min_samples_split=30, random_state=0)

# train models with training dataset
rf_depth_none.fit(X_train, y_train)
rf_depth_1.fit(X_train, y_train)
rf_depth_3.fit(X_train, y_train)
rf_depth_5.fit(X_train, y_train)
rf_depth_10.fit(X_train, y_train)
rf_depth_25.fit(X_train, y_train)

rf_min_2.fit(X_train, y_train)
rf_min_5.fit(X_train, y_train)
rf_min_10.fit(X_train, y_train)
rf_min_15.fit(X_train, y_train)
rf_min_25.fit(X_train, y_train)
rf_min_30.fit(X_train, y_train)

# test models i.e. make predictions using features from testing dataset
rf_pred_depth_none = rf_depth_none.predict(X_test)
rf_pred_depth_1 = rf_depth_1.predict(X_test)
rf_pred_depth_3 = rf_depth_3.predict(X_test)
rf_pred_depth_5 = rf_depth_5.predict(X_test)
rf_pred_depth_10 = rf_depth_10.predict(X_test)
rf_pred_depth_25 = rf_depth_25.predict(X_test)

rf_pred_min_2 = rf_min_2.predict(X_test)
rf_pred_min_5 = rf_min_5.predict(X_test)
rf_pred_min_10 = rf_min_10.predict(X_test)
rf_pred_min_15 = rf_min_15.predict(X_test)
rf_pred_min_25 = rf_min_25.predict(X_test)
rf_pred_min_30 = rf_min_30.predict(X_test)

# EVALUATION
# calculate metrics for evaluation
def calc_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1

# store metric results in a dictionary
depth_metrics = {
    'Max Depth = None': calc_metrics(y_test, rf_pred_depth_none),
    'Max Depth = 1': calc_metrics(y_test, rf_pred_depth_1),
    'Max Depth = 3': calc_metrics(y_test, rf_pred_depth_3),
    'Max Depth = 5': calc_metrics(y_test, rf_pred_depth_5),
    'Max Depth = 10': calc_metrics(y_test, rf_pred_depth_10),
    'Max Depth = 25': calc_metrics(y_test, rf_pred_depth_25)
    }

min_metrics = {
    'Min Samples Split = 2': calc_metrics(y_test, rf_pred_min_2),
    'Min Samples Split = 5': calc_metrics(y_test, rf_pred_min_5),
    'Min Samples Split = 10': calc_metrics(y_test, rf_pred_min_10),
    'Min Samples Split = 15': calc_metrics(y_test, rf_pred_min_15),
    'Min Samples Split = 25': calc_metrics(y_test, rf_pred_min_25),
    'Min Samples Split = 30': calc_metrics(y_test, rf_pred_min_30)
}

# display metric results in a pd.DataFrame
depth_metrics_df = pd.DataFrame(depth_metrics, index=['Accuracy', 'Precision', 'Recall', 'F1 Score']).T
print('Results for Random Forest model, varying max depth:')
print(depth_metrics_df)

min_metrics_df = pd.DataFrame(min_metrics, index=['Accuracy', 'Precision', 'Recall', 'F1 Score']).T
print('\nResults for Random Forest model, varying min sample split:')
print(min_metrics_df)

# generate and display confusion matrices for each model
def gen_cm(cm, title):
    plt.figure()
    sns.heatmap(cm/np.sum(cm), fmt=".2%", annot=True, cmap="Blues", xticklabels=["Malignant", "Benign"], yticklabels=["Malignant", "Benign"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.savefig(f'./matrix/rf_study/confusion_matrix_{title}.png') # uncomment to download images
    plt.show() # uncomment to display images

gen_cm(confusion_matrix(y_test, rf_pred_depth_none), 'Max Depth = None')
gen_cm(confusion_matrix(y_test, rf_pred_depth_1), 'Max Depth = 1')
gen_cm(confusion_matrix(y_test, rf_pred_depth_3), 'Max Depth = 3')
gen_cm(confusion_matrix(y_test, rf_pred_depth_5), 'Max Depth = 5')
gen_cm(confusion_matrix(y_test, rf_pred_depth_10), 'Max Depth = 10')
gen_cm(confusion_matrix(y_test, rf_pred_depth_25), 'Max Depth = 25')

gen_cm(confusion_matrix(y_test, rf_pred_min_2), 'Min Samples Split = 2')
gen_cm(confusion_matrix(y_test, rf_pred_min_5), 'Min Samples Split = 5')
gen_cm(confusion_matrix(y_test, rf_pred_min_10), 'Min Samples Split = 10')
gen_cm(confusion_matrix(y_test, rf_pred_min_15), 'Min Samples Split = 15')
gen_cm(confusion_matrix(y_test, rf_pred_min_25), 'Min Samples Split = 25')
gen_cm(confusion_matrix(y_test, rf_pred_min_30), 'Min Samples Split = 30')