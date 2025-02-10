from sklearn.datasets import load_breast_cancer     # dataset to be used
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier  # KNN model
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

# scaling data for KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# MODEL TRAINING
# initialize models, varying the value of K
knn_1 = KNeighborsClassifier(n_neighbors=1)
knn_5 = KNeighborsClassifier(n_neighbors=5)
knn_10 = KNeighborsClassifier(n_neighbors=10)
knn_25 = KNeighborsClassifier(n_neighbors=25)
knn_50 = KNeighborsClassifier(n_neighbors=50)
knn_100 = KNeighborsClassifier(n_neighbors=100)

# train models with training dataset
knn_1.fit(X_train_scaled, y_train) # use scaled dataset for KNN
knn_5.fit(X_train_scaled, y_train)
knn_10.fit(X_train_scaled, y_train)
knn_25.fit(X_train_scaled, y_train)
knn_50.fit(X_train_scaled, y_train)
knn_100.fit(X_train_scaled, y_train)

# test models i.e. make predictions using features from testing dataset
knn_pred_1 = knn_1.predict(X_test_scaled) # use scaled dataset for KNN
knn_pred_5 = knn_5.predict(X_test_scaled)
knn_pred_10 = knn_10.predict(X_test_scaled)
knn_pred_25 = knn_25.predict(X_test_scaled)
knn_pred_50 = knn_50.predict(X_test_scaled)
knn_pred_100 = knn_100.predict(X_test_scaled)

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
    'K = 1': calc_metrics(y_test, knn_pred_1),
    'K = 5': calc_metrics(y_test, knn_pred_5),
    'K = 10': calc_metrics(y_test, knn_pred_10),
    'K = 25': calc_metrics(y_test, knn_pred_25),
    'K = 50': calc_metrics(y_test, knn_pred_50),
    'K = 100': calc_metrics(y_test, knn_pred_100)
    }

print('Results for KNN model, varying K:')
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
    # plt.savefig(f'./matrix/knn_study/confusion_matrix_{title}.png') # uncomment to download images
    # plt.show()

gen_confusion_matrix(confusion_matrix(y_test, knn_pred_1), 'K = 1')
gen_confusion_matrix(confusion_matrix(y_test, knn_pred_5), 'K = 5')
gen_confusion_matrix(confusion_matrix(y_test, knn_pred_10), 'K = 10')
gen_confusion_matrix(confusion_matrix(y_test, knn_pred_25), 'K = 25')
gen_confusion_matrix(confusion_matrix(y_test, knn_pred_50), 'K = 50')
gen_confusion_matrix(confusion_matrix(y_test, knn_pred_100), 'K = 100')
