from sklearn.datasets import load_breast_cancer     # dataset to be used
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier  # KNN model
from sklearn.tree import DecisionTreeClassifier     # Decision Tree model
from sklearn.ensemble import RandomForestClassifier # Random Forest model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd                                 # used for displaying data in a table
import seaborn as sns                               # used for generating confusion matrix visuals
import matplotlib.pyplot as plt
import numpy as np

# PRE-PROCESSING
# load breast cancer data to be used
breast_cancer = load_breast_cancer()
X, y = breast_cancer.data, breast_cancer.target # extract features, target from dataset

# split dataset into 80% train, 20% test
# set random seed to 0 for reproducibility, ensures each model trains and tests with the exact same sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.20)

# scaling data for KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# MODEL TRAINING
# initialize models, set random seed to 0 for reproducibility
knn = KNeighborsClassifier(n_neighbors=5)
dt = DecisionTreeClassifier(random_state=0)
rf = RandomForestClassifier(n_estimators=100, random_state=0)

# train models with training dataset
knn.fit(X_train_scaled, y_train) # use scaled dataset for KNN
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

# test models i.e. make predictions using features from testing dataset
knn_pred = knn.predict(X_test_scaled) # use scaled dataset for KNN
dt_pred = dt.predict(X_test)
rf_pred = rf.predict(X_test)

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
    'KNN': calc_metrics(y_test, knn_pred),
    'Decision Tree': calc_metrics(y_test, dt_pred),
    'Random Forest': calc_metrics(y_test, rf_pred)
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
    # plt.savefig(f'./matrix/initial/confusion_matrix_{title}.png') # uncomment to download images
    # plt.show()  # uncomment to display images

gen_confusion_matrix(confusion_matrix(y_test, knn_pred), 'KNN')
gen_confusion_matrix(confusion_matrix(y_test, dt_pred), 'Decision Tree')
gen_confusion_matrix(confusion_matrix(y_test, rf_pred), 'Random Forest')