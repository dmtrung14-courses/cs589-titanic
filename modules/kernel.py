import os
from matplotlib import pyplot as plt
import numpy as np 
import pandas as pd
import pickle

from sklearn.svm import SVC
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV, LearningCurveDisplay, learning_curve, train_test_split

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

# Accelerate the program if you are running on intel machines
from sklearnex import patch_sklearn

from .preprocessing import load_data 

patch_sklearn()

parameters = ["Pclass", "TitleLady", "TitleMs", "TitleRev", "TitleMr", "TitleOfficer", "TitleDr", "TitleMrs", "TitleSir", "SexMale", "SexFemale", "Age", "SibSp", "Parch", "Fare", "EmbS", "EmbC", "EmbQ"]

def hinge_loss(y_true, y_pred):
  """Custom hinge loss function for SVM"""
  hinge_loss = np.maximum(0, 1 - y_true * y_pred)
  return np.mean(hinge_loss)

@ignore_warnings(category=ConvergenceWarning)
def kernel_model(x_train, x_test, y_train, y_test):
    model_svm = SVC(max_iter=2000, tol=1e-5)
    # Hyperparameter tuning
    param_grid_svm = {
        "kernel": ["poly", "rbf", "linear"],
        "C": [10**i for i in range(-2, 3)],
        "gamma": [10**i for i in range(-2, 3)],
        "degree": [2, 3, 4]
    }

    clf_svm = GridSearchCV(model_svm, param_grid_svm, scoring="accuracy", refit=True)
    clf_svm.fit(x_train, y_train)

    y_pred_svm = clf_svm.best_estimator_.predict(x_test)

    print("SVM F1 score:", f1_score(y_test, y_pred_svm))
    print("SVM ROC AUC score:", roc_auc_score(y_test, y_pred_svm))
    print("SVM Accuracy score:", accuracy_score(y_test, y_pred_svm))
    print("SVM Best params:", clf_svm.best_params_)


    # Save the model - Make sure the directory exists
    if not(os.path.exists('models/ckpt/')):
        os.makedirs('models/ckpt/')
    if not(os.path.exists('models/res/')):
        os.makedirs('models/res/')
    if not(os.path.exists('models/plt/')):
        os.makedirs('models/plt/')

    with open('models/ckpt/kernel_model.pkl', 'wb') as f:
        pickle.dump(clf_svm, f)

    train_sizes, train_score, test_scores_svm = learning_curve(clf_svm.best_estimator_, x_train, y_train, scoring='accuracy', return_times=False, train_sizes=np.linspace(0.1, 1.0, 100))
    # plot the thing
    plt.figure()
    plt.title("SVM Learning Curve")
    plt.xlabel("Training Examples")
    plt.ylabel("Score")
    # plt.grid()
    plt.plot(train_sizes, test_scores_svm.mean(axis=1), color="r", label="Cross-validation score")
    plt.legend(loc="best")
    plt.savefig('models/plt/kernel.png')

    

    

    return clf_svm.best_params_

def kernel_infer(x_anon, retrain=False):
    if not(os.path.exists('models/ckpt/')):
        os.makedirs('models/ckpt/')
    if retrain or (not os.path.exists('models/ckpt/kernel_model.pkl')):
        x, y = load_data()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        kernel_model(x_train, x_test, y_train, y_test)
    
    with open('models/ckpt/kernel_model.pkl', 'rb') as f:
        clf_svm = pickle.load(f)
    # Predict on x_anon
    y_pred_anon = clf_svm.predict(x_anon)
    
    # Save prediction to CSV file
    passenger_ids = np.arange(892, 1310)
    df_pred = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': y_pred_anon})
    df_pred.to_csv('models/res/kernel.csv', index=False)
    
    plt.savefig('models/plt/kernel_best_params.png')


    

    return clf_svm.best_params_

def kernel_infer(x_anon, retrain=False):
    if retrain or (not os.path.exists('models/ckpt/kernel_model.pkl')):
        x, y = load_data()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        kernel_model(x_train, x_test, y_train, y_test)
    
    with open('models/ckpt/kernel_model.pkl', 'rb') as f:
        clf_svm = pickle.load(f)
    # Predict on x_anon
    y_pred_anon = clf_svm.predict(x_anon)
    
    # Save prediction to CSV file
    passenger_ids = np.arange(892, 1310)
    df_pred = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': y_pred_anon})
    df_pred.to_csv('models/res/kernel.csv', index=False)
