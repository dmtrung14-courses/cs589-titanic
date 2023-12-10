import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, learning_curve, train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


from sklearnex import patch_sklearn

from .preprocessing import load_data 

patch_sklearn()

parameters = ["Pclass", "TitleLady", "TitleMs", "TitleRev", "TitleMr", "TitleOfficer", "TitleDr", "TitleMrs", "TitleSir", "SexMale", "SexFemale", "Age", "SibSp", "Parch", "Fare", "EmbS", "EmbC", "EmbQ"]


@ignore_warnings(category=ConvergenceWarning)
def neural_model(x_train, x_test, y_train, y_test):
    hidden_layer_sizes = (30, 20, 10,)
    model_nn = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=1000, random_state=42)
    # Hyperparameter tuning
    param_grid_nn = {
        "activation": ["logistic", "tanh", "relu"],
        "solver": ["sgd", "adam"],
        "alpha": [10**i for i in range(-3, 2)],
    }

    clf_nn = GridSearchCV(model_nn, param_grid_nn, scoring="f1")
    clf_nn.fit(x_train, y_train)

    y_pred_nn = clf_nn.predict(x_test)

    print("Neural Network F1 score:", f1_score(y_test, y_pred_nn))
    print("Neural Network ROC AUC score:", roc_auc_score(y_test, y_pred_nn))
    print("Neural Network Accuracy score:", accuracy_score(y_test, y_pred_nn))
    print("Neural Network Best params:", clf_nn.best_params_)

    if not(os.path.exists('models/ckpt/')):
        os.makedirs('models/ckpt/')
    if not(os.path.exists('models/res/')):
        os.makedirs('models/res/')
    if not(os.path.exists('models/plt/')):
        os.makedirs('models/plt/')
    #save the model
    with open('models/ckpt/neural_model.pkl', 'wb') as f:
        pickle.dump(clf_nn, f)

    train_sizes, train_score, test_scores_svm = learning_curve(clf_nn.best_estimator_, x_train, y_train, scoring='accuracy', return_times=False, train_sizes=np.linspace(0.1, 1.0, 100))
    # plot the thing
    plt.figure()
    plt.title("Neural Network Learning Curve")
    plt.xlabel("Training Examples")
    plt.ylabel("Score")
    plt.plot(train_sizes, test_scores_svm.mean(axis=1), color="r", label="Cross-validation score")
    # plt.plot(train_sizes, train_score.mean(axis=1), 'o-', color="b", label="Training score")
    plt.legend(loc="best")
    plt.savefig('models/plt/neural.png')

    # print("Neural Network Final Weights:", len(clf_nn.best_estimator_.coefs_[0]))
    plt.figure()
    plt.title("NN Final Weights")
    plt.xlabel("Parameters")
    plt.ylabel("Frequency")
    plt.bar(parameters, np.mean(clf_nn.best_estimator_.coefs_[0].T, axis = 0))
    plt.xticks(rotation=90)
    plt.savefig('models/plt/neural_best_params.png')

    return clf_nn.best_params_

def neural_infer(x_anon, retrain=False):
    if not(os.path.exists('models/ckpt/')):
        os.makedirs('models/ckpt/')
    if retrain or (not os.path.exists('models/ckpt/neural_model.pkl')):
        x, y = load_data()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        neural_model(x_train, x_test, y_train, y_test)
    
    with open('models/ckpt/neural_model.pkl', 'rb') as f:
        clf_svm = pickle.load(f)
    
    # Predict on x_anon
    y_pred_anon = clf_svm.predict(x_anon)
    
    # Save prediction to CSV file
    passenger_ids = np.arange(892, 1310)
    df_pred = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': y_pred_anon})
    df_pred.to_csv('models/res/neural.csv', index=False)
    
    