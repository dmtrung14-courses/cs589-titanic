import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, LearningCurveDisplay, learning_curve
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from sklearnex import patch_sklearn

from .preprocessing import load_data 


patch_sklearn()

parameters = ["Pclass", "TitleLady", "TitleMs", "TitleRev", "TitleMr", "TitleOfficer", "TitleDr", "TitleMrs", "TitleSir", "SexMale", "SexFemale", "Age", "SibSp", "Parch", "Fare", "EmbS", "EmbC", "EmbQ"]


@ignore_warnings(category=ConvergenceWarning)
def tree_model(x_train, x_test, y_train, y_test):
    model_rf = RandomForestClassifier()

    # Hyperparameter tuning
    param_grid_rf = {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15],
        "max_features": ["sqrt", "log2"],
    }

    clf_rf = GridSearchCV(model_rf, param_grid_rf, scoring="accuracy")
    clf_rf.fit(x_train, y_train)

    y_pred_rf = clf_rf.predict(x_test)

    print("Random Forest F1 score:", f1_score(y_test, y_pred_rf))
    print("Random Forest ROC AUC score:", roc_auc_score(y_test, y_pred_rf))
    print("Random Forest accuracy:", accuracy_score(y_test, y_pred_rf))
    print("Random Forest Best params:", clf_rf.best_params_)

    if not(os.path.exists('models/ckpt/')):
        os.makedirs('models/ckpt/')
    if not(os.path.exists('models/res/')):
        os.makedirs('models/res/')
    if not(os.path.exists('models/plt/')):
        os.makedirs('models/plt/')

    #save the model
    with open('models/ckpt/tree_model.pkl', 'wb') as f:
        pickle.dump(clf_rf, f)

    
    train_sizes, train_score, test_scores_svm = learning_curve(clf_rf.best_estimator_, x_train, y_train, scoring='accuracy', return_times=False, train_sizes=np.linspace(0.1, 1.0, 100))
    # plot the thing
    plt.figure()
    plt.title("Random Forest Learning Curve")
    plt.xlabel("Training Examples")
    plt.ylabel("Score")
    plt.plot(train_sizes, test_scores_svm.mean(axis=1), color="r", label="Cross-validation score")
    # plt.plot(train_sizes, train_score.mean(axis=1), 'o-', color="b", label="Training score")
    plt.legend(loc="best")
    plt.savefig('models/plt/tree.png')

    fig, ax = plt.subplots(nrows = 1,ncols = 1, figsize = (100,100))
    tree.plot_tree(clf_rf.best_estimator_.estimators_[0],
                feature_names = parameters, 
                class_names=["Survived", "Not Survived"],
                filled = True);
    fig.savefig('models/plt/rf_individualtree.png')

    return clf_rf.best_params_

def tree_infer(x_anon, retrain=False):
    if not(os.path.exists('models/ckpt/')):
        os.makedirs('models/ckpt/')
    if retrain or (not os.path.exists('models/ckpt/tree_model.pkl')):
        x, y = load_data()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        tree_model(x_train, x_test, y_train, y_test)
    
    with open('models/ckpt/tree_model.pkl', 'rb') as f:
        clf_svm = pickle.load(f)
    # Predict on x_anon
    y_pred_anon = clf_svm.predict(x_anon)
    
    # Save prediction to CSV file
    passenger_ids = np.arange(892, 1310)
    df_pred = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': y_pred_anon})
    df_pred.to_csv('models/res/tree.csv', index=False)