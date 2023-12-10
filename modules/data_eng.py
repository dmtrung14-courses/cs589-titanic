import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import tensorflow as tf
from sklearn import preprocessing



# lady = {"Dona.", "Lady.", "Madame.", "the Countess."}
# sir = {"Don.", "Jonkheer.", "Sir."}
namemap={"Dona.": "Lady.", "Lady.": "Lady.", "Madame.": "Lady.", 
         "the Countess.": "Lady.", "Don.": "Sir.", "Jonkheer.": "Sir.", 
         "Sir.": "Sir.", "Mr.": "Mr.", "Master.": "Mr.","Miss.": "Ms.",
         "Ms.": "Ms.", "Mrs.": "Mrs.", "Mme.": "Mrs.", "Mlle.": "Ms.", 
         "Capt.": "Officer.", "Col.": "Officer.", "Major.": "Officer."}
titles=['Lady.', 'Ms.', 'Rev.', 'Mr.', 'Officer.', 'Dr.', 'Mrs.', 'Sir.']

def fill_missing_age(group, row):
    if pd.isnull(row['Age']):
        return group.loc[row['Pclass'], row['Survived'], row['Sex']]
    else:
        return row['Age']

def fill_missing_fare(group, row):
    if pd.isnull(row['Fare']):
        return group.loc[row['Pclass'], row['Sex']]
    else:
        return row['Age']

def all_titles(data):
    pattern = r',\s*(.*?\.)'
    result = set()
    for name in data['Name']:
        title = re.search(pattern, name).group(1).strip()
        #grouping the titles
        if title in namemap:
            result.add(namemap[title])
        else:
            result.add(title)
    return list(result)

def names_to_one_hot(name):
    pattern = r',\s*(.*?\.)'
    title = re.search(pattern, name).group(1).strip()
    #grouping the titles
    title = namemap[title] if title in namemap else title
    one_hot= [0] * len(titles)
    one_hot[titles.index(title)] = 1
    return one_hot

def embark_to_onehot(embark):
	return {'S': [1,0,0], 'C': [0,1,0], 'Q': [0,0,1]}[embark]
def gender_to_onehot(gender):
    return {'male': [1, 0], 'female': [0,1]}[gender]

def flatten_rows(data):
    res = []
    for i in range(len(data)):
        row = []
        for j in data[i]:
            if isinstance(j, list):
                row.extend(j)
            else:
                row.append(j)
        res.append(row)
    return np.array(res)

def data_preprocessing(path="./data/", file="train.csv", train = True):
    csv_path = path + file
    data = pd.read_csv(csv_path)
    data = data.drop(['PassengerId', 'Cabin', "Ticket"], axis=1)
    # Preprocess data
    # Fill missing values
    if not train:
        data['Survived'] = 0
    grouped_ages = data.groupby(['Pclass', 'Survived', 'Sex'])['Age'].mean()
    grouped_fares = data.groupby(['Pclass', 'Sex'])['Fare'].mean()
    data['Age'] = data["Age"].fillna(data.apply(lambda row: fill_missing_age(grouped_ages, row), axis=1))
    data['Fare'] = data["Fare"].fillna(data.apply(lambda row: fill_missing_fare(grouped_fares, row), axis=1))
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)


    # Convert categorical data to numerical data
    data['Sex'] = data['Sex'].map(gender_to_onehot)
    data['Embarked'] = data['Embarked'].map(embark_to_onehot)

    data["Name"] = data["Name"].map(lambda name: names_to_one_hot(name))

    # Drop unnecessary columns
    x = data.drop(['Survived'], axis=1)
    y = data['Survived']

    return flatten_rows(np.array(x)), np.array(y)


def printrowswithna():
    path = "./data/test.csv"
    data = pd.read_csv(path)
    # print("age", data[data["Age"].isna()])
    # print("em", data[data["Embarked"].isna()])
    print("f", data[data["Fare"].isna()])
    # print("cab", data[data["Cabin"].isna()])
    print("tick", data[data["Ticket"].isna()])



if __name__ == "__main__":
    x, y = data_preprocessing(train=False, file='test.csv')
    print(x[0])
    print(x.shape)
    print(y.shape)

    # print(x.shape)
    # print(y.shape)