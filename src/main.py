import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sidetrek.types.dataset import SidetrekDataset
from sidetrek.dataset import load_dataset, build_dataset
from dataclasses_json import dataclass_json
from dataclasses import dataclass
from typing import Tuple

@dataclass_json
@dataclass
class Hyperparameters(object):
    source: str = "data/train.csv"
    max_iter: int = 100
    random_state: int = 34
    n_nearest_features: int = 2
    test_size: float = 0.2

hp = Hyperparameters()

# Create dataframe
def get_ds(hp:Hyperparameters)-> pd.DataFrame:
    ds = build_dataset(io="upload", source=hp.source)
    data = load_dataset(ds=ds, data_type="csv")
    data_dict = {}
    cols = list(data)[0]
    for k,v in enumerate(data):
        if k>0:
            data_dict[k]=v
    df = pd.DataFrame.from_dict(data_dict, columns=cols, orient="index")

    return df


# PRE-PROCESSING FUNCTIONS #
# Group the family_size column
def assign_passenger_label(family_size):
    if family_size == 0:
        return "Alone"
    elif family_size <=3:
        return "Small_family"
    else:
        return "Big_family"
    
# Group the Ticket column
def assign_label_ticket(first):
    if first in ["F", "1", "P", "9"]:
        return "Ticket_high"
    elif first in ["S", "C", "2"]:
        return "Ticket_middle"
    else:
        return "Ticket_low"
    
# Group the Title column    
def assign_label_title(title):
    if title in ["the Countess", "Mlle", "Lady", "Ms", "Sir", "Mme", "Mrs", "Miss", "Master"]:
        return "Title_high"
    elif title in ["Major", "Col", "Dr"]:
        return "Title_middle"
    else:
        return "Title_low"
    
# Group the Cabin column  
def assign_label_cabin(cabin):
    if cabin in ["D", "E", "B", "F", "C"]:
        return "Cabin_high"
    elif cabin in ["G", "A"]:
        return "Cabin_middle"
    else:
        return "Cabin_low"



# Pre-process
def preprocess(train: pd.DataFrame, hp: Hyperparameters) -> pd.DataFrame:
    # Replace single spaces with 0 in 'Age'
    train['Age'] = train['Age'].replace('', 0)
    # Drop rows where 'Cabin' is missing
    train = train[train['Cabin'] != '']
    # Define the dictionary mapping column names to their original data types
    data_types = {'PassengerId': int, 'Survived': int, 'Pclass': int, 'Name': str, 'Sex': str,
                'Age': float, 'SibSp': int, 'Parch': int, 'Ticket': str, 'Fare': float,
                'Cabin': str, 'Embarked': str}
    # Convert the columns back to their original data types
    train = train.astype(data_types)
    # Imputers
    imp_embarked = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    imp_age = IterativeImputer(max_iter=hp.max_iter, random_state=hp.random_state, n_nearest_features=hp.n_nearest_features)
    # Impute Embarked
    train["Embarked"] = imp_embarked.fit_transform(train[["Embarked"]])
    
    # Impute Age
    train["Age"] = np.round(imp_age.fit_transform(train[["Age"]]))
    # Initialize a Label Encoder
    le = LabelEncoder()
    # Encode Sex
    train["Sex"] = le.fit_transform(train[["Sex"]].values.ravel())
    # Family Size
    train["Fsize"] = train["SibSp"] + train["Parch"]
    # Ticket first letters
    train["Ticket"] = train["Ticket"].apply(lambda x: str(x)[0])
    # Cabin first letters
    train["Cabin"] = train["Cabin"].apply(lambda x: str(x)[0])
    # Titles
    train["Title"] = train['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
    # Family size
    train["Fsize"] = train["Fsize"].apply(assign_passenger_label)
    # Ticket
    train["Ticket"] = train["Ticket"].apply(assign_label_ticket)
    # Title
    train["Title"] = train["Title"].apply(assign_label_title)
    # Cabin
    train["Cabin"] = train["Cabin"].apply(assign_label_cabin)
    train = pd.get_dummies(columns=["Pclass", "Embarked", "Ticket", "Cabin","Title", "Fsize"], data=train, drop_first=True)
    return train



# Split train and test dataset
def split_ds(df: pd.DataFrame, hp: Hyperparameters) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # Select the features and the target
    X = df.drop(["Survived", "SibSp", "Parch", "Name", "PassengerId"], axis=1)
    y = df["Survived"]   
    # Split the data info training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=hp.test_size, random_state=hp.random_state, stratify=y)
    return X_train, X_test, y_train, y_test


# Train model
def train_model(hp: Hyperparameters, X_train: pd.DataFrame, y_train: pd.Series)->RandomForestClassifier:
    # Initialize a RandomForestClassifier
    rf = RandomForestClassifier(random_state=34)
    params = {'n_estimators': [350],
            'max_depth': [4],
            'min_samples_leaf' : [1],
            'min_samples_split': [20, 25, 30],
            'max_leaf_nodes':[5,6,7],
            }
    clf = GridSearchCV(estimator=rf,param_grid=params,cv=10, n_jobs=-1)
    return clf.fit(X_train, y_train.ravel())


# Main workflow
def main(hp: Hyperparameters):
    df = get_ds(hp=hp)
    df = preprocess(train = df, hp = hp)
    X_train, X_test, y_train, y_test = split_ds(df=df, hp=hp)
    model = train_model(hp=hp, X_train=X_train, y_train=y_train)
    return model


if __name__=="__main__":
    main(hp=hp)

