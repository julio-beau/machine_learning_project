from ucimlrepo import fetch_ucirepo 
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


# fetch dataset 
occupancy_detection = fetch_ucirepo(id=357) 
id_train = list(range(8143))
id_test = list(range(8144, 8144+2665))
X_train = occupancy_detection.data.features.iloc[id_train]
y_train = occupancy_detection.data.targets.iloc[id_train]

X_test = occupancy_detection.data.features.iloc[id_test]
y_test = occupancy_detection.data.targets.iloc[id_test]
X_train["date"] = pd.to_datetime(X_train["date"], errors="coerce")
X_test["date"] = pd.to_datetime(X_test["date"], errors="coerce")
X_train["cos_hour"] = np.cos(
    2 * np.pi * (X_train["date"].dt.hour + X_train["date"].dt.minute / 60) / 24
)
X_train["sin_hour"] = np.sin(
    2 * np.pi * (X_train["date"].dt.hour + X_train["date"].dt.minute / 60) / 24
)
# Position dans la semaine, exprimée en jours décimaux
week_position = (
    X_train["date"].dt.weekday
    + X_train["date"].dt.hour / 24
    + X_train["date"].dt.minute / (24 * 60)
)

X_train["cos_day"] = np.cos(2 * np.pi * week_position / 7)
X_train["sin_day"] = np.sin(2 * np.pi * week_position / 7)

X_test["cos_hour"] = np.cos(
    2 * np.pi * (X_test["date"].dt.hour + X_test["date"].dt.minute / 60) / 24
)
X_test["sin_hour"] = np.sin(
    2 * np.pi * (X_test["date"].dt.hour + X_test["date"].dt.minute / 60) / 24
)
week_position_test = (
    X_test["date"].dt.weekday
    + X_test["date"].dt.hour / 24
    + X_test["date"].dt.minute / (24 * 60)
)
X_test["cos_day"] = np.cos(2 * np.pi * week_position_test / 7)
X_test["sin_day"] = np.sin(2 * np.pi * week_position_test / 7)

decision_tree_model = tree.DecisionTreeClassifier(max_depth=1, random_state=42)
decision_tree_model.fit(X_train[["cos_hour", "sin_hour", "cos_day", "sin_day","Temperature","Light","CO2","Humidity"]], y_train)

y_pred = decision_tree_model.predict(X_test[["cos_hour", "sin_hour", "cos_day", "sin_day","Temperature","Light","CO2","Humidity"]])
print("Decision Tree Model Accuracy: ", accuracy_score(y_test, y_pred))


random_forest_model = RandomForestClassifier(n_estimators=200, criterion='entropy')
random_forest_model.fit(X_train[["cos_hour", "sin_hour", "cos_day", "sin_day","Temperature","Light","CO2","Humidity"]], y_train)
y_pred_rf = random_forest_model.predict(X_test[["cos_hour", "sin_hour", "cos_day", "sin_day","Temperature","Light","CO2","Humidity"]])
print("Random Forest Model Accuracy: ", accuracy_score(y_test, y_pred_rf))
