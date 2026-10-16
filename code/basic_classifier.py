"""
Le but de ce fichier est d'implémenter des modèles vraiment naïfs, qui serviront de base à battre.
J'en propose 2 : 
1)Disons que l'on trouve que la classe majoritaire est "occupé" à 70%, alors un modèle qui ressort toujours
"occupé" comme prédiction a 70% d'accuracy, prédire en dessous est un TRES mauvais modele de ML
2)Un peu moins naïf mais quand même mauvais en théorie : juste mettre un seuil sur la lumière, comme on a vu que c'était
la meilleure variable
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ucimlrepo import fetch_ucirepo 
from sklearn.metrics import accuracy_score

#Importation des données
occupancy_detection = fetch_ucirepo(id=357) 
X = occupancy_detection.data.features 
y = occupancy_detection.data.targets 
features_names = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
for feature in features_names:
    X[feature] = pd.to_numeric(X[feature], errors="coerce")



### Proportion d'occupé/non-occupé
dataset_len = y.shape[0]
y_occupied  = y[y["Occupancy"]==1.0]
nb_occupied = y_occupied.shape[0]
if nb_occupied/dataset_len >0.5:
    print("Simplest classifier ever : accuracy = ", nb_occupied/dataset_len, " class = occupied")
else:
    print("Simplest classifier ever : accuracy = ", 1-nb_occupied/dataset_len, " class = empty")

#On obtient 76% d'accuracy et on prédit toujours que la pièce est vide, c'est une première base élémentaire pour nos modèles
#On va maintenant fixer un seuil de lumière, au vu du deuxième histogramme du fichier eda.py, un seuil à 400 devrait être bon.

y_pred = (X["Light"] > 400).astype(int)

#ATTENTION: Le print qui suit fait suite à un bug observé : il y a deux NaN dans y
#Il faut donc les benner
print(y["Occupancy"].isna().sum())


# Garder uniquement les lignes où occupancy n'est pas NaN
mask = y["Occupancy"].notna()

# Calculer l'accuracy sur les lignes valides
y_true = y.loc[mask, "Occupancy"]
y_pred_valid = y_pred.loc[mask]

accuracy = accuracy_score(y_true, y_pred_valid)
print("Light threshold model, accuracy : ", accuracy)

#98.75% d'accuracy juste avec ce seuil naïf sur la lumière! Il va falloir avoir des modèles très bons pour justifier leur utilisation
#Je pense qu'on touche la à un point philosophique de la techno : pq utiliser la grosse artillerie et s'embarquer dans les grandes manoeuvres
#alors qu'avec un simple capteur de lumière on sait avec précision si la pièce est occupée, la puissance de calcul déployée dans les grands modèles
#est elle justifiée? Ne doit-on pas se rappeler que le flair humain reste bon pour constuire des choix peu coûteux et malins?
#Jonathan risque de ne pas être complètement d'accord mais si on me demandait ça au taf je m'arrête la.
#SOCIETER #REFLECHIRE