
from ucimlrepo import fetch_ucirepo 
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
# fetch dataset 
occupancy_detection = fetch_ucirepo(id=357) 

occupancy_detection = fetch_ucirepo(id=357)
df = occupancy_detection.data.original.copy()
print(df["date"].iloc[8140:8151])
# Supprimer d'éventuelles lignes d'en-tête parasites
df = df[df["date"] != "date"]
df = df[df["Temperature"] != "Temperature"]

# Conversions de types
df["date"] = pd.to_datetime(df["date"], errors="coerce")
for c in ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio", "Occupancy"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["date"]).reset_index(drop=True)
print(df.dtypes)
print(df.shape)

X =df.copy()


X["cos_hour"] = np.cos(
    2 * np.pi * (X["date"].dt.hour + X["date"].dt.minute / 60) / 24
)
X["sin_hour"] = np.sin(
    2 * np.pi * (X["date"].dt.hour + X["date"].dt.minute / 60) / 24
)
# Position dans la semaine, exprimée en jours décimaux
week_position = (
    X["date"].dt.weekday
    + X["date"].dt.hour / 24
    + X["date"].dt.minute / (24 * 60)
)

X["cos_day"] = np.cos(2 * np.pi * week_position / 7)
X["sin_day"] = np.sin(2 * np.pi * week_position / 7)

plt.plot(X["cos_day"])
plt.plot(X["sin_day"])

plt.show()

plt.plot(X["cos_hour"])
plt.plot(X["sin_hour"])
plt.show()




