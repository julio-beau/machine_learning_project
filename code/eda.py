from ucimlrepo import fetch_ucirepo 
import pandas as pd
import matplotlib.pyplot as plt
# fetch dataset 
occupancy_detection = fetch_ucirepo(id=357) 
  
# data (as pandas dataframes) 
X = occupancy_detection.data.features 
y = occupancy_detection.data.targets 
print(occupancy_detection.variables) 
features_names = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
#Checking if the features are normalized
for feature in features_names:
    X[feature] = pd.to_numeric(X[feature], errors="coerce")
    mean = X[feature].mean()
    std = X[feature].std()
    print("Feature :", feature)
    print("Mean : ", mean)
    print("Std :", std)
    print()
#Nos features ne sont pas normalisées, il faudra le faire pour certaines analyses -> à garder en tête


fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()


#Avec ces histogrammes, on se rend compte que les features CO2 et Light sont en mode bcp de zero et longue queue
#Car elles changent bcp pour occupé/non-occupé, alors que le reste est plus indépendant de l'occupation de la pièce.
#Ca veut dire qu'elles auront plus d'importance pour une classification
for i, feature in enumerate(features_names):
    axes[i].hist(X[feature].dropna(), bins=30, edgecolor="black")
    axes[i].set_title(feature)
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel("Fréquence")

plt.tight_layout()
plt.show()


#On compare maintenant les histogrammes pièce occupée ou non.
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()
for i, feature in enumerate(features_names):
    for classe, couleur in [(0.0, "blue"), (1.0, "red")]:
        valeurs = X.loc[y["Occupancy"] == classe, feature].dropna()

        axes[i].hist(
            valeurs,
            bins=30,
            alpha=0.5,
            color=couleur,
            label=f"Occupancy = {classe}",
            density=True
        )

    axes[i].set_title(feature)
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel("Densité")
    axes[i].legend()

plt.tight_layout()
plt.show()

#On voit clairement que la température est intéressante, l'humidité l'est beaucoup moins. On pourrait imaginer travailler
#avec le log de la lumière et du CO2 pour avoir qqch de gaussien peut-être

#La question de savoir l'utilité de l'humidité est réelle, on doit se poser la question de si on la vire ou pas.
#Aussi, je ne l'ai pas considéré jusqu'ici, mais le fait que les variables aient une date peut avoir son importance, dans le sens ou les config où la pièce
#est occupée se suivent dans le temps, c'est pas random.

#Petite matrice de corrélation histoire de.


import seaborn as sns
X_numeric = X.select_dtypes(include="number")


corr = X_numeric.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1,
    vmax=1
)

plt.title("Matrice de corrélation")
plt.tight_layout()
plt.show()
#On voit notamment qu'il n'y a pas de corrélation entre humidité et lumière, alors que la lumière
#est le meilleur proxy en ordre 1 pour l'occupation, raison de plus pour potentiellement benner cette variables

#Je propose de faire une pca, histoire de voir, meme si je pense que globalement on va avoir 3 composantes principales(lumière, CO2, Temp), une moyenne (humidity ratio), et du caca (humidité)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Garder les features numériques
X_numeric = X.select_dtypes(include="number").copy()

# 2. Gérer les valeurs manquantes
X_numeric = X_numeric.fillna(X_numeric.median())

# 3. Standardiser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

# 4. PCA en 5 dimensions car on a 5 variables
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)

# 5. Créer un DataFrame pour les composantes principales
df_pca = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2","PC3","PC4","PC5"],
    index=X.index
)

print(df_pca.head())

# Variance expliquée
print("Variance expliquée :", pca.explained_variance_ratio_)
print("Variance totale expliquée :",
      pca.explained_variance_ratio_.sum())
#Update : seulement deux composantes et pas 3, on a 49%, 34%, 10%, 6.3%, 2.56%

#Visualisation de la PCA
target = y["Occupancy"].to_numpy()

plt.figure(figsize=(10, 7))

for classe, couleur in [(0, "blue"), (1, "red")]:
    masque = target == classe

    plt.scatter(
        X_pca[masque, 0],
        X_pca[masque, 1],
        c=couleur,
        alpha=0.5,
        label=f"Target = {classe}",
        s=25
    )

plt.xlabel(
    f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f} %)"
)
plt.ylabel(
    f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f} %)"
)
plt.title("PCA — Projection des données en 2D")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

#Avec la PCA en 2D, on voit une séparation des données possible mais une SVM donnerait à vue de nez du 90%, je pense que ce sera pas suffisant
#Meme si intéressant de commencer par ça je pense.


#Juste pour voir, matrice de corrélation en incluant la target, histoire de se conforter dans ce qu'on a vu jusqu'ici.
data = X.select_dtypes(include="number").copy()
data["target"] = y["Occupancy"].to_numpy()

# Matrice de corrélation
corr = data.corr()

# Heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1,
    vmax=1,
    center=0
)

plt.title("Matrice de corrélation avec la target")
plt.tight_layout()
plt.show()

#Le résultat est clair : c'est bien la lumière en premier, puis le co2 et température qui sont corrélés à l'occupation
#L'humidity ratio est légèrement corrélé car la temp entre dans le calcul et l'humidité ne semble vraiment servir à rien
#et ajoute du bruit je pense.