from ucimlrepo import fetch_ucirepo  
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# fetch dataset 
occupancy_detection = fetch_ucirepo(id=357) 
id_train = list(range(8143))
id_test = list(range(8144, 8144+2665))
features_names = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
X_train = occupancy_detection.data.features.iloc[id_train]
y_train = occupancy_detection.data.targets.iloc[id_train]

X_test = occupancy_detection.data.features.iloc[id_test]
y_test = occupancy_detection.data.targets.iloc[id_test]
X_train = X_train[features_names]
X_test = X_test[features_names]
# X_train["date"] = pd.to_datetime(X_train["date"], errors="coerce")
# X_test["date"] = pd.to_datetime(X_test["date"], errors="coerce")
# X_train["cos_hour"] = np.cos(
#     2 * np.pi * (X_train["date"].dt.hour + X_train["date"].dt.minute / 60) / 24
# )
# X_train["sin_hour"] = np.sin(
#     2 * np.pi * (X_train["date"].dt.hour + X_train["date"].dt.minute / 60) / 24
# )
# # Position dans la semaine, exprimée en jours décimaux
# week_position = (
#     X_train["date"].dt.weekday
#     + X_train["date"].dt.hour / 24
#     + X_train["date"].dt.minute / (24 * 60)
# )

# X_train["cos_day"] = np.cos(2 * np.pi * week_position / 7)
# X_train["sin_day"] = np.sin(2 * np.pi * week_position / 7)

# X_test["cos_hour"] = np.cos(
#     2 * np.pi * (X_test["date"].dt.hour + X_test["date"].dt.minute / 60) / 24
# )
# X_test["sin_hour"] = np.sin(
#     2 * np.pi * (X_test["date"].dt.hour + X_test["date"].dt.minute / 60) / 24
# )
# week_position_test = (
#     X_test["date"].dt.weekday
#     + X_test["date"].dt.hour / 24
#     + X_test["date"].dt.minute / (24 * 60)
# )
# X_test["cos_day"] = np.cos(2 * np.pi * week_position_test / 7)
# X_test["sin_day"] = np.sin(2 * np.pi * week_position_test / 7)

print('\n###################### SVM CLASSIFIER CODE EXECUTED #########################\n')
print(X_train.head())
# 1. Garder les features numériques
X_numeric = X_train.apply(pd.to_numeric, errors="coerce")
X_numeric_test = X_test.apply(pd.to_numeric, errors="coerce")

# 2. Gérer les valeurs manquantes
X_numeric = X_numeric.fillna(X_numeric.median())
X_numeric_test = X_numeric_test.fillna(X_numeric.median())
print(X_numeric.head())
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_numeric)
X_test_scaled = scaler.fit_transform(X_numeric_test)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)
pca = PCA(n_components=2)
X_pca_test = pca.fit_transform(X_test_scaled)
y_train = y_train.to_numpy().flatten()
y_test = y_test.to_numpy().flatten( )

df_pca = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2"],
    index=X_train.index
)
df_pca_test = pd.DataFrame(
    X_pca_test,
    columns=["PC1", "PC2"],
    index=X_test.index
)

X_pca = np.column_stack((df_pca['PC1'].to_numpy(dtype=float), df_pca['PC2'].to_numpy(dtype=float)))
X_pca_test = np.column_stack((df_pca_test['PC1'].to_numpy(dtype=float), df_pca_test['PC2'].to_numpy(dtype=float)))

print('X_pca shape : ,', X_pca.shape)
print('X_pca_test shape : ,', X_pca_test.shape)
print('y train shape :', y_train.shape)
print('y test shape : ', y_test.shape)
print('\n -  Linear SVM analysis starting ...  - \n')

Xblue = X_pca[y_train == 1.]
Xred = X_pca[y_train == 0.]

plt.figure()
plt.scatter(Xblue[:,0], Xblue[:,1], c='b',label='occupancy = 1')
plt.scatter(Xred[:,0], Xred[:,1], c='r', label ='occupancy = 0')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend()
plt.show()


mySVC = svm.SVC(kernel='linear', C=1)
mySVC.fit(X_pca,y_train)

# Compute margin and find support vectors
w = mySVC.coef_[0]
w0 = mySVC.intercept_
M = 1./np.linalg.norm(w)
print("SV per class:", mySVC.n_support_)
print("w_i:", w)
print("w_0:", w0)
print("Margin:", M)
print("w^T x0 + w_0:", np.dot(w,mySVC.support_vectors_[0,:])+w0)
print("w^T x1 + w_0:", np.dot(w,mySVC.support_vectors_[1,:])+w0)
print("w^T x2 + w_0:", np.dot(w,mySVC.support_vectors_[2,:])+w0)


y_pred = mySVC.predict(X_pca_test)


print('\n -  Linear SVM analysis finished ...  - \n')

print('\n -  Linear plot starting ...  - \n')



ratio = np.where(y_pred == y_test, np.ones(y_pred.shape),np.zeros(y_pred.shape))

print('Accuracy : ', (np.sum(ratio) / y_pred.shape) * 100, '%')

Xblue = X_pca[y_train == 1.]
Xred = X_pca[y_train == 0.]

plt.figure()
plt.scatter(Xblue[:,0], Xblue[:,1], c='b',label='occupancy = 1')
plt.scatter(Xred[:,0], Xred[:,1], c='r', label ='occupancy = 0')
XX = np.arange(-3.,7.,0.1)
YY = -(w[0]*XX+w0)/w[1]
plt.plot(XX,YY,'g')
YY = -(w[0]*XX+w0+1)/w[1]
plt.plot(XX,YY,'g--')
YY = -(w[0]*XX+w0-1)/w[1]
plt.plot(XX,YY,'g--')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend()
plt.show()

print('\n -  Linear plot finished ...  - \n')

print('\n -  RBF SVM analysis starting ...  - \n')
mySVC = svm.SVC(kernel='rbf', C=1.)
mySVC.fit(X_pca,y_train)

# Compute margin and find support vectors

print("SV per class:", mySVC.n_support_)



y_pred = mySVC.predict(X_pca_test)


print('\n -  RBF SVM analysis finished ...  - \n')

print('\n -  RBF plot starting ...  - \n')

ratio = np.where(y_pred == y_test, np.ones(y_pred.shape),np.zeros(y_pred.shape))

print('Accuracy : ', (np.sum(ratio) / y_pred.shape) * 100, '%')


plt.figure()
plt.scatter(Xblue[:,0],Xblue[:,1],c='b')
plt.scatter(Xred[:,0],Xred[:,1],c='r')
XX, YY = np.meshgrid(np.arange(np.min(X_pca[:,0]),np.max(X_pca[:,0]),2.), np.arange(np.min(X_pca[:,1]),np.max(X_pca[:,1]),2.))
ZZ = mySVC.decision_function(np.c_[XX.ravel(), YY.ravel()])
ZZ = ZZ.reshape(XX.shape)
plt.contour(XX, YY, ZZ, levels=[0],alpha=0.75)
plt.show()

print('\n -  RBF plot finished ...  - \n')


print('\n###################### SVM CLASSIFIER CODE ENDED #########################\n')