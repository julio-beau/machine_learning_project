from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn import svm 
import numpy as np
import matplotlib.pyplot as plt

# fetch dataset 
occupancy_detection = fetch_ucirepo(id=357) 
  
# data (as pandas dataframes) 
X = occupancy_detection.data.features 
y = occupancy_detection.data.targets 
# print(occupancy_detection.variables) 
# features_names = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
X_test = X.iloc[8144:8144+2665]
y_test = y.iloc[8144:8144+2665]
X = X.iloc[:8143]
y = y.iloc[:8143]
X_svm = np.column_stack((X['Light'].to_numpy(dtype=float), X['CO2'].to_numpy(dtype=float)))
y_svm = y['Occupancy'].to_numpy(dtype=float)

print('\n###################### SVM CLASSIFIER CODE EXECUTED #########################\n')

print('X_svm shape : ', X_svm.shape, ', y_svm shape : ', y_svm.shape)

print('X_svm : ', X_svm)
print('y_svm : ', y_svm)

print('\n -  Linear SVM analysis starting ...  - \n')
mySVC = svm.SVC(kernel='linear', C=1)
mySVC.fit(X_svm,y_svm)

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

Xtest = np.column_stack((X_test['Light'].to_numpy(dtype=float), X_test['CO2'].to_numpy(dtype=float)))
ytest = y_test['Occupancy'].to_numpy(dtype=float)
y_pred = mySVC.predict(Xtest)


print('\n -  Linear SVM analysis finished ...  - \n')

print('\n -  Linear plot starting ...  - \n')

ratio = np.where(y_pred == ytest, np.ones(y_pred.shape),np.zeros(y_pred.shape))

print('Accuracy : ', (np.sum(ratio) / y_pred.shape) * 100, '%')

Xblue = X_svm[y_svm == 1.]
Xred = X_svm[y_svm == 0.]

plt.figure()
plt.scatter(Xblue[:,0], Xblue[:,1], c='b',label='occupancy = 1')
plt.scatter(Xred[:,0], Xred[:,1], c='r', label ='occupancy = 0')
XX = np.arange(-1.,500.,1.)
YY = -(w[0]*XX+w0)/w[1]
plt.plot(XX,YY,'g')
YY = -(w[0]*XX+w0+1)/w[1]
plt.plot(XX,YY,'g--')
YY = -(w[0]*XX+w0-1)/w[1]
plt.plot(XX,YY,'g--')
plt.xlabel('Light')
plt.ylabel('CO2')
plt.legend()
plt.show()

print('\n -  Linear plot finished ...  - \n')

print('\n -  RBF SVM analysis starting ...  - \n')
mySVC = svm.SVC(kernel='rbf', C=1.)
mySVC.fit(X_svm,y_svm)

# Compute margin and find support vectors

print("SV per class:", mySVC.n_support_)


Xtest = np.column_stack((X_test['Light'].to_numpy(dtype=float), X_test['CO2'].to_numpy(dtype=float)))
ytest = y_test['Occupancy'].to_numpy(dtype=float)
y_pred = mySVC.predict(Xtest)


print('\n -  RBF SVM analysis finished ...  - \n')

print('\n -  RBF plot starting ...  - \n')

ratio = np.where(y_pred == ytest, np.ones(y_pred.shape),np.zeros(y_pred.shape))

print('Accuracy : ', (np.sum(ratio) / y_pred.shape) * 100, '%')


plt.figure()
plt.scatter(Xblue[:,0],Xblue[:,1],c='b')
plt.scatter(Xred[:,0],Xred[:,1],c='r')
XX, YY = np.meshgrid(np.arange(np.min(X_svm[:,0]),np.max(X_svm[:,0]),2.), np.arange(np.min(X_svm[:,1]),np.max(X_svm[:,1]),2.))
ZZ = mySVC.decision_function(np.c_[XX.ravel(), YY.ravel()])
ZZ = ZZ.reshape(XX.shape)
plt.contour(XX, YY, ZZ, levels=[0],alpha=0.75)
plt.show()

print('\n -  RBF plot finished ...  - \n')

print('\n###################### SVM CLASSIFIER CODE ENDED #########################\n')
