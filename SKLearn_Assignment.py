import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
%matplotlib inline

# Part 1
from sklearn.datasets import load_boston
boston_dataset = load_boston()
boston = pd.DataFrame(boston_dataset.data, columns = boston_dataset.feature_names)
boston.head()

boston['MEDV'] = boston_dataset.target
boston.isnull().sum()

X = boston.drop('MEDV', axis = 1)
Y = boston['MEDV']

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

lin = LinearRegression()
lin.fit(X_train, Y_train)

# model evaluation for training set
y_pred = lin.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_pred)))
r2 = r2_score(Y_train, y_pred)
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))

# model evaluation for testing set
y_test_pred = lin.predict(X_test)
rmse_test = (np.sqrt(mean_squared_error(Y_test, y_test_pred)))
r2_test = r2_score(Y_test, y_test_pred)
print('RMSE is {}'.format(rmse_test))
print('R2 score is {}'.format(r2_test))

from matplotlib import pyplot

importance = lin.coef_
# summarize feature importance
for i, j in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i, j))

# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


# PART 2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.cluster import KMeans
from sklearn import datasets
iris = datasets.load_iris()

# loading the iris dataset
iris_data = pd.DataFrame(iris['data'])

iris_data.head()

d = []
K = range(1,10)
for k in K:
    kmeans_mod = KMeans(n_clusters = k)
    kmeans_mod.fit(iris_data)
    d.append(kmeans_mod.inertia_)

plt.figure(figsize = (16,8))
plt.plot(K, d, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
# the plot shows that k=3 is the point where it curves


kmeans_mod = KMeans(n_clusters = 3)
kmeans_mod.fit(iris_data)

wine = datasets.load_wine()
# loading the iris dataset
wine_data = pd.DataFrame(wine['data'])
wine_data.head()

d = []
K = range(1,10)
for k in K:
    kmeans_mod = KMeans(n_clusters=k)
    kmeans_mod.fit(wine_data)
    d.append(kmeans_mod.inertia_)

plt.figure(figsize=(16,8))
plt.plot(K, d, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

kmeans_mod = KMeans(n_clusters=3)
kmeans_mod.fit(wine_data)
