import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from perceptron import Perceptron
from adaline import AdalineGD
from sklearn.datasets import load_iris

data, target = load_iris(return_X_y=True, as_frame=True)

y = target.iloc[0:100]
y = np.where(y == 0, 0, 1) # considering 2 classes
X = data.iloc[0:100, [0,2]].values # 2 features [col 0, col 2]

agd = AdalineGD(eta=0.01, n_iter=50)
agd.fit(X, y)
print(agd.w_, agd.b_)


X1_min = X[:, 0].min()
X1_max = X[:, 0].max()

x1 = np.linspace(X1_min, X1_max, 100)
x2 = (-(agd.w_[0]/agd.w_[1]) * x1 ) - (agd.b_ / agd.w_[1])

#plot the training data
plt.scatter(X[0:50, 0], X[0:50, 1] , color='red', marker='o', label='Setosa') # X[0:50, 0] 50 examples of feature 1 (col = 0) plotted with X[0:50, 1] 50 examples of feature 2 (col 1)
plt.scatter(X[50:, 0], X[50:, 1] , color='blue', marker='s', label='Versicolor')

#plot the line (which is the decision function)
plt.plot(x1,x2)
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
plt.show()