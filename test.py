import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron
from sklearn.datasets import load_iris

data, target = load_iris(return_X_y=True, as_frame=True)

y = target.iloc[0:100]
y = np.where(y == 0, 0, 1) # considering 2 classes
X = data.iloc[0:100, [0,2]].values # 2 features [col 0, col 2]

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
print(ppn.w_, ppn.b_)

# as we have two features x1 and x2, the equation is w1x1 + w2x2 + b = 0
# therefore, w2x2 = -w1x1 - b
# x2 = (-w1x1 - b) / w2
# x2 = (-w1/w2) x1 - (b/w2)
# now this in in the form of the standard line equation y = ax + c where a = -w1/w2 and c = -b/w2

# let's plot it. First let us get the min and max value in feature 1 so that we can create some random values between in this range
X1_min = X[:, 0].min()
X1_max = X[:, 0].max()

x1 = np.linspace(X1_min, X1_max, 100)
x2 = (-(ppn.w_[0]/ppn.w_[1]) * x1 ) - (ppn.b_ / ppn.w_[1])

#plot the training data
plt.scatter(X[0:50, 0], X[0:50, 1] , color='red', marker='o', label='Setosa') # X[0:50, 0] 50 examples of feature 1 (col = 0) plotted with X[0:50, 1] 50 examples of feature 2 (col 1)
plt.scatter(X[50:, 0], X[50:, 1] , color='blue', marker='s', label='Versicolor')

#plot the line (which is the decision function)
plt.plot(x1,x2)
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
plt.show()

