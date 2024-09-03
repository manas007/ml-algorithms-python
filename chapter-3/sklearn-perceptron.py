from sklearn import datasets
import numpy as np

iris = datasets.load_iris()

X = iris.data[ : , [2,3]]
y = iris.target

print('Class labels :', np.unique(y))

# train_test_split function to split dataset
from sklearn.model_selection import train_test_split

# dataset is shuffled internally by train_test_split prior to splitting
# stratification means that the train_test_split method returns training and test subsets that have the
# same proportions of class labels as the input dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)

# standardize the data a.k.a feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# fit estimates the sample mean and standard deviation for each feature dimension from THE TRAINING DATA
sc.fit(X_train)

# we use the same scaling parameteres from THE TRAINING DATA, to standardize the TEST DATA
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# train perceptron model on the standardized dat
from sklearn.linear_model import Perceptron
ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)

# predict 
y_pred = ppn.predict(X_test_std)
classification_errors = (y_test != y_pred).sum()
print('Number of examples that were not correctly classified a.k.a classification error', classification_errors)

classification_error_pct = (classification_errors / y_test.size) * 100
print('Classification error percentage', classification_error_pct)

# accuracy is nothing but the 1 - classification error
classification_accuracy_pct = 100 - classification_error_pct
print('Classification accuracy', classification_accuracy_pct)



