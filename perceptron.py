import numpy as np

class Perceptron:
    """Perceptron classifier

    Parameters
    ------------
    eta : float
        Learning rate between 0.0 and 1.0
    
    n_iter : int
        Number of iterations over the training dataset

    random_state : int
        Random number generator seed for random weight initialization

    Attributes:
    ------------

    w_ : 1d-array
        Weights after fitting
    
    b_ : scalar
        Bias
    
    errors : list
        Number of misclassifications in each epoch / iteration
    
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1) -> None:
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit the training data
        
        Parameters
        ------------
        X : {array-like}, shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of
            examples and n_features is the number of features.

        y : {array-like}, shape = [n_examples]
            Target values.

        Returns
        ------------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)

        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.0)
        self.errors_ = []

        for _ in range(self.n_iter):
            error = 0
            for xi, yi in zip(X, y):
                update = self.eta * (yi - self.predict(xi))                
                self.w_ += update * xi
                self.b_ = update

                if update != 0.0:
                    error += 1
            self.errors_.append(error)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0 , 1, 0)
    

    