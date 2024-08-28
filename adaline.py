import numpy as np

class AdalineGD:
    """Adaptive Linear Neuron classifier

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
        Mean squared loss error in each epoch/iteration 
    
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
        self.b_ = np.float64(0.0)
        self.losses_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            # the identity function used as the activation function on the net_input
            output = self.activation(net_input)

            errors = (y - output) # size : (n_examples, 1)

            # update the weights and bias
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            
            # calculate loss for the iteration
            loss = (errors**2).mean() # mean squared error loss
            print(f'At iter : {i} -> loss : {loss}')

            # keep track of loss for iteration
            self.losses_.append(loss)

        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_
    
    def activation(self, X):
        """Compute Linear activation"""
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5 , 1, 0)