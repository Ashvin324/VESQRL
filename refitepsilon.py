import numpy as np
import scipy
import GPy

# Pass environment to Gaussian network
class RefitEpsilon:
    def __init__(self,min_epsilon,X_data,Y_data):
        self.env_min_epsilon=min_epsilon
        self.kernel=GPy.kern.RBF(input_dim=X_data.shape[1],lengthscale=10)
        self.likelihood=GPy.likelihoods.Gaussian()
        # Bound the likelihood between min epsilon and 1
        self.likelihood.constrain_bounded(self.env_min_epsilon,1)
        self.process=GPy.core.GP(X_data,Y_data,self.kernel,self.likelihood)

    def fit(self):
        self.process.optimize()
    
    # Return means as prediction
    def predict(self,X_data):
        means,vars=self.process.predict(X_data)
        return means.flatten()
    
    
    # Return epsilon from normal distribution
    def predict1(self,X_data):
        means,vars=self.process.predict(X_data)
        distributions=[scipy.stats.norm(loc=means[i],scale=vars[i]) for i in range(means.shape[0])]
        epsilons=np.array([distributions[i].rvs(size=1) for i in range(len(distributions))])
        return epsilons

    # Add samples to Gaussian process
    def add_samples(self,X_new,Y_new):
        self.process.set_XY(np.vstack((self.process.X,X_new)),np.vstack((self.process.Y,Y_new)))
    

        


