import numpy as np
from scipy.stats import norm
from sklearn.tree import DecisionTreeRegressor

class NormalLogScore: 
    
    def __init__(self, y, init_params):
        self.y = y
        self.var = np.exp(init_params[:,1])
        
    def score(self, params):
        return - norm.logpdf(self.y, loc=params[:,0], scale=self.var**0.5).mean()

    def calc_d_score(self, params):
        D = np.zeros(params.shape)
        D[:, 0] = (params[:,0] - self.y) /self.var
        D[:, 1] = 1 - ((params[:,0] - self.y) ** 2) / self.var
        return D

    def calc_fisher(self, params):
        FI = np.zeros((len(params), 2, 2))
        FI[:, 0, 0] = 1 / self.var
        FI[:, 1, 1] = 2
        return FI

    def grad(self, params):        
        d_score, fisher = self.calc_d_score(params), self.calc_fisher(params)
        return np.linalg.solve(fisher, d_score)
    
    def update_params(self, params, learning_rate, scale, pred):        
        params = params-learning_rate*scale*pred
        self.var = np.exp(params[:,1])
        return params

class NormalNGBoost(object):
    
    def __init__(self, n_learner=100, learning_rate=1e-2):       
        self.n_learner = n_learner
        self.learning_rate = learning_rate
        self.learner_lst = []
        self.scale_lst = []
        
    def base_learner(self):
        return DecisionTreeRegressor(max_depth=3, min_samples_leaf=5)
    
    def line_search(self,dist_score, params, pred):
        init_loss = dist_score.score(params)
        scale = 1
        
        while True:
            new_params = params-scale*pred
            loss = dist_score.score(new_params)
            if(not np.isfinite(loss) or loss>init_loss or scale>256):
                break            
            scale = scale*2
            
        while True:
            new_params = params-scale*pred
            loss = dist_score.score(new_params)
            if(scale < 0.5**8):
                break
            if(np.isfinite(loss) and loss<init_loss):
                break
            scale = scale*0.5
            
        return scale                           

    def fit(self, X, y):  
        N = len(X)
        init_mean, init_std = norm.fit(y)
        init_log_var = np.log(init_std**2)
        self.init_params = np.array([np.ones((N,))*init_mean, np.ones((N,))*init_log_var]).T
        dist_score = NormalLogScore(y, self.init_params)
        params = self.init_params

        for i in range(self.n_learner):
            
            grads = dist_score.grad(params)
            mean_learner = self.base_learner()
            mean_learner.fit(X, grads[:,0])
            log_var_learner  = self.base_learner()
            log_var_learner.fit(X, grads[:,1])
            self.learner_lst.append([mean_learner, log_var_learner])
            
            pred = np.zeros((N, 2))
            pred[:,0] = mean_learner.predict(X)
            pred[:,1] = log_var_learner.predict(X)
            
            scale = self.line_search(dist_score, params, pred)
            self.scale_lst.append(scale)
            
            params = dist_score.update_params(params, self.learning_rate, scale, pred)
            #print(params[0])
            
    def predict(self, X):
        N = len(X)
        mean = np.ones((N,))*self.init_params[:,0][0]
        log_var = np.ones((N,))*self.init_params[:,1][0]
        
        for learner, scale in zip(self.learner_lst, self.scale_lst):
            mean -= self.learning_rate * scale * learner[0].predict(X)
            log_var -= self.learning_rate * scale * learner[1].predict(X)
            #print(mean[0], log_var[0])
        
        std = np.sqrt(np.exp(log_var))
        
        return mean, std