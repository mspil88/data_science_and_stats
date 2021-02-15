
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import itertools
import numpy as np
from collections import defaultdict


# TO DO: 
# add different scaler types
# Errors potentially looking on the low side - need to ensure this is working correctly
# move kfolds into the init

class ModelOptimiser:
    def __init__(self, model_grid = None):
        self.model_grid = model_grid
        self.optimal_params = None
        self.optimal_error = None
        self.one_sd_params = None
        self.one_sd_idx = None
        self.one_sd_error = None
        self.folds = None
        self.num_folds = None
        
    
    def kfolds(self, n_folds, folds_kwargs={}):
        if isinstance(n_folds, KFold):
            self.folds = n_folds
            self.num_folds = n_folds.n_splits
        else:
            self.folds = KFold(n_splits = n_folds, shuffle=True, **folds_kwargs)
            self.num_folds = self.folds.n_splits
            
    def grid_search(self, model, X_train, Y_train, X_test, Y_test, param_grid, score):
        params_list = []
        mean_sq_error = []
        for i in itertools.product(*self.model_grid.values()):
            params = dict(zip(self.model_grid.keys(), i))
            params_list.append(params)
            en = model(**params).fit(X_train, Y_train)
            pred = en.predict(X_test)
            mse = score(Y_test, pred)
            mean_sq_error.append(mse)
        return list(zip(params_list, mean_sq_error)), mean_sq_error

    @staticmethod
    def unpack_params(param_args, model_grid, optimal_index):
        for i, j in enumerate(itertools.product(*param_args)):
            if i == optimal_index:
                return dict(zip(model_grid.keys(), j))
    
    def cross_validation_errors(self, gs):
        output = []
        for i in range(self.num_folds):
            output.append(gs.cv_results_[f'split{i}_test_score'])
        return np.vstack(output).T
    
    def optimal_errors(self, cv_errors):
        mean_errors = np.mean(cv_errors, axis = 1)
        std_errors = np.std(cv_errors, axis = 1)/np.sqrt(self.num_folds)
        minimum_index = np.argmin(mean_errors)
        min_error = mean_errors[minimum_index]
        min_sd = std_errors[minimum_index]
        get_one_sd = min_error + min_sd
        one_sd_index = np.argmax(mean_errors < get_one_sd)
        print(min_error)
        print(min_sd)
        print(mean_errors)
        
        return mean_errors[minimum_index], minimum_index, mean_errors[one_sd_index], one_sd_index 
    
    def scaler_util(self, X_train, X_val, _type = 'normalise'):
        if _type == 'normalise':
            ss = StandardScaler()
            X_train = ss.fit_transform(X_train)
            X_val = ss.transform(X_val)
            return X_train, X_val
                
    def optimiser(self, model, X, y, score):
        
        self.errors = defaultdict(list)
        i = -1
        
        for train, val in self.folds.split(X,y):
            i +=1
            X_train, X_val = X.iloc[train], X.iloc[val]
            y_train, y_val = y.iloc[train], y.iloc[val]
            
            X_train, X_val = self.scaler_util(X_train,X_val)
            
            _, mean_sq_error = self.grid_search(model=model, X_train = X_train, Y_train = y_train, X_test = X_val, Y_test = y_val, param_grid = self.model_grid, score=score)
            self.errors[f'split_{i}'].append(np.array(mean_sq_error))
        
        cv_errors = np.vstack([self.errors[f'split_{i}'] for i in range(10)]).T 
        self.min_error, self.min_error_idx, self.one_sd_error, self.one_sd_idx =  self.optimal_errors(cv_errors)
        self.one_sd_params = self.unpack_params(list(self.model_grid.values()), self.model_grid, self.one_sd_idx)
        self.min_params = self.unpack_params(list(self.model_grid.values()), self.model_grid, self.min_error_idx)
