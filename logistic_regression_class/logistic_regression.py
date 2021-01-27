
import numpy as np
import pandas as pd
from scipy.stats.distributions import chi2
from scipy.stats import t, zscore
import scipy.stats as st
from matplotlib import pyplot as plt


#TO DO:
# 1 - Convergence is pretty slow, need to make this faster and potentially replace the matrix inversion used for the Hessian with np.solve
# 2 - Find a way to efficiently estimate the null model and recover the null loglikelihood, output the result of an LR test and output McFadden's R-squared


class LogisticRegression:
    def __init__(self):
        pass
    
    @staticmethod
    def __sigmoid(z):
        return 1/(1+np.exp(-z))
    
    @staticmethod
    def __prob(z):
        return np.exp(z)/(1+np.exp(z))
    
    @staticmethod
    def __include_intercept(X):
        return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    
    @property
    def n_obs(self):
        return self.__obs
    @property
    def coefs(self):
        return self.__coefs
    @property
    def cost_history(self):
        return self.__cost_history
    @property
    def std_errors(self):
        return self.__std_errors
    @property
    def log_likelihood(self):
        return self.__log_likelihood
    @property
    def resids(self):
        return self.__model_resids
    
    @property
    def aic(self):
        return self.__aic

    def __cross_entropy_cost(self, X, y, theta):
        z = np.dot(X, theta)
        return np.mean(np.sum(y*np.log(LogisticRegression.__sigmoid(z)) + 
                                       (1-y)*np.log(1-LogisticRegression.__sigmoid(z))))
    
    def __gradient(self, X, y, theta):
        
        z = np.dot(X, theta)
        return (np.dot((LogisticRegression.__sigmoid(z)- y),X))

    def __hessian(self, X, y, theta):
        
        z = np.dot(X, theta)
        h = LogisticRegression.__sigmoid(z)     
        diag = np.dot(h,1-h)
        
        hess = np.dot(np.dot(X.T, diag), X)
        
                
        return hess
   
    def fit(self, X, y, max_iterations=10000, threshold = 0.0000001, include_intercept=False):
        
        if isinstance(X, pd.DataFrame):
            self.variable_names = list(X.columns)
            X = np.array(X)
        elif isinstance(X, np.ndarray):
            self.variable_names = ['x_'+str(i) for i in range(1, X.shape[1]+1)]
        
        
        if include_intercept:
            X = LogisticRegression.__include_intercept(X)
            self.theta = np.zeros(X.shape[1]) 
            #print(self.theta)
        else:
            self.theta = np.zeros(X.shape[1]) 


        cost_history = []
        cost_history.append(self.__cross_entropy_cost(X, y, self.theta))
        
        threshold = threshold
        
        i = 0
        for i in range(max_iterations):
            i +=1
            grad = self.__gradient(X, y, self.theta)
            hess = self.__hessian(X,y, self.theta)            
            self.theta = self.theta -  np.dot(np.linalg.inv(hess), grad)
            cost = self.__cross_entropy_cost(X, y, self.theta)
            
            if i > 1:
                converged = self.__convergence(cost_history[-1], cost, threshold)
                if converged:
                    break
                
            cost_history.append(cost)                
                
        #print(f'Converged on the {i}th iteration')
        self.__obs = X.shape[0]
        self.__params = X.shape[1]
        self.__coefs = self.theta 
        self.__cost_history = cost_history
        self.__std_errors = self.__compute_standard_errors(X)
        self.__log_likelihood = self.__cross_entropy_cost(X, y, self.theta)
        self.__model_resids = y - self.predict_probs(X)
        self.__aic = -2*(self.__log_likelihood) + 2*self.__params
        
    def __convergence(self, previous_cost, current_cost, threshold):
        
        difference = np.abs(previous_cost - current_cost)

        converged = False
        if np.any(difference < threshold):
            converged = True
        
        #if np.isnan(current_cost):
        #    self.converged = True
            
            
        #elif current_cost > previous_cost:
        #    self.converged = True
            
        return bool(converged)
            
    def predict_probs(self, X, include_intercept=False):
        if include_intercept:
            return LogisticRegression.__prob(np.dot(LogisticRegression.__include_intercept(X), self.theta))
        else:
            return LogisticRegression.__prob(np.dot(X, self.theta))
    
    def __compute_standard_errors(self, X):
        probs = self.predict_probs(X)
        #X_design = LogisticRegression._include_intercept(X)
        V = (probs*(1-probs))*np.eye(X.shape[0])
        return np.sqrt(np.diag(np.linalg.inv(np.dot(np.dot(X.T, V), X))))
    
    def __compute_confidence_intervals_or(self, confidence="95%", coef_type = 'log_odds'):
        
        #df = self.obs - self.params
        
        if confidence == "95%" or confidence == "90%" or confidence == "99%":
            pass
        else:
            raise ValueError("Must specify 90%, 95% or 99% confidence interval")
        
        self.confidence = confidence
        
        z_scores = {'z_95%': st.norm.ppf(.975),
                    'z_90%': st.norm.ppf(.95),
                    'z_99%': st.norm.ppf(.99)}
        
        if coef_type == 'log_odds':
            z_upper = self.__coefs + z_scores[f'z_{str(confidence)}']*self.__std_errors
            z_lower = self.__coefs - z_scores[f'z_{str(confidence)}']*self.__std_errors
            return z_lower, z_upper
        elif coef_type == 'odds_ratio':
            z_upper = np.exp(self.__coefs + z_scores[f'z_{str(confidence)}']*self.__std_errors)
            z_lower = np.exp(self.__coefs - z_scores[f'z_{str(confidence)}']*self.__std_errors)
            
            return z_lower, z_upper
       
        
        
    def __z_scores(self):
        return self.__coefs/self.__std_errors
    
    def __p_values(self):
        return [st.norm.sf(abs(i))*2 for i in self.__z_scores()]

    def __odds_ratios(self):
        ors = np.exp(self.__coefs)
        or_std_errors = np.exp(self.__std_errors)
        return ors, or_std_errors
        
    def __get_coefs(self, coef_type='log_odds'):
        if coef_type == 'log_odds':
            return self.__coefs, self.__std_errors
        elif coef_type == 'odds_ratio':
            return self.__odds_ratios()[0], self.__odds_ratios()[1]  
    
    def plot_coefs(self, intercept = False, coef_type='log_odds'):        
            
        l_90, u_90 = self.__compute_confidence_intervals_or('90%', coef_type=coef_type)
        l_95, u_95 = self.__compute_confidence_intervals_or('95%', coef_type=coef_type)
        l_99, u_99 = self.__compute_confidence_intervals_or('99%', coef_type=coef_type)
        
        title = coef_type
        
        if coef_type == 'log_odds':
            df = pd.DataFrame([['intercept']+self.variable_names, self.__coefs, l_90, u_90, l_95, u_95, l_99, u_99]).T
            x_line = 0
        elif coef_type == 'odds_ratio':
            df = pd.DataFrame([['intercept']+self.variable_names, self.__odds_ratios()[0], l_90, u_90, l_95, u_95, l_99, u_99]).T
            x_line = 1
        
        df.columns = ['variable', 'coef', 'l_90', 'u_90', 'l_95', 'u_95', 'l_99', 'u_99']
        plt.errorbar(df['coef'],df['variable'],xerr=(df['coef']-df['l_90'],df['u_90']-df['coef']), fmt='o', c = 'black')
        plt.errorbar(df['coef'],df['variable'],xerr=(df['coef']-df['l_95'],df['u_95']-df['coef']), fmt='o', c = 'black', alpha=0.5)
        plt.errorbar(df['coef'],df['variable'],xerr=(df['coef']-df['l_99'],df['u_99']-df['coef']), fmt='o', c = 'black', alpha=0.2)
        plt.axvline(x=x_line, color='black', linestyle='--', linewidth=0.5)
        plt.title(title)
        plt.show()           
    
    def plot_binned_residuals(self, x, x_lab, n_class=None):
        
        residual_plotter.plot_binned_residuals(x, self.__model_resids, x_lab, n_class)
        
        
        
    def summary(self, coef_type = 'log_odds'):
        output = zip(['intercept']+list(self.variable_names), list(self.__get_coefs(coef_type)[0]), 
                     list(self.__get_coefs(coef_type)[1]), list(self.__z_scores()), 
                     list(self.__p_values()),
                     list(self.__compute_confidence_intervals_or(coef_type=coef_type)[0]),
                     list(self.__compute_confidence_intervals_or(coef_type=coef_type)[1]))

        print(f'variable            coef          se      zstat    pvalues   {self.confidence}_l    {self.confidence}_h')
        print('----------------------------------------------------------------------------')
        for variable, coef, se, zstat, pvalues, ci_l, ci_u in output:
            print(f'{variable:>10}{coef:>14.2f}{se:>13.2f}{zstat:>10.2f}{pvalues:>10.2f}{ci_l:>9.2f}{ci_u:>9.2f}')
        print('----------------------------------------------------------------------------')
        print(f"No of observations: {self.__obs}")
        print(f"Model degrees of freedom: {self.__obs - self.__params}")
        print(f"Log-Likelihood: {np.round(self.__log_likelihood,2)}")
        print(f"AIC {np.round(self.__aic,2)}")




class residual_plotter:
    def __init__(self):
        pass

    @staticmethod
    def __binned_residuals(x, y, n_class=None):
        
        if n_class is None:
            n_class = np.sqrt(len(x))
            
        if isinstance(x, np.ndarray):
            pass
        else:
            x = np.array(x)
                
        output = []
        breaks_idx = np.int_(np.floor(len(x)*(np.arange(1, n_class,1)/n_class)))
        breaks = np.insert(np.append(np.sort(x)[breaks_idx], float('inf')),0,float('-inf'))
        x_binned = pd.cut(x, bins=breaks, labels=False)
            
        for i in range(0,int(n_class)):
            items = np.arange(0, len(x),1)[x_binned==i]
            x_lo = np.min(x[items])
            x_hi = np.max(x[items])
            xbar = np.mean(x[items])
            ybar = np.mean(y[items])
            n = len(items)
            sdev = np.std(y[items])
            output.append((xbar, ybar, n, x_lo, x_hi, 2*sdev/np.sqrt(n)))
        
        return pd.DataFrame(output, columns=('xbar','ybar', 'n', 'xlo', 'xhi', '2se'))
    
    @staticmethod
    def plot_binned_residuals(x, y, x_lab, n_class=None):
        df = residual_plotter.__binned_residuals(x, y, n_class=n_class)
        plt.scatter(df['xbar'], df['ybar'], color='black', s=2)
        plt.plot(df['xbar'], df['2se'], color='grey', linewidth=0.5)
        plt.plot(df['xbar'], -1*df['2se'], color='grey',linewidth=0.5)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        plt.xlabel(x_lab)
        plt.ylabel("Average Resid")
        plt.title("Binned Residual Plot")
