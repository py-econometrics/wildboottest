import numpy as np
import pandas as pd
from numba import jit
from wildboottest.weights import draw_weights
import warnings
from typing import Union, Tuple, Callable
from statsmodels.regression.linear_model import OLS

class WildDrawFunctionException(Exception):
    pass

class TestMatrixNonConformabilityException(Exception):
  pass

class WildboottestHC: 

    def __init__(self, X : Union[np.ndarray, pd.DataFrame, pd.Series], 
            Y: Union[np.ndarray, pd.DataFrame, pd.Series], 
            R : Union[np.ndarray, pd.DataFrame], 
            B: int, 
            seed:  Union[int, None] = None) -> None:
            
        """Initializes the Heteroskedastic Wild Bootstrap Class

        Args:
            X (Union[np.ndarray, pd.DataFrame, pd.Series]): Exogeneous variable array or dataframe
            Y (Union[np.ndarray, pd.DataFrame, pd.Series]): Endogenous variable array or dataframe
            R (Union[np.ndarray, pd.DataFrame]): Constraint matrix for running bootstrap
            B (int): bootstrap iterations
            seed (Union[int, None], optional): Random seed for random weight types. Defaults to None.

        Raises:
            TypeError: Raise if input arrays are lists
            TestMatrixNonConformabilityException: Raise if constraint matrix shape does not conform to X
        """    

    for i in [X, Y]:
      if isinstance(i, list):
        raise TypeError(f"{i} cannot be a list")

    if isinstance(X, (pd.DataFrame, pd.Series)):
      self.X = X.values
    else:
      self.X = X
      
    if isinstance(Y, (pd.DataFrame, pd.Series)):
      self.Y = Y.values
    else:
      self.Y = Y

    if isinstance(seed, int):
      np.random.seed(seed)

    self.N = X.shape[0]
    self.k = X.shape[1]
    self.B = B
    self.R = R
    
    if self.X.shape[1] != self.R.shape[0]:
      raise TestMatrixNonConformabilityException("The number of rows in the test matrix R, does not ")

    
def get_scores(self, bootstrap_type : str, impose_null : bool, weights_type: Union[str, Callable]): 

    self.tXXinv = np.linalg.inv(np.transpose(self.X) @ np.transpose(X))
    self.tXy = np.transpose(self.X) @ self.Y
    self.beta_hat = self.tXXing @ self.tXy 

    # solve restricted least squares problem 
    r = 0
    self.beta_r = self.beta_hat - self.tXXinv @ self.R @ (np.transpose(self.R) @ self.tXXinv @ np.transpose(self.R)) @ (np.transpose(self.R) @ self.beta - r)
    self.yhat_r = self.X @ self.beta_r
    resid_r = self.Y - self.yhat_r
    self.XXinvX = self.tXXinv @ self.X
    
    self.RXXinvX = (np.transpose(R) @ XXinv) @ np.transpose(X) 
    self.RXXinvX_squared = np.power(self.RXXinvX, 2)


    if bootstrap_type == "11":
       self.resid_multiplier = np.ones(self.N)
    else: 
        hatmat = self.X @ self.tXXinv @ np.transpose(self.X)
        diag_hatmat = np.diag(hatmat)
        if bootstrap_type == "21": 
            self.resid_multiplier = 1 / np.sqrt(1-diag_hatmat)
        elif bootstrap_type == "31":
            self.resid_multiplier = 1 / (1-diag_hatmat)
    
    self.score = X self.resid_multiplier @ resid_r
    self.XXinv_score = self.XXinv @ self.score

def get_tboot(self):

    t_boot = np.zeros(B)
    for b in range(1, B+1):

        v = draw_weights(
            t = self.weights_type, 
            full_enumeration = False, 
            N_G_bootcluster = self.N,
            boot_iter = 1
            )  
       
        # self.yhat_r + self.resid_multiplier * self.resid_r can be moved out of for loop
        boot_score = self.XXinv_score * v
        coef_boot = self.beta_hat + boot_score
        boot_var = pow(np.diag(boot_score), 2)

        t_boot[b] = (np.transpose(R) @ coef_boot - r) / np.sqrt(self.small_sample_correction * (np.transpose(R) @ sigma_boot @ R))
    
  
def get_tstat(self):

        boot_score = self.XXinv_score
        coef_boot = self.beta_hat + boot_score
        boot_var = pow(np.diag(boot_score), 2)

        self.t_stat = (np.transpose(R) @ coef_boot - r) / np.sqrt(self.small_sample_correction * (np.transpose(R) @ sigma_boot @ R))

def get_pvalue(self, pval_type = "two-tailed"):
    
    if pval_type == "two-tailed":
      self.pvalue = np.mean(np.abs(self.t_stat) < abs(self.t_boot))
    elif pval_type == "equal-tailed":
      pl = np.mean(self.t_stat < self.t_boot)
      ph = np.mean(self.t_stat > self.t_boot)
      self.pvalue = 2 * min(pl, ph)
    elif pval_type == ">":
      self.pvalue = np.mean(self.t_stat < self.t_boot)
    else: 
      self.pvalue = np.mean(self.t_stat > self.t_boot)
           