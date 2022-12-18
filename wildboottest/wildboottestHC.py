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

  def get_adjustments(self, bootstrap_type):

    # allow for arbitrary different adjustments for bootstrap and standard t-stat
    self.tXXinv = np.linalg.inv(np.transpose(self.X) @ self.X)
    self.resid_multiplier_boot = _adjust_scores(self.X, self.tXXinv, bootstrap_type[0])
    if bootstrap_type[0] == bootstrap_type[1]:
      self.resid_multiplier_stat = self.resid_multiplier_boot
    else: 
      self.resid_multiplier_stat = _adjust_scores(self.X, self.tXXinv, bootstrap_type[1])

  def get_uhat(self, impose_null : bool): 
      
    self.tXy = np.transpose(self.X) @ self.Y
    self.beta_hat = self.tXXinv @ self.tXy 
    self.uhat = self.Y - self.X @ self.beta_hat
    
    
    self.uhat_stat = self.uhat * self.resid_multiplier_stat

    if impose_null: 
      self.impose_null = True
      r = 0
      self.beta_r = self.beta_hat - self.tXXinv @ self.R * ( 1 / (np.transpose(self.R) @ self.tXXinv @ self.R)) * (np.transpose(self.R) @ self.beta_hat - r)#self.uhat_r = self.Y - self.beta_r 
      self.uhat_r = self.Y - self.X @ self.beta_r 
      self.uhat_boot = self.uhat_r * self.resid_multiplier_boot
    else: 
      self.impose_null = False    
      self.uhat_boot = self.uhat * self.resid_multiplier_boot

  def get_tboot(self, weights_type: Union[str, Callable]):

    self.weights_type = weights_type
      
    k = np.where(self.R) == 1
    self.tXXinvX = self.tXXinv @ np.transpose(self.X)  
  
    if self.impose_null == True: 
      beta_center = self.beta_hat
    else: 
      beta_center = np.zeros(self.k)

    self.t_boot = np.zeros(self.B)
    for b in range(0, self.B):
      # draw N x 1 weights vector for each iteration - currently v always attaches column of ones
      # this column is not used anyways, so drop it
      v = draw_weights(
          t = self.weights_type, 
          full_enumeration = False, 
          N_G_bootcluster = self.N,
          boot_iter = 1
      )
    # get score boot
      beta_b = self.tXXinvX @ (self.uhat_boot * v)
      beta_bk = beta_b[k] - beta_center
      cov_v = np.pow(beta_bk, 2)
      self.t_boot[b] = beta_bk / np.sqrt(cov_v)
 
 
  def get_tstat(self):
    
    k = np.where(self.R == 1)
    cov = np.sqrt(self.tXXinvX[k,k], 2) @ self.uhat_stat
    self.t_stat = self.beta / np.sqrt(cov)
        
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
             


def _adjust_scores(X, tXXinv, variant):
    
  N = X.shape[0]
  if variant == "1":
    # HC1
    resid_multiplier = np.ones(N)
  else: 
    hatmat = X @ tXXinv @ np.transpose(X)
    diag_hatmat = np.diag(hatmat)
    if variant == "2": 
      # HC2
      resid_multiplier = 1 / np.sqrt(1-diag_hatmat)
    elif variant == "3":
      # HC2
      resid_multiplier = 1 / (1-diag_hatmat)

  return resid_multiplier