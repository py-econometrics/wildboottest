import numpy as np
import pandas as pd
from numba import jit
from itertools import product

class Wildboottest: 
  
  '''
  Create an object of Wildboottest and get p-value by successively applying
  methods in the following way: 
    
  wb = Wildboottest(X = X, Y = y, cluster = cluster, R = R, B = B)
  wb.get_scores(bootstrap_type = "11", impose_null = True)
  wb.get_numer()
  wb.get_denom()
  wb.get_tboot()
  wb.get_vcov()
  wb.get_tstat()
  wb.get_pvalue()  
  
  Later we can replace X, Y, cluster, R with an object estimated via 
  statsmodels or linearmodels and a "param" values (as in fwildclusterboot::boottest())
  
  '''
  
  def __init__(self, X, Y, cluster, bootcluster, R, B, seed = None):
      
      "Initialize the wildboottest class"
      #assert bootstrap_type in ['11', '13', '31', '33']
      #assert impose_null in [True, False]

      if isinstance(X, pd.DataFrame):
        self.X = X.values
      if isinstance(Y, pd.DataFrame):
        self.Y = Y.values
      if isinstance(cluster, pd.DataFrame):
        clustid = cluster.unique()
        self.cluster = cluster.values
      if isinstance(bootcluster, pd.DataFrame):
        bootclustid = bootcluster.unique()
        self.bootcluster = bootcluster.values
      else:
        clustid = np.unique(cluster)
        bootclustid = np.unique(bootcluster)
        
      if isinstance(seed, int):
        np.random.seed(seed)

      self.N_G_bootcluster = len(bootclustid)
      self.G  = len(clustid)

      k = R.shape[0]
      self.k = k 
      self.B = B
      self.X = X
      self.R = R
      
      self.ssc = 1
      
      X_list = []
      y_list = []
      tXgXg_list = []
      tXgyg_list = []
      tXX = np.zeros((k, k))
      tXy = np.zeros(k)
      
      #all_cluster = np.unique(bootcluster)
      
      for g in bootclustid:
        
        # split X and Y by (boot)cluster
        X_g = X[np.where(bootcluster == g)]
        Y_g = Y[np.where(bootcluster == g)]
        tXgXg = np.transpose(X_g) @ X_g
        tXgyg = np.transpose(X_g) @ Y_g
        X_list.append(X_g)
        y_list.append(Y_g)
        tXgXg_list.append(tXgXg)
        tXgyg_list.append(tXgyg)
        tXX = tXX + tXgXg
        tXy = tXy + tXgyg
      
      self.clustid = clustid
      self.bootclustid = bootclustid
      self.X_list = X_list
      self.Y_list = y_list
      self.tXgXg_list = tXgXg_list
      self.tXgyg_list = tXgyg_list
      self.tXX = tXX
      self.tXy = tXy
        
      tXXinv = np.linalg.inv(tXX)
      self.RtXXinv = np.matmul(R, tXXinv)
      self.tXXinv = tXXinv 
      
  def get_weights(self, weights_type):
    
    self.weights_type = weights_type 
    
    if 2**self.N_G_bootcluster < self.B:
      self.full_enumeration = True
    else: 
      self.full_enumeration = False
      
    self.v, self.B = draw_weights(
      t = self.weights_type, 
      full_enumeration = self.full_enumeration, 
      N_G_bootcluster = self.N_G_bootcluster,
      boot_iter = self.B
    )  
    
  def get_scores(self, bootstrap_type, impose_null):
    
      if bootstrap_type[1:2] == '1':
        self.crv_type = "crv1"
      elif bootstrap_type[1:2] == '3':
        self.crv_type = "crv3"

      bootstrap_type_x = bootstrap_type[0:1] + 'x'

      if impose_null == True:
        self.bootstrap_type = "WCR" + bootstrap_type_x
      else:
        self.bootstrap_type = "WCU" + bootstrap_type_x
    
      # precompute required objects for computing scores & vcov's
      if self.bootstrap_type in ["WCR3x", "WCU3x"]: 
          
        X = self.X
        X1 = X[:,R != 1]
        X1_list = []
        tX1gX1g = []
        tX1gyg = []
        tXgX1g = []
        tX1X1 = np.zeros((self.k-1, self.k-1))
        tX1y = np.array(self.k-1)
          
        for ix, g in enumerate(self.bootclustid):
          #ix = g = 1
          X1_list.append(X1[np.where(bootcluster == g)])
          tX1gX1g.append(np.transpose(X1_list[ix]) @ X1_list[ix])
          tX1gyg.append(np.transpose(X1_list[ix]) @ self.Y_list[ix])
          tXgX1g.append(np.transpose(self.X_list[ix]) @  X1_list[ix])
          tX1X1 = tX1X1 + tX1gX1g[ix]
          tX1y = tX1y + tX1gyg[ix]
            
        self.tX1X1inv = np.invert(tX1X1)
        
        if self.bootstrap_type in ["WCR3x"]:
          
          inv_tXX_tXgXg = []
          beta_1g_tilde = []
          
          for ix, g in enumerate(self.bootclustid):
            # use generalized inverse 
            inv_tXX_tXgXg.append(np.linalg.pinv(self.tXX - self.tXgXg_list[ix]))
            beta_1_tilde = np.linalg.pinv(self.tX1X1 - self.tX1gX1g_list[ix]) @ (self.tX1y - self.tX1gyg_list[ix])
            beta_1g_tilde.append()
          
          beta = beta_1g_tilde
          M = self.tXgX1g_list
            
        elif self.bootstrap_type in ["WCU3x"]: 
            
          beta_g_hat = []
          for ix, g in enumerate(self.bootclustid):
            beta_g_hat.append(np.linalg.pinv(self.tXX - self.tXgXg_list[ix]) @ (self.tXy - self.tXgyg_list[ix]))
  
          beta = beta_g_hat
          M = tXgXg_list
          
      elif self.bootstrap_type in ["WCR1x"]: 
            
        self.beta_hat = self.tXXinv @ self.tXy
        A = 1 / (np.transpose(self.R) @ self.tXXinv @ self.R)
        beta_tilde = self.beta_hat - self.tXXinv @ self.R / A * (self.R @ self.beta_hat - 0)
        beta = beta_tilde
        M = self.tXgXg_list
          
      elif self.bootstrap_type in ["WCU1x"]: 
            
        beta_hat = np.matmul(self.tXXinv, self.tXy)
        self.beta_hat = beta_hat
        beta = beta_hat 
        M = self.tXgXg_list

      # compute the list of scores
      scores_list = []
      for ix, g in enumerate(self.bootclustid):
          # A - B x c
        if isinstance(M, list):
          scores_list.append(self.tXgyg_list[ix] - M[ix] @ beta)
        else: 
          scores_list.append(self.tXgyg_list[ix] - M @ beta)

      self.scores_mat = np.transpose(np.array(scores_list)) # k x G 
      
  
  def get_numer(self):
    
      # Calculate the bootstrap numerator
      self.Cg = self.R @ self.tXXinv @ self.scores_mat 
      self.numer = self.Cg @ self.v
    
  def get_denom(self):
    
      if self.crv_type == "crv1":
    
        H = np.zeros((self.G, self.G))
    
        # numba optimization possible? 
        for ixg, g in enumerate(self.bootclustid):
          for ixh, h in enumerate(self.bootclustid):
            # can be improved by replacing list calls with matrices; 
            H[ixg,ixh] = self.R @ self.tXXinv @ self.tXgXg_list[ixg] @ self.tXXinv @ self.scores_mat[:,ixh]
  
        # now compute denominator
        # numba / cython / c++ optimization possible? Porting this part from 
        # R to c++ gives good speed improvements
        @jit
        def compute_denom(Cg, H, bootclustid, B, G, v, ssc):
          
          denom = np.zeros(B + 1)
      
          for b in range(0, B+1):
            Zg = np.zeros(G)
            for ixg, g in enumerate(bootclustid):
              vH = 0
              for ixh, h in enumerate(bootclustid):
                vH = vH + v[ixh,b] * H[ixg,ixh]
              Zg[ixg] = Cg[ixg] * v[ixg,b] - vH
            
            # todo: ssc
            denom[b] = ssc * np.sum(np.power(Zg,2))
            
          return denom
          
        self.denom = compute_denom(self.Cg, H, self.bootclustid, self.B, self.G, self.v, self.ssc)
      
  def get_tboot(self):
    
      t_boot = self.numer / np.sqrt(self.denom)
      self.t_boot = t_boot[1:(self.B+1)] # drop first element - might be useful for comp. of

  def get_vcov(self):
    
    if self.crv_type == "crv1":
          
      meat = np.zeros((self.k,self.k))
      for ixg, g in enumerate(self.bootclustid):
        score = np.transpose(self.X_list[ixg]) @ (self.Y_list[ixg] - self.X_list[ixg] @ self.beta_hat)
        meat = meat + np.outer(score, score)
      
      self.vcov = self.tXXinv @ meat @ self.tXXinv
    
        
  def get_tstat(self):
        
    se = np.sqrt(self.ssc * self.R @ self.vcov @ np.transpose(self.R))
    t_stats = self.beta_hat / se
    self.t_stat = t_stats[np.where(self.R == 1)]

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
      
      
      
      
class WildDrawFunctionException(Exception):
    pass

def rademacher(n: int) -> np.ndarray:
    rng = np.random.default_rng()
    return rng.choice([-1,1],size=n, replace=True)

def mammen(n: int) -> np.ndarray:
    rng = np.random.default_rng()
    return rng.choice(
        a= np.array([-1, 1]) * (np.sqrt(5) + np.array([-1, 1])) / 2, #TODO: #10 Should this divide the whole expression by 2 or just the second part
        size=n,
        replace=True,
        p = (np.sqrt(5) + np.array([1, -1])) / (2 * np.sqrt(5))
    )
    
def norm(n):
    rng = np.random.default_rng()
    return rng.normal(size=n)

def webb(n):
    rng = np.random.default_rng()
    return rng.choice(
        a = np.concatenate([-np.sqrt(np.array([3,2,1]) / 2), np.sqrt(np.array([1,2,3]) / 2)]),
        replace=True,
        size=n
    )
    
wild_draw_fun_dict = {
    'rademacher' : rademacher,
    'mammen' : mammen,
    'norm' : norm,
    'webb' : webb
}

  
def draw_weights(t : str, full_enumeration: bool, N_G_bootcluster: int, boot_iter: int) -> np.ndarray:
    """draw bootstrap weights

    Args:
        t (str): the type of the weights distribution. Either 'rademacher', 'mammen', 'norm' or 'webb'
        full_enumeration (bool): should deterministic full enumeration be employed
        N_G_bootcluster (int): the number of bootstrap clusters
        boot_iter (int): the number of bootstrap iterations

    Returns:
        np.ndarray: a matrix of dimension N_G_bootcluster x (boot_iter + 1)
    """    
    
    #TODO: we can use the `case` feature in python, but that's only available in 3.10+ will do a 3.7 version for now
    # Will take out this and make separate functions for readability
    
    wild_draw_fun = wild_draw_fun_dict.get(t)
    
    if wild_draw_fun is None:
        raise WildDrawFunctionException("Function type specified is not supported or there is a typo.")
  
    # do full enumeration for rademacher weights if bootstrap iterations
    # B exceed number of possible permutations else random sampling

    # full_enumeration only for rademacher weights (set earlier)
    if full_enumeration: 
        t = 0 # what is this needed for? 
        # with N_G_bootcluster draws, get all combinations of [-1,1] WITH 
        # replacement, in matrix form
        v0 = np.transpose(np.array(list(product([-1,1], repeat=N_G_bootcluster))))
    else:
        # else: just draw with replacement - by chance, some permutations
        # might occur more than once
        v0 = wild_draw_fun(n = N_G_bootcluster * boot_iter)
        v0 = v0.reshape(N_G_bootcluster, boot_iter) # weights matrix
    
    # update boot_iter (B) - only relevant in enumeration case
    boot_iter = v0.shape[1] 
    v = np.insert(v0, 0, 1,axis = 1)

    return [v, boot_iter]
  
  
def wildboottest(model, param, cluster, B, weights_type = 'rademacher',impose_null = True, bootstrap_type = '11', seed = None):
  
  '''
  Run a wild cluster bootstrap based on an object of class 'statsmodels.regression.linear_model.OLS'
  
  Args: 
    model(statsmodels.regression.linear_model.OLS'): A statsmodels regression object
    param(str): A string of length one, containing the test parameter of interest
    cluster(np.array): A numpy array of dimension one, containing the clustering variable.
    B(int): The number of bootstrap iterations to run
    weights_type(str): The type of bootstrap weights. Either 'rademacher', 'mammen', 'webb' or 'normal'. 
                       'rademacher' by default.
    impose_null(logical): Should the null hypothesis be imposed on the bootstrap dgp, or not?
                          True by default. 
    bootstrap_type(str). A string of length one. Allows to choose the bootstrap type 
                         to be run. Either '11', '31', '13' or '33'. '11' by default.
    seed(int). Option to provide a random seed. 
    
  Returns: 
    A wild cluster bootstrapped p-value. 
    
  Example: 
    
    import statsmodels.api as sm
    import numpy as np

    np.random.seed(12312312)
    N = 1000
    k = 10
    G= 10
    X = np.random.normal(0, 1, N * k).reshape((N,k))
    beta = np.random.normal(0,1,k)
    beta[0] = 0.005
    u = np.random.normal(0,1,N)
    Y = 1 + X @ beta + u
    cluster = np.random.choice(list(range(0,G)), N)

    model = sm.OLS(Y, X)

    boottest(model, param = "X1", cluster = cluster, B = 9999)
    
  '''

  # set param to lowercase? model.data.xnames all seem to be lowercase?
  param = str.lower(param)
  # does model.exog already exclude missing values?
  X = model.exog
  # interestingly, the dependent variable is called 'endogeneous'
  Y = model.endog
  # weights not yet used, only as a placeholder
  weights = model.weights
  
  xnames = model.data.xnames
  ynames = model.data.ynames
  
  R = np.zeros(len(xnames))
  R[xnames.index(param)] = 1
  
  # is it possible to fetch the clustering variables from the pre-processed data 
  # frame, e.g. with 'excluding' observations with missings etc
  # cluster = ...
  
  # set bootcluster == cluster for one-way clustering
  bootcluster = cluster
  
  boot = Wildboottest(X = X, Y = Y, cluster = cluster, bootcluster = bootcluster, R = R, B = B, seed = seed)
  boot.get_scores(bootstrap_type = bootstrap_type, impose_null = impose_null)
  boot.get_weights(weights_type = weights_type)
  boot.get_numer()
  boot.get_denom()
  boot.get_tboot()
  boot.get_vcov()
  boot.get_tstat()
  boot.get_pvalue(pval_type = "two-tailed")
  
  return boot.pvalue
  

