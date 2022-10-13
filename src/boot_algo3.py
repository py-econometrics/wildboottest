import numpy as np
import pandas as pd
from numba import jit

class Wildboottest: 
  
  '''
  Create an object of Wildboottest and get p-value by successively applying
  methods in the following way: 
    
  wb = Wildboottest(X = X, Y = y, clusters = clusters, R = R, B = B)
  wb.get_scores(bootstrap_type = "11", impose_null = True)
  wb.get_numer()
  wb.get_denom()
  wb.get_tboot()
  wb.get_vcov()
  wb.get_tstat()
  wb.get_pvalue()  
  
  Later we can replace X, Y, clusters, R with an object estimated via 
  statsmodels or linearmodels and a "param" values (as in fwildclusterboot::boottest())
  
  '''
  
  def __init__(self, X, Y, clusters, R, B, seed = None):
      
      "Initialize the wildboottest class"
      #assert bootstrap_type in ['11', '13', '31', '33']
      #assert impose_null in [True, False]

      if isinstance(X, pd.DataFrame):
        self.X = X.values
      if isinstance(y, pd.DataFrame):
        self.y = y.values
      if isinstance(clusters, pd.DataFrame):
        clustid = clusters.unique()
        self.clusters = clusters.values
      if isinstance(bootcluster, pd.DataFrame):
        bootclustid = bootcluster.unique()
        self.bootcluster = bootcluster.values
      else:
        clustid = np.unique(clusters)
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
      
      n_draws = (N_G_bootcluster * (B+1))
      v = np.random.choice([-1,1], int(n_draws))
      v = v.reshape((N_G_bootcluster, int(B + 1)))
      v[:,0] = 1
      self.v = v
      self.ssc = 1
      
      X_list = []
      y_list = []
      tXgXg_list = []
      tXgyg_list = []
      tXX = np.zeros((k, k))
      tXy = np.zeros(k)
      
      #all_clusters = np.unique(bootcluster)
      
      for g in bootclustid:
        
        # split X and Y by (boot)cluster
        X_g = X[np.where(bootcluster == g)]
        Y_g = y[np.where(bootcluster == g)]
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
      
  def get_scores(self, bootstrap_type, impose_null):
    
      if bootstrap_type[0:1] == '1':
        self.crv_type = "crv1"
      elif bootstrap_type[0:1] == '3':
        self.crv_type = "crv3"

      bootstrap_type_x = bootstrap_type[1:2] + 'x'

      if impose_null == True:
        self.bootstrap_type = "WCR" + bootstrap_type_x
      else:
        self.bootstrap_type = "WCU" + bootstrap_type_x
    
      # precompute required objects for computing scores & vcov's
      if self.bootstrap_type in ["WCR3x", "WCU3x"]: 
          
        X = self.X
        X1 = X[:,np.where(self.R == 0)]
        X1_list = []
        tX1gX1g = []
        tX1gyg = []
        tXgX1g = []
        tX1X1 = np.zeros((self.k-1, self.k-1))
        tX1y = np.array(self.k-1)
          
        for ix, g in enumerate(self.bootclustid):
          X1_list.append(X1[np.where(bootcluster == g)])
          tX1gX1g.append(np.transpose(X1_list[ix]) @ X1_list[ix])
          tX1gyg.append(np.transpose(X1_list[ix]) @ y_list[ix])
          tXgX1g.append(np.transpose(X_list[ix]) @  X1_list[ix])
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
            beta_g_hat.append(np.linalg.pinv(tXX - tXgXg_list[ix]) @ (tXy - tXgyg_list[ix]))
  
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
        M = tXgXg_list

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
      self.t_boot = t_boot[1:(B+1)] # drop first element - might be useful for comp. of

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

  def get_pvalue(self):
    
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

  
