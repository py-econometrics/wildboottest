import numpy as np
import pandas as pd

def boot_algo3(X, y, bootstrap_type, N_G_bootcluster, R, impose_null, clusters): 

  assert bootstrap_type in ['11', '13', '31', '33']

  if isinstance(X, pd.DataFrame):
    X = X.values
  if isinstance(y, pd.DataFrame):
    y = y.values
  if isinstance(clusters, pd.DataFrame):
    clustid = clusters.unique()
    clusters = clusters.values
  if isinstance(bootcluster, pd.DataFrame):
    bootclustid = bootcluster.unique()
    bootcluster = bootcluster.values
  else:
    clustid = np.unique(clusters)
    bootclustid = np.unique(bootcluster)

  #TODO: what is fe supposed to be? 
  #fe <- preprocessed_object$fe

  #TODO: assume pandas dataframe and then use pd.Categorical?
  #cluster <- as.factor(cluster_df[,1])
  G  = N_G_bootcluster 

  #TODO: Might need to be changed
  k = R.shape[0]

  if bootstrap_type[0:1] == '1':
    crv_type = "crv1"
  elif bootstrap_type[0:1] == '3': 
    crv_type = "crv3"

  bootstrap_type_x = bootstrap_type[1:2] + 'x'

  if impose_null == True:
    bootstrap_type = "WCR" + bootstrap_type_x
  else: 
    bootstrap_type = "WCU" + bootstrap_type_x

  # write dedicated get_weights() function
  # TODO: add get_weights here, once merged
  n_draws = (N_G_bootcluster * (B+1))
  v = np.random.choice([-1,1], int(n_draws))
  v = v.reshape((N_G_bootcluster, int(B + 1)))

  X_list = []
  y_list = []
  tXgXg_list = []
  tXgyg_list = []
  tXX = np.zeros((k, k))
  tXy = np.zeros(k)
  
  #all_clusters = np.unique(bootcluster)
  
  for g in bootclustid:
    
    # split X and Y by (boot)cluster
    X_g = X[np.where(bootcluster == g, 1, 0),:]
    Y_g = y[np.where(bootcluster == g, 1, 0),:]
    tXgXg = np.transpose(X_g) @ X_g
    tXgyg = np.transpose(X_g) @ Y_g
    X_list.append(X_g)
    y_list.append(Y_g)
    tXgXg_list.append(tXgXg)
    tXgyg_list.append(tXgyg)
    tXX = tXX + tXgXg
    tXy = tXy + tXgyg
    
  tXXinv = np.linalg.inv(tXX)
  RtXXinv = np.matmul(R, tXXinv)

  tXgX1g = None 
  beta_hat = None
  beta_tilde = None
  beta_g_hat = None
  beta_1g_tilde = None
  inv_tXX_tXgXg = None
  
  #np.matmul(tXXinv, tXy).shape
  
  
  # pre-compute required objects for different bootstrap types: 
  
  if(bootstrap_type in ["WCR3x", "WCU3x"]: 
      
    X1 = X[:,np.where(R == 0)]
    X1_list = []
    tX1gX1g = []
    tX1gyg = []
    tXgX1g = []
    tX1X1 = np.zeros((k-1, k-1))
    tX1y = np.array(k-1)
      
    for ix, g in enumerate(bootclustid):
        X1_list.append(np.where(X1[bootcluster == g, 1, 0),:])
        tX1gX1g.append(np.transpose(X1_list[ix]) @ X1_list[ix])
        tX1gyg.append(np.transpose(X1_list[ix]) @ y_list[ix])
        tXgX1g.append(np.transpose(X_list[ix]) @  X1_list[ix])
        tX1X1 = tX1X1 + tX1gX1g[ix]
        tX1y = tX1y + tX1gyg[ix]
        
    tX1X1inv = np.invert(tX1X1)
  
  elif(bootstrap_type in ["WCR1x"]): 
      
    beta_hat = tXXinv @ tXy
    A = np.linalg.inv(np.transpose(R) @ tXXinv @ R)
    beta_tilde = tXXinv @ R @ A @ (R @ beta_hat - 0)
    
  elif(bootstrap_type in "WCU1x): 
    
    beta_hat = np.matmul(tXXinv, tXy)

  elif(bootstrap_type in "WCR3x"):
    
    inv_tXX_tXgXg = []
    beta_1g_tilde = []
    
      for ix, g in enumerate(bootclustid):
      # use generalized inverse 
      inv_tXX_tXgXg.append(np.linalg.pinv(tXX - tXgXg_list[ix]))
      beta_1_tilde = np.linalg.pinv(tX1X1 - tX1gX1g_list[ix]) @ (tX1y - tX1gyg_list[ix])
      beta_1g_tilde.append()
      
  elif(bootstrap_type in "WCU3x"): 
      
      beta_g_hat = []
      for ix, g in enumerate(bootclustid):
        beta_g_hat.append(np.linalg.pinv(tXX - tXgXg_list[ix]) @ (tXy - tXgyg_list[ix]))
  # precomputed required objects for CRV-types
  
  scores_list = get_scores(bootstrap_type, G, tXgyg, tXgXg, beta_tilde)
      
    
  if crv_type in ["crv1"]:
    # Ag no longer needed, should be deleted from R code!
    Ag = None
  elif crv_type in ["crv3"]:
    # might have been computed already above
    if inv_tXX_tXgXg is not None: 
      for ix, g in enumerate(bootclustid):
        np.linalg.pinv(tXX - tXgXg_list[ix])
    
  # Calculate the bootstrap numerator
  Cg = np.matmul(np.matmul(R, tXXinv), np.array(scores_list))
  numer = np.matmul(Cg, v)
  
  if crv_type == "crv1":
    
    H = np.zeros((G, G))
    
    for ixg, g in enumerate(bootclustid):
      for ixh, h in enumerate(bootclustid):
        H[ixg,ixh] = R @ tXXinv @ tXgXg[ixg] @ tXXinv @ scores_list[ixh]
  
    # now compute denominator
    Zg = np.array(G)
    for ixg, g in enumerate(bootclustid):
      vH = 0
      for ixh, h in enumerate(bootclustid):
        vH = vH + v(ixh,b) * H(ixg,ixh)
      Zg[ixg] = Cg[ixg] * v[ixg,b] - vH
      
    denom = ssc * np.sum(np.power(Zg,2))
    
  else: 
    None
    
  # compute the t-statistics
  t_stat = numer / denom
  
  # compute the original t-stat
  
  
  # compute the bootstrapped p-value
  

  
  
      

  
  
  
  
  
  

  
