import numpy as np

def boot_algo3(X, y, bootstrap_type, N_G_bootcluster, R, impose_null): 
  
  assert bootstrap_type in ['11', '13', '31', '33']
  assert if isinstance(X, np.ndarray)
  assert if isinstance(y, np.ndarray)

  if(bootstrap_type[0:1] == '1'):
    crv_type = "crv1"
  elif(bootstrap_type[0:1] == '3'): 
    crv_type = "crv3"
    
  bootstrap_type_x = bootstrap_type[1:2] + 'x'
  
  if(impose_null == True):
    bootstrap_type = "WCR" + bootstrap_type_x
  else: 
    bootstrap_type = "WCU" + bootstrap_type_x

  # write dedicated get_weights() function
  v = np.random.normal(N_G, B+1)
  
  X_list = []
  y_list = []
  XgXg_list = []
  Xgyg_list = []
  tXX = np.zeros((k, k))
  tXy = np.zeros((N,1))
  
  for g in np.unique(bootcluster):
    
    X_g = X[bootcluster == g, :]
    Y_g = y[bootcluster == g, :]
    tXgXg = np.matmul(np.transpose(X_g), X_g)
    tXgyg = np.matmul(np.transpose(X_g), Y_g)
    
    X_list.append(X_g)
    y_list.append(y_g)
    XgXg_list.append(tXgXg)
    Xgyg_list.append(tXgyg)
    tXX = tXX + XgXg
    tXy = tXy + tXgyg
    
  tXXinv = np.invert(tXX)
  RtXXinv = np.matmul(R, tXXinv)

  tXgX1g = None 
  beta_hat = None
  beta_tilde = None
  beta_g_hat = None
  beta_1g_tilde = None
  inv_tXX_tXgXg = None
  
  
  # pre-compute required objects for different bootstrap types: 
  
    if(bootstrap_type in ["WCR3x", "WCU3x"]: 
      
      X1 = X[:,np.where(R == 0)]
      X1_list = []
      tX1gX1g = []
      tX1gyg = []
      tXgX1g = []
      tX1X1 = np.array()
      tX1y = np.array()
      
    for g in np.unique(bootcluster):
        X1_list.append(X1[bootcluster == g,:])
        # tX1gX1g.append(np.matmul(np.transpose(X1_list[g]), X1_list[g])
        # tX1gyg.append(np.matmul(np.transpose(X1_list[x], y_list[x])))
        # tXgX1g.append(np.transpose(X_list[x]), X1_list[x])
        # tX1X1 = tX1X1 + tX1gX1g[x]
        # tX1y = tX1y + tX1gyg[x]
        
      tX1X1inv = np.invert(tX1X1)
  
    elif(bootstrap_type in ["WCR1x"]): 
      
      beta_hat = np.matmul(tXXinv, tXy)
      beta_tilde = beta_hat - np.matmul(np.matmul(tXXinv, R), np.invert())
      
    elif(bootstrap_type in "WCU1x): 
      
    elif(bootstrap_type in "WCR3x"):
      
    elif(bootstrap_type in "WCU3x"): 
      
      
  # precomputed required objects for CRV-types
  
  
      

  
  
  
  
  
  

  
