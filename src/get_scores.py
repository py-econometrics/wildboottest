def get_scores(bootstrap_type, N_G_bootcluster, tXgyg_list, tXgXg_list, beta_tilde, beta_hat, beta_1g_tilde, beta_g_hat):
  
  """
  Compute score vectors for all clusters
  
  Args: 
    bootstrap_type(str): The type of the bootstrap. Either WCR1x, WCR3x, WCU1x 
                         or WCU3x
    N_G_bootcluster(int): The number of bootstrapping clusters
    tXgYg_list (List): List of tXg @ yg for grouped by all clusters
    tXgXg_list (List or None): List of tXg @ Xg for all g or None if not required
    beta_tilde (np.array or None): Restricted Least Square estimate. Only required
                                   for WCR1x, else None.
    beta_hat(np.array or None): OLS Estimate. Only required for WCU1x, else None.
    beta_1g_tilde(List of np.arrays or None): Restricted Least Square, estimated
                                              independently for each cluster, 
                                              collected in a list. Only required 
                                              for WCR3x, else None. 
    beta_g_hat(List of np.arrays or None): OLS Estimate, estimated independently
                                           for each cluster,collected in a list.
                                           Only required for WCU3x, else None. 

  Returns 
    A list of np.arrays containing score vectors for each cluster. 
  """
  
  # General comment: Lots of the things here can be abstracted, i.e. 
  # all scores are computed as A - B @ c with A fixed among all bootstrap types
  # and varying B, c
  
  scores_list = []
  
  if(bootstrap_type == "WCR1x"):
    
    for g in range(0,N_G_bootcluster):
      
      scores_list.append(tXgyg_list[g] - tXgXg_list[g] @ beta_tilde)
  
  elif bootstrap_type == "WCU1x:
    
    for g in range(0,N_G_bootcluster):

      scores_list.append(tXgyg_list[g] - tXgXg_list[g] @ beta_hat)
      
  elif bootstrap_type == "WCR3x":
    
    for g in range(0,N_G_bootcluster):

      scores_list.append(tXgyg_list[g] - tXgX1g_list[g] @ beta_1g_tilde[g])

  elif bootstrap_type == "WCU3x":
    
    for g in range(0,N_G_bootcluster):

      scores_list.append(tXgyg_list[g] - tXgXg_list[g] @ beta_g_hat[g])

  return scores_list


