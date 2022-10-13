def get_scores(bootstrap_type, N_G_bootcluster, tXgyg_list, tXgXg_list, beta_tilde):
  
  '''
  Returns a list of scores vectors (for all N_G bootclusters)
  '''
  
  scores_list = []
  
  if(bootstrap_type == "WCR1x"):
    
    for g in range(0,N_G_bootcluster):
      
      scores_list.append(tXgyg_list[g] - tXgXg_list[g] @ beta_tilde)
  
  else: 
    
    None
    
  return scores_list



def get_pvalue(t_stat, t_boot, pval_type):
  
  #assert pval_type in ["two-tailed", "equal-tailde", ">", "<"]
  
  if pval_type == "two-tailed":
    np.mean(np.abs(t_stat) < abs(t_boot))
  elif pval_type == "equal-tailed":
    pl = np.mean(t_stat < t_boot)
    ph = np.mean(t_stat > t_boot)
    2 * min(pl, ph)
  elif pval_type == ">":
    np.mean(t_stat < t_boot)
  else 
    np.mean(t_stat > t_boot)
  

