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
  
