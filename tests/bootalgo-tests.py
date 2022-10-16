import pytest
import numpy as np

def bootstrap_types_diff_restults():
  
  '''
  Do the different bootstrap types lead to similar, but 
  not exactly identical results?
  '''
  
  N = 1000
  k = 2
  G= 25
  X = np.random.normal(0, 1, N * k).reshape((N,k))
  beta = np.random.normal(0,1,k)
  beta[0] = 0.1
  u = np.random.normal(0,1,N)
  y = 1 + X @ beta + u
  cluster = np.random.choice(list(range(0,G)), N)
  bootcluster = cluster
  R = np.zeros(k)
  R[0] = 1
  B = 999

  bootstrap_types = ['11', '31']
  impose_nulls = [True, False]
  res_all = []
  
  # fix seed 
  for bootstrap_type in bootstrap_types:
    for impose_null in impose_nulls:
        boot = Wildboottest(X = X, Y = y, cluster = cluster, bootcluster = bootcluster, R = R, B = 99999, seed = 12341)
        boot.get_scores(bootstrap_type = bootstrap_type, impose_null = impose_null)
        boot.get_weights(weights_type = "rademacher")
        boot.get_numer()
        boot.get_denom()
        boot.get_tboot()
        boot.get_vcov()
        boot.get_tstat()
        boot.get_pvalue(pval_type = "two-tailed")   
        res_all.append(boot)
  
  # 1) test that values that should (not) be identical are (not) identical    
  for x in range(0, len(res_all) - 1):
    
    assert not np.array_equal(res_all[x].scores_mat, res_all[x+1].scores_mat)
    assert not np.array_equal(res_all[x].t_boot, res_all[x+1].t_boot)
    assert np.array_equal(res_all[x].t_stat, res_all[x+1].t_stat)
    assert not np.array_equal(res_all[x].pvalue, res_all[x+1].pvalue) # unless both pvals are zero or 1...


def test_r_vs_py():
  
  # based on data created via the development_notebook.Rmd
  # with B = 99999 bootstrap iterations, WCR11
  # automate this via rpy2 (?) or add reproducible R and Python scripts
  # after wildboottest.py has nice interface for statsmodels/linearmodels
  # to reproduce: search for the commit, run dev notebookm run WCR11 in python
  # etc ...
  fwildclusterboot_pval = 0.499435 
  wildboottest_py_pval = 0.49314931493149317

  
  
def full_enum_works():
  
  N = 1000
  k = 10
  G= 4
  X = np.random.normal(0, 1, N * k).reshape((N,k))
  beta = np.random.normal(0,1,k)
  beta[0] = 0.005
  u = np.random.normal(0,1,N)
  Y = 1 + X @ beta + u
  cluster = np.random.choice(list(range(0,G)), N)
  bootcluster = cluster
  R = np.zeros(k)
  R[0] = 1
  B = 99999
  
  wcr = Wildboottest(X = X, Y = Y, cluster = cluster, bootcluster = bootcluster, R = R, B = B, seed = 12341)
  boot.get_scores(bootstrap_type = "11", impose_null = False)
  boot.get_weights(weights_type = "rademacher")
  boot.get_numer()
  boot.get_denom()
  boot.get_tboot()
  
  assert len(boot.t_boot) == 2**G
  assert boot.full_enumeration == True
  




