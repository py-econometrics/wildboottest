import pytest
import numpy as np

def WCR11_not_WCU11():
  
  N = 100
  k = 2
  G= 20
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

  wcr = wb.Wildboottest(X = X, Y = y, cluster = cluster, bootcluster = bootcluster, R = R, B = 99999, seed = 12341)
  wcr.get_scores(bootstrap_type = "11", impose_null = True)
  wcr.get_weights(weights_type = "rademacher")
  wcr.get_numer()
  wcr.get_denom()
  wcr.get_tboot()
  wcr.get_vcov()
  wcr.get_tstat()
  wcr.get_pvalue(pval_type = "two-tailed")
  
  wcu = wb.Wildboottest(X = X, Y = y, cluster = cluster, bootcluster = bootcluster, R = R, B = 99999, seed = 12341)
  wcu.get_scores(bootstrap_type = "11", impose_null = False)
  wcu.get_weights(weights_type = "rademacher")
  wcu.get_numer()
  wcu.get_denom()
  wcu.get_tboot()
  wcu.get_vcov()
  wcu.get_tstat()
  wcu.get_pvalue(pval_type = "two-tailed")

  # score matrices of WCR11 and WCU11 should be different - currently not the case
  assert not np.array_equal(wcr.scores_mat, wcu.scores_mat)
  assert not np.array_equal(wcr.t_boot, wcu.t_boot)
  assert np.array_equal(wcr.t_stat, wcu.t_stat)
  assert not np.array_equal(wcr.pvalue, wcu.pvalue) # unless both pvals are zero or 1...


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
  wcr.get_scores(bootstrap_type = "11", impose_null = False)
  wcr.get_weights(weights_type = "rademacher")
  wcr.get_numer()
  wcr.get_denom()
  wcr.get_tboot()
  
  assert len(wcr.t_boot) == 2**G
  assert wcr.full_enumeration == True
  




