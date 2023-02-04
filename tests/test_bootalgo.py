import pytest
import numpy as np

from wildboottest.wildboottest import wildboottest, WildboottestCL
from wildboottest.weights import wild_draw_fun_dict
import statsmodels.api as sm
import numpy as np
import pandas as pd

# rpy2 imports
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

fwildclusterboot = importr("fwildclusterboot")
stats = importr('stats')

ts = list(wild_draw_fun_dict.keys())


def data(G):
  np.random.seed(12312)
  N = 1000
  k = 3
  # small sample size -> full enumeration
  X = np.random.normal(0, 1, N * k).reshape((N,k))
  X[:,0] = 1
  beta = np.random.normal(0,1,k)
  beta[1] = 0.005
  u = np.random.normal(0,1,N)
  Y = X @ beta + u
  cluster = np.random.choice(list(range(0,G)), N)
  X_df = pd.DataFrame(X)
  Y_df = pd.DataFrame(Y)
  cluster_df = pd.DataFrame(cluster)
  df = pd.concat([X_df, Y_df, cluster_df], axis = 1)  
  df.columns = ['intercept','X1','X2','Y', 'cluster']
  B = 99999

  return df, B


def hc_vs_cluster_bootstrap(): 

  '''
  compare results from HC vs cluster bootstrap
  '''

  def reldiff(x, y): 
      1 - x / y 

  df, B = data(5)
  N = df.shape[0]
  cluster = pd.Series(range(0, N))
  X = df[['intercept', 'X1', 'X2']]
  Y = df['Y']
  R = np.array([0,1,0])

  fit = sm.OLS(Y, X)
  # all allowed types
  cl = wildboottest(fit, param = "X1", cluster = cluster, bootstrap_type='11')

  types = ['11', '22', '33']

  for type in types: 

      hc = wildboottest(fit, param = "X1", bootstrap_type=type)

      assert reldiff(hc.xs('X1')[0], cl.xs('X1')[0])
      assert reldiff(hc.xs('X1')[1], cl.xs('X1')[1])


def test_r_vs_py_deterministic():
  
  '''
  test compares bootstrapped t-statistics for R and Python 
  versions in the full enumeration case. Under full enum, 
  the weights matrices are identical (up to the ordering 
  of columns), and therefore bootstrap t-statistics need to be
  *exactly* identical (if the same small sample correction 
  is applied). 
  '''
  # based on data created via the development_notebook.Rmd
  # with B = 99999 bootstrap iterations, WCR11
  # automate this via rpy2 (?) or add reproducible R and Python scripts
  # after wildboottest.py has nice interface for statsmodels/linearmodels
  # to reproduce: search for the commit, run dev notebookm run WCR11 in python
  # etc ...
  
  df, B = data(5)
  cluster = df['cluster']
  X = df[['intercept', 'X1', 'X2']]
  Y = df['Y']
  R = np.array([0,1,0])

  # convert df to an R dataframe
  with localconverter(ro.default_converter + pandas2ri.converter):
    r_df = ro.conversion.py2rpy(df)

  r_model = stats.lm("Y ~ X1 + X2", data=r_df)
  R = np.array([0,1,0])
  
  boot_tstats = []
  fwildclusterboot_boot_tstats = []
  
  for bootstrap_type in ['11', '31', '13', '33']: 
    for impose_null in [True, False]:
      # python implementation
      boot = WildboottestCL(X = X, Y = Y, cluster = cluster, bootcluster = cluster, R = R, B = B, seed = 12341)
      boot.get_scores(bootstrap_type = bootstrap_type, impose_null = impose_null)
      boot.get_weights(weights_type = "rademacher")
      boot.get_numer()
      boot.get_denom()
      boot.get_tboot()
      boot.get_vcov()
      boot.get_tstat()
      boot.get_pvalue(pval_type = "two-tailed")
      boot_tstats.append(boot.t_boot)
      
      # R implementation
      r_t_boot = fwildclusterboot.boottest(
        r_model,
        param = "X1",
        clustid = ro.Formula("~cluster"),
        B=99999,
        bootstrap_type=bootstrap_type,
        impose_null=impose_null
      )
      
      fwildclusterboot_boot_tstats.append(list(r_t_boot.rx2("t_boot")))
      
  df = pd.DataFrame(np.transpose(np.array(boot_tstats)))
  df.columns = ['WCR11', 'WCU11', 'WCR31', 'WCU31', 'WCR13', 'WCU13', 'WCR33', 'WCU33']
  
  # r_df = pd.read_csv("data/test_df_fwc_res.csv")[['WCR11', "WCR31", "WCU11", "WCU31"]]
  r_df = pd.DataFrame(np.transpose(np.array(fwildclusterboot_boot_tstats)))
  r_df.columns = ['WCR11', 'WCU11', 'WCR31', 'WCU31', 'WCR13', 'WCU13', 'WCR33', 'WCU33']
  
  # all values need to be sorted
  print("Python")
  print(df.sort_values(by=list(df.columns),axis=0).head())
  print("\n")
  print("R")
  print(r_df.sort_values(by=list(r_df.columns),axis=0).head())  
  
  def mse(x, y):
    return np.mean(np.power(x - y, 2))
  
  assert mse(df['WCR11'].sort_values(), r_df['WCR11'].sort_values()) < 1e-15
  assert mse(df['WCU11'].sort_values(), r_df['WCU11'].sort_values()) < 1e-15
  assert mse(df['WCR31'].sort_values(), r_df['WCR31'].sort_values()) < 1e-15
  assert mse(df['WCU31'].sort_values(), r_df['WCU31'].sort_values()) < 1e-15
  assert mse(df['WCR13'].sort_values(), r_df['WCR13'].sort_values()) < 1e-15
  assert mse(df['WCU13'].sort_values(), r_df['WCU13'].sort_values()) < 1e-15
  assert mse(df['WCR33'].sort_values(), r_df['WCR33'].sort_values()) < 1e-15
  assert mse(df['WCU33'].sort_values(), r_df['WCU33'].sort_values()) < 1e-15

def test_r_vs_py_stochastic():
  
  '''
  test compares bootstrapped inference for R and Python 
  versions for large B. 
  p-values, confidence intervals (tba) should be identical
  (difference converges to 0 in probability).
  Non-bootstrapped test statistics should be exactly equal 
  given the same small sample adjustments are applied
  '''

  df, B = data(25)
  cluster = df['cluster']
  X = df[['intercept', 'X1', 'X2']]
  Y = df['Y']
  R = np.array([0,1,0])
   
  # convert df to an R dataframe
  with localconverter(ro.default_converter + pandas2ri.converter):
    r_df = ro.conversion.py2rpy(df)

  r_model = stats.lm("Y ~ X1 + X2", data=r_df)
  R = np.array([0,1,0])


  boot_pvals = []
  fwildclusterboot_boot_pvals = []

  for bootstrap_type in ['11', '31', '13', '33']: 
    for impose_null in [True, False]:
      for weights_type in ['rademacher','mammen', 'webb','norm']:
        for pval_type in ['two-tailed', 'equal-tailed', '>', '<']:
          
          # python implementation
          boot = WildboottestCL(X = X, Y = Y, cluster = cluster, bootcluster = cluster, R = R, B = B, seed = 12341)
          boot.get_scores(bootstrap_type = bootstrap_type, impose_null = impose_null)
          boot.get_weights(weights_type = weights_type)
          boot.get_numer()
          boot.get_denom()
          boot.get_tboot()
          boot.get_vcov()
          boot.get_tstat()
          boot.get_pvalue(pval_type = pval_type)
          boot_pvals.append(boot.pvalue)
          
          # R implementation
          r_t_boot = fwildclusterboot.boottest(
            r_model,
            param = "X1",
            clustid = ro.Formula("~cluster"),
            B=99999,
            bootstrap_type=bootstrap_type,
            impose_null=impose_null,
            p_val_type = pval_type, 
            type = weights_type,
            ssc=fwildclusterboot.boot_ssc(adj=False, cluster_adj=False)
          )

          # test condition ... 
          fwildclusterboot_boot_pvals.append(list(r_t_boot.rx2("p_val")))
      
  df = pd.DataFrame(np.transpose(np.array(boot_pvals)), columns=['p_val'],
                    index=pd.MultiIndex.from_product([
                      ['11', '31', '31', '33'],
                      [True, False],
                      ['rademacher','mammen', 'webb','norm'],
                      ['two-tailed', 'equal-tailed', '>', '<']
                    ]))
  
  # r_df = pd.read_csv("data/test_df_fwc_res.csv")[['WCR11', "WCR31", "WCU11", "WCU31"]]
  r_df = pd.DataFrame(np.array(fwildclusterboot_boot_pvals), columns=['p_val'],
                      index=pd.MultiIndex.from_product([
                      ['11', '31', '31', '33'],
                      [True, False],
                      ['rademacher','mammen', 'webb','norm'],
                      ['two-tailed', 'equal-tailed', '>', '<']
                    ]))
  print(df.to_markdown())
  print(r_df.to_markdown())
  
  assert all(np.isclose(df.values, r_df.values, rtol=1e-2, atol=1e-2))

def test_error_warnings():
  '''
  test that errors and warnings are thrown when appropriate for 
  both the statsmodels interface and the Wildboottest method, e.g.
  - that an error is thrown when regression weights are used
  - ... other things supported by statsmodels.OLS but not wildboottest
    are tried
  - know edge cases that lead to errors provide valuable info (e.g. WCR with 
    one param regressions)
  '''

def test_data_is_list():
  
  df, B = data(15)
  cluster = df['cluster'].values.tolist()
  X = df[['intercept', 'X1', 'X2']].values.tolist()
  Y = df['Y'].values.tolist()
  R = np.array([0,1,0])

  with pytest.raises(TypeError):
    WildboottestCL(X = X, Y = Y, cluster = cluster, bootcluster = cluster, R = R, B = B, seed = 12341)

#@pytest.mark.skip(reason="exhaustive runtime")
#def test_seeds():
#  
#  df, B = data(15)
#  cluster = df['cluster']
#  X = df[['intercept', 'X1', 'X2']]
#  Y = df['Y']
#  R = np.array([0,1,0])
#  
#  results_dict = []
#
#  for s in range(1,10000):
#    for w in ts:
#      boot = WildboottestCL(X = X, Y = Y, cluster = cluster, bootcluster = cluster, R = R, B = B, seed = s)
#      boot.get_scores(bootstrap_type = "11", impose_null = True)
#      boot.get_weights(weights_type = w)
#      boot.get_numer()
#      boot.get_denom()
#      boot.get_tboot()
#      boot.get_vcov()
#      boot.get_tstat()
#      boot.get_pvalue(pval_type = "two-tailed")
#      results_dict.append(boot.pvalue)
#        
#  results_series = pd.Series(results_dict)
#  mapd = results_series.mad() / results_series.mean()
#  assert  mapd <= .1 # make sure mean absolute percentage deviation is less than 10% (ad hoc)
#  
# if __name__ == '__main__':
#   test_r_vs_py_stochastic()
#   test_r_vs_py_deterministic()
#   test_error_warnings()
