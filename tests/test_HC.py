import pytest
import numpy as np

from wildboottest.wildboottest import  WildboottestHC
from wildboottest.weights import wild_draw_fun_dict
import statsmodels.api as sm
import numpy as np
import pandas as pd

# rpy2 imports
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.vectors import StrVector, FloatVector
pandas2ri.activate()


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


def test_r_vs_py_heteroskedastic_stochastic():

    '''
    test compares bootstrapped p-values for non-clustered errors 
    for R (fwildclusterboot) and Python (wildboottest)
    '''

    df, B = data(20)
    X = df[['intercept', 'X1', 'X2']]
    Y = df['Y']
      
    r_model = stats.lm("Y ~ X1 + X2", data=df)

    boot_pvals = []
    #boot_tstats = []
    fwildclusterboot_boot_pvals = []

    i = 0

    for bootstrap_type in ['11']: 
      for impose_null in [True]:
        for weights_type in ["rademacher"]:
          for pval_type in ['two-tailed', 'equal-tailed', '>', '<']:
  
            #if i % 2: 
            #  r = 0.02
            #else: 
            #  r = 0

            #if i % 4: 
            #  R = np.array([0.2, 0.1, 0])  
            #else: 
            #  R = np.array([1, 0, 0])
            
            # only test X1 as long as multi-param hypotheses not supported
            # by fwildclusterboot

            R = np.array([0, 1, 0])
            r = 0

            #i += 1

            boot = WildboottestHC(X = X, Y = Y, R = R, r = r, B = B, seed = 12341)
            boot.get_adjustments(bootstrap_type = bootstrap_type)
            boot.get_uhat(impose_null = impose_null)
            boot.get_tboot(weights_type = weights_type)
            boot.get_tstat()
            boot.get_pvalue(pval_type = pval_type)  
            boot_pvals.append(boot.pvalue)
            #boot_tstats.append(boot.tstats)

            r_t_boot = fwildclusterboot.boottest(
              r_model,
                param = StrVector(["X1"]),
                B=B,
                #R=FloatVector(R), 
                r=r, 
                bootstrap_type=bootstrap_type,
                impose_null=impose_null,
                p_val_type = pval_type, 
                type = weights_type,
                ssc=fwildclusterboot.boot_ssc(adj=False, cluster_adj=False)
            )

  
            # test condition ... 
            fwildclusterboot_boot_pvals.append(list(r_t_boot.rx2("p_val")))
      
    df = pd.DataFrame(
      np.transpose(np.array(boot_pvals)), 
      columns=['p_val'],
      index=pd.MultiIndex.from_product([
                        ['11'],
                        [True],
                        ['rademacher'],
                        ['two-tailed', 'equal-tailed', '>', '<']
                      ])
    )
  
    # r_df = pd.read_csv("data/test_df_fwc_res.csv")[['WCR11', "WCR31", "WCU11", "WCU31"]]
    r_df = pd.DataFrame(
      np.array(fwildclusterboot_boot_pvals), 
      columns=['p_val'],
      index=pd.MultiIndex.from_product([
                        ['11'],
                        [True],
                        ['rademacher'],
                        ['two-tailed', 'equal-tailed', '>', '<']
                      ])
      )
    print(df.to_markdown())
    print(r_df.to_markdown())
      
    assert all(np.isclose(df.values, r_df.values, rtol=1e-2, atol=1e-2))