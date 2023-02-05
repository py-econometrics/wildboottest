import pytest
import pandas as pd
import statsmodels.formula.api as sm
from wildboottest.wildboottest import wildboottest
import numpy as np

@pytest.fixture
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

  return df

def test_results_from_same_seed():
    
    df = data(G = 20)
    model = sm.ols(formula='Y ~ X1 + X2', data=df)    

    cluster_list = [df.years, None]
    for x in cluster_list: 

        # same seed used in function -> same results
        a = wildboottest(model, param = "X1", cluster = x, B= 999, seed=11232198237961)
        b = wildboottest(model, param = "X1", cluster = x, B= 999, seed=11232198237961)
        pd.testing.assert_frame_equal(a,b)

        # random seed set outside of function and in function produce equal results
        # I suppose this will never work?
        np.random.seed(123)
        a1 = wildboottest(model, param = "X1", cluster = x, B= 999)
        b1 = wildboottest(model, param = "X1", cluster = x, B= 999, seed=123)
        pd.testing.assert_frame_equal(a1,b1)
        
        # random seed outside of function 2x -> same results
        np.random.seed(123)
        a2 = wildboottest(model, param = "X1", cluster = x, B= 999)
        np.random.seed(123)
        b2 = wildboottest(model, param = "X1", cluster = x, B= 999)
        pd.testing.assert_frame_equal(a2,b2)

