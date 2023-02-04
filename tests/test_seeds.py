import pytest
import pandas as pd
import statsmodels.formula.api as sm
from wildboottest.wildboottest import wildboottest
import numpy as np

@pytest.fixture
def data():
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
    
    return X, y, cluster, bootcluster, R, B

def test_results_from_same_seed():
    
    df = pd.read_csv("https://raw.github.com/vincentarelbundock/Rdatasets/master/csv/sandwich/PetersenCL.csv")
    df['treat'] = np.random.choice([0, 1], df.shape[0], True)
    model = sm.ols(formula='y ~ treat', data=df)    

    cluster_list = [df.years, None]:   
    for x in cluster_list: 
        
        # same seed used in function -> same results
        a = wildboottest(model, param = "treat", cluster = x, B= 999, seed=11232198237961)
        b = wildboottest(model, param = "treat", cluster = x, B= 999, seed=11232198237961)
        pd.testing.assert_frame_equal(a,b)

        # random seed set outside of function and in function produce equal results
        np.random.default_rng(123)
        a1 = wildboottest(model, param = "treat", cluster = x, B= 999)
        b1 = wildboottest(model, param = "treat", cluster = x, B= 999, seed=123)
        pd.testing.assert_frame_equal(a1,b1)
        
        # random seed outside of function 2x -> same results
        np.random.default_rng(123)
        a2 = wildboottest(model, param = "treat", cluster = x, B= 999)
        np.random.default_rng(123)
        b2 = wildboottest(model, param = "treat", cluster = x, B= 999)
        pd.testing.assert_frame_equal(a2,b2)

