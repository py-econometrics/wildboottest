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
    results_dict = {}
    
    np.random.seed(123)
    a = wildboottest(model, param = "treat", cluster = df.year, B= 999)
    np.random.seed(123)
    b=wildboottest(model, param = "treat", cluster = df.year, B= 999)
    print(a)
    print(b)

    assert 1==2
    