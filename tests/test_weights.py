import pytest
from wildboottest.weights import WildDrawFunctionException, draw_weights, wild_draw_fun_dict
from wildboottest.wildboottest import Wildboottest
import numpy as np
import pandas as pd

ts = list(wild_draw_fun_dict.keys())
full_enum = [True, False]
ng_bootclusters = list(range(0,100, 10))
boot_iter = list(range(0,1000,400))

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

def test_none_wild_draw_fun():
    with pytest.raises(WildDrawFunctionException):
        draw_weights(None, True, 1,1)

def test_string_not_avail_wild_draw_fun():
    with pytest.raises(WildDrawFunctionException):
        draw_weights('something weird', True, 1,1)
        
def test_wrong_type_wild_draw_fun():
    with pytest.raises(ValueError):
        draw_weights([1], True, 1,1)
        
def test_different_weights(data):
    
    X, y, cluster, bootcluster, R, B = data
    
    results_dict = {}

    for w in ts:
        boot = Wildboottest(X = X, Y = y, cluster = cluster, bootcluster = bootcluster, R = R, B = 99999, seed = 12341)
        boot.get_scores(bootstrap_type = "11", impose_null = True)
        boot.get_weights(weights_type = w)
        boot.get_numer()
        boot.get_denom()
        boot.get_tboot()
        boot.get_vcov()
        boot.get_tstat()
        results_dict[w] = boot.get_pvalue(pval_type = "two-tailed")
        
    df = pd.DataFrame(results_dict)
    
    
