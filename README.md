## wildboottest

![PyPI](https://img.shields.io/pypi/v/wildboottest?label=pypi%20package)
![PyPI - Downloads](https://img.shields.io/pypi/dm/wildboottest)

`wildboottest` implements multiple fast wild cluster
bootstrap algorithms as developed in [Roodman et al
(2019)](https://econpapers.repec.org/paper/qedwpaper/1406.htm) and
[MacKinnon, Nielsen & Webb
(2022)](https://www.econ.queensu.ca/sites/econ.queensu.ca/files/wpaper/qed_wp_1485.pdf).

It has similar, but more limited functionality than Stata's [boottest](), R's [fwildcusterboot]() or Julia's [WildBootTests.jl](). It supports

-   The wild cluster bootstrap for OLS ([Cameron, Gelbach & Miller 2008](https://direct.mit.edu/rest/article-abstract/90/3/414/57731/Bootstrap-Based-Improvements-for-Inference-with),
    [Roodman et al (2019)](https://econpapers.repec.org/paper/qedwpaper/1406.htm)).
-   Multiple new versions of the wild cluster bootstrap as described in
    [MacKinnon, Nielsen & Webb (2022)](https://www.econ.queensu.ca/sites/econ.queensu.ca/files/wpaper/qed_wp_1485.pdf), including the WCR13, WCR31, WCR33,
    WCU13, WCU31 and WCU33.
-   CRV1 and CRV3 robust variance estimation, including the CRV3-Jackknife as 
    described in [MacKinnon, Nielsen & Webb (2022)](https://arxiv.org/pdf/2205.03288.pdf).
    
At the moment, `wildboottest` only computes wild cluster bootstrapped p-values, and no confidence intervals. 

Other features that are currently not supported: 

- The (non-clustered) wild bootstrap for OLS ([Wu, 1986](https://projecteuclid.org/journals/annals-of-statistics/volume-14/issue-4/Jackknife-Bootstrap-and-Other-Resampling-Methods-in-Regression-Analysis/10.1214/aos/1176350142.full)).
-   The subcluster bootstrap ([MacKinnon and Webb 2018](https://academic.oup.com/ectj/article-abstract/21/2/114/5078969?login=false)).
-   Confidence intervals formed by inverting the test and iteratively
    searching for bounds.
-   Multiway clustering.


Direct support for [statsmodels](https://github.com/statsmodels/statsmodels) and 
[linearmodels](https://github.com/bashtage/linearmodels) is work in progress.

If you'd like to cooperate, either send us an 
[email](alexander-fischer1801@t-online.de) or comment in the issues section!


## Installation 

You can install the package from `PyPi` by running 

```bash
pip install wildboottest
```

## Example 

```python
from wildboottest.wildboottest import wildboottest
import statsmodels.api as sm
import numpy as np
import pandas as pd

# create data
np.random.seed(12312312)
N = 1000
k = 10
G = 25
X = np.random.normal(0, 1, N * k).reshape((N,k))
X = pd.DataFrame(X)
X.rename(columns = {0:"X1"}, inplace = True)
beta = np.random.normal(0,1,k)
beta[0] = 0.005
u = np.random.normal(0,1,N)
Y = 1 + X @ beta + u
cluster = np.random.choice(list(range(0,G)), N)

# estimation
model = sm.OLS(Y, X)

wildboottest(model, param = "X1", cluster = cluster, B = 9999, bootstrap_type = "11")
#   param              statistic   p-value
# 0    X1  [-1.0530803154504016]  0.308831

wildboottest(model, param = "X1", cluster = cluster, B = 9999, bootstrap_type = "31")
#   param              statistic   p-value
# 0    X1  [-1.0530803154504016]  0.307631

wildboottest(model, param = "X1", cluster = cluster, B = 9999, bootstrap_type = "33")
#   param              statistic   p-value
# 0    X1  [-1.0394791020434824]  0.294286


wildboottest(model, cluster = cluster, B = 9999)
#   param              statistic   p-value
# 0    X1  [-1.0530803154504016]  0.315132
# 1     1    [-18.5149486170657]  0.000000
# 2     2    [7.831855813581191]  0.000000
# 3     3   [-16.85188951397906]  0.000000
# 4     4  [-12.721095348008182]  0.000000
# 5     5    [1.200524160940055]  0.243624
# 6     6    [6.870946666836135]  0.000000
# 7     7   [-31.31653422266621]  0.000000
# 8     8    [10.26443257212472]  0.000000
# 9     9  [-20.650361366939535]  0.000000
```
