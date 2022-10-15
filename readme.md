## wildboottest.py

In this repo, we aim to develop a Python version of the wild cluster 
bootstrap as implemented in Stata's [boottest](https://github.com/droodman/boottest), R's [fwildclusterboot](https://github.com/s3alfisc/fwildclusterboot) or 
Julia's [WildBootTests.jl](https://github.com/droodman/WildBootTests.jl)
as a post-estimation command for [statsmodels](https://github.com/statsmodels/statsmodels) and 
[linearmodels](https://github.com/bashtage/linearmodels). 

If you'd like to cooperate, either send us an 
[email](alexander-fischer1801@t-online.de) or comment in the issues section!


## Example 

Note: everything is still very much work in progress. Still, I believe that the implementation of the WCR11 and WCU 11 is more or less correct.

```
from wildboottest.wildboottest import wildboottest, Wildboottest
import statsmodels.api as sm
import numpy as np
import timeit 
import time

N = 1000
k = 10
G= 12
X = np.random.normal(0, 1, N * k).reshape((N,k))
beta = np.random.normal(0,1,k)
beta[0] = 0.005
u = np.random.normal(0,1,N)
Y = 1 + X @ beta + u
cluster = np.random.choice(list(range(0,G)), N)
B = 99999


model = sm.OLS(Y, X)
model.exog
results = model.fit(cov_type = 'cluster', cov_kwds = {
   'groups': cluster
})
results.summary()
# >>> results.summary()
# <class 'statsmodels.iolib.summary.Summary'>
# """
#                                  OLS Regression Results                                
# =======================================================================================
# Dep. Variable:                      y   R-squared (uncentered):                   0.799
# Model:                            OLS   Adj. R-squared (uncentered):              0.797
# Method:                 Least Squares   F-statistic:                              790.3
# Date:                Sat, 15 Oct 2022   Prob (F-statistic):                    3.16e-14
# Time:                        12:03:43   Log-Likelihood:                         -1784.6
# No. Observations:                1000   AIC:                                      3589.
# Df Residuals:                     990   BIC:                                      3638.
# Df Model:                          10                                                  
# Covariance Type:              cluster                                                  
# ==============================================================================
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# x1             0.0128      0.064      0.200      0.841      -0.113       0.138
wildboottest(model, "X1", cluster, B)
# 0.8408408408408409


# execute, method by method

# some preparations
bootcluster = cluster
R = np.zeros(k)
R[0] = 1

start_time = timeit.default_timer()
wcr = wb.Wildboottest(X = X, Y = Y, cluster = cluster, bootcluster = bootcluster, R = R, B = B, seed = 12341)
wcr.get_scores(bootstrap_type = "11", impose_null = True)
wcr.get_weights(weights_type = "rademacher")
wcr.get_numer()
wcr.get_denom()
wcr.get_tboot()
wcr.get_vcov()
wcr.get_tstat()
wcr.get_pvalue(pval_type = "two-tailed")
print("estimation time:", timeit.default_timer() - start_time)
# >>> 0.9225496 seconds
print("p value:", wcr.pvalue)
# >>> p value: 0.8408408408408409
```
