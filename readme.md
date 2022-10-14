## wildboottest.py

In this repo, we aim to develop a Python version of the wild cluster 
bootstrap as implemented in Stata's [boottest](https://github.com/droodman/boottest), R's [fwildclusterboot](https://github.com/s3alfisc/fwildclusterboot) or 
Julia's [WildBootTests.jl](https://github.com/droodman/WildBootTests.jl)
as a post-estimation command for [statsmodels](https://github.com/statsmodels/statsmodels) and 
[linearmodels](https://github.com/bashtage/linearmodels). 

If you'd like to cooperate, either send us an 
[email](alexander-fischer1801@t-online.de) or comment in the issues section!


## Example 

Note: everything is still very much work in progress, and there are multiple errors in the code that I am aware of. Still, I believe that the implementation of the WCR11 is more or less correct.

```
import wildboottest.wildboottest as wb
import numpy as np
import timeit 
import time

N = 100000
k = 100
G= 50
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

start_time = timeit.default_timer()
wcr = wb.Wildboottest(X = X, Y = Y, cluster = cluster, bootcluster = bootcluster, R = R, B = B, seed = 12341)
wcr.get_scores(bootstrap_type = "11", impose_null = True)
wcr.get_weights(weights_type = "rademacher")
wcr.get_numer()
wcr.get_denom()
wcr.numer
wcr.denom
wcr.get_tboot()
wcr.t_boot
wcr.get_vcov()
wcr.get_tstat()
wcr.get_pvalue(pval_type = "two-tailed")
print("estimation time:", timeit.default_timer() - start_time)
# >>> 0.9225496 seconds
print("p value:", wcr.pvalue)
# >>> p value: 0.15258152581525816
```
