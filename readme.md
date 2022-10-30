
<!-- README.md is generated from README.Rmd. Please edit that file -->

## wildboottest

In this repo, we aim to develop a Python version of the wild cluster
bootstrap as implemented in Stata’s
[boottest](https://github.com/droodman/boottest), R’s
[fwildclusterboot](https://github.com/s3alfisc/fwildclusterboot) or
Julia’s [WildBootTests.jl](https://github.com/droodman/WildBootTests.jl)
as a post-estimation command for
[statsmodels](https://github.com/statsmodels/statsmodels) and
[linearmodels](https://github.com/bashtage/linearmodels).

If you’d like to cooperate, either send us an
[email](alexander-fischer1801@t-online.de) or comment in the issues
section!

## Example

Note: everything is still very much work in progress.

``` python
from wildboottest.wildboottest import wildboottest
import statsmodels.api as sm
import numpy as np
import pandas as pd

# prepare data
np.random.seed(12312312)
N = 1000
k = 10
G = 10
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
wildboottest(model, param = "X1", cluster = cluster, B = 9999)
#>   param              statistic   p-value
#> 0    X1  [-0.9161101602560844]  0.414062
wildboottest(model, cluster = cluster, B = 9999)
#>   param              statistic   p-value
#> 0    X1  [-0.9161101602560844]  0.414062
#> 1     1   [-23.47104637779698]  0.000000
#> 2     2   [14.490683161123965]  0.000000
#> 3     3  [-12.622808111755516]  0.000000
#> 4     4   [-19.98559455990591]  0.000000
#> 5     5   [1.4925442694763955]  0.183594
#> 6     6    [7.231015380060432]  0.000000
#> 7     7   [-41.37320307285552]  0.000000
#> 8     8    [17.31090484734575]  0.000000
#> 9     9   [-37.31996125067985]  0.000000
```
