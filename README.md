## wildboottest

`wildboottest` implements fast routines for wild cluster 
bootstrap inference in Python. 

Support for [statsmodels](https://github.com/statsmodels/statsmodels) and 
[linearmodels](https://github.com/bashtage/linearmodels) is work in progress.

If you'd like to cooperate, either send us an 
[email](alexander-fischer1801@t-online.de) or comment in the issues section!


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
