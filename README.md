## wildboottest

![PyPI](https://img.shields.io/pypi/v/wildboottest?label=pypi%20package)
![PyPI - Downloads](https://img.shields.io/pypi/dm/wildboottest)

`wildboottest` implements multiple fast wild cluster
bootstrap algorithms as developed in [Roodman et al
(2019)](https://econpapers.repec.org/paper/qedwpaper/1406.htm) and
[MacKinnon, Nielsen & Webb
(2022)](https://www.econ.queensu.ca/sites/econ.queensu.ca/files/wpaper/qed_wp_1485.pdf).

It has similar, but more limited functionality than Stata's [boottest](https://github.com/droodman/boottest), R's [fwildcusterboot](https://github.com/s3alfisc/fwildclusterboot) or Julia's [WildBootTests.jl](https://github.com/droodman/WildBootTests.jl). It supports

-   The wild cluster bootstrap for OLS ([Cameron, Gelbach & Miller 2008](https://direct.mit.edu/rest/article-abstract/90/3/414/57731/Bootstrap-Based-Improvements-for-Inference-with),
    [Roodman et al (2019)](https://econpapers.repec.org/paper/qedwpaper/1406.htm)).
-   Multiple new versions of the wild cluster bootstrap as described in
    [MacKinnon, Nielsen & Webb (2022)](https://www.econ.queensu.ca/sites/econ.queensu.ca/files/wpaper/qed_wp_1485.pdf), including the WCR13, WCR31, WCR33,
    WCU13, WCU31 and WCU33.
-   CRV1 and CRV3 robust variance estimation, including the CRV3-Jackknife as 
    described in [MacKinnon, Nielsen & Webb (2022)](https://arxiv.org/pdf/2205.03288.pdf).
- The (non-clustered) wild bootstrap for OLS ([Wu, 1986](https://projecteuclid.org/journals/annals-of-statistics/volume-14/issue-4/Jackknife-Bootstrap-and-Other-Resampling-Methods-in-Regression-Analysis/10.1214/aos/1176350142.full)).

    
At the moment, `wildboottest` only computes wild cluster bootstrapped *p-values*, and no confidence intervals. 

Other features that are currently not supported: 

-   The subcluster bootstrap ([MacKinnon and Webb 2018](https://academic.oup.com/ectj/article-abstract/21/2/114/5078969?login=false)).
-   Confidence intervals formed by inverting the test and iteratively
    searching for bounds.
-   Multiway clustering.


Direct support for [statsmodels](https://github.com/statsmodels/statsmodels) and 
[linearmodels](https://github.com/bashtage/linearmodels) is work in progress.

If you'd like to cooperate, either send us an 
[email](alexander-fischer1801@t-online.de) or comment in the issues section!

## Installation 

You can install `wildboottest` from [PyPi](https://pypi.org/project/wildboottest/) by running 

```
pip install wildboottest
```

## Example 

```python
import pandas as pd
import statsmodels.formula.api as sm
from wildboottest.wildboottest import wildboottest

df = pd.read_csv("https://raw.github.com/vincentarelbundock/Rdatasets/master/csv/sandwich/PetersenCL.csv")
model = sm.ols(formula='y ~ x', data=df)

wildboottest(model, param = "x", cluster = df.firm, B = 9999, bootstrap_type = '11')
# | param   |   statistic |   p-value |
# |:--------|------------:|----------:|
# | x       |      20.453 |     0.000 |

wildboottest(model, param = "x", cluster = df.firm, B = 9999, bootstrap_type = '31')
# | param   |   statistic |   p-value |
# |:--------|------------:|----------:|
# | x       |      30.993 |     0.000 |

# bootstrap inference for all coefficients
wildboottest(model, cluster = df.firm, B = 9999, bootstrap_type = '31')
# | param     |   statistic |   p-value |
# |:----------|------------:|----------:|
# | Intercept |       0.443 |     0.655 |
# | x         |      20.453 |     0.000 |

# non-clustered wild bootstrap inference
wildboottest(model, B = 9999, bootstrap_type = '11')
# | param     |   statistic |   p-value |
# |:----------|------------:|----------:|
# | Intercept |       1.047 |     0.295 |
# | x         |      36.448 |     0.000 |

```
