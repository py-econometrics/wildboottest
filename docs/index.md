# Welcome to Wildboottest

![PyPI](https://img.shields.io/pypi/v/wildboottest?label=pypi%20package)
![PyPI - Downloads](https://img.shields.io/pypi/dm/wildboottest)

`wildboottest` implements multiple fast wild cluster
bootstrap algorithms as developed in [Roodman et al
(2019)](https://econpapers.repec.org/paper/qedwpaper/1406.htm) and
[MacKinnon, Nielsen & Webb
(2022)](https://www.econ.queensu.ca/sites/econ.queensu.ca/files/wpaper/qed_wp_1485.pdf).

## Functionality

It has similar, but more limited functionality than Stata's [boottest](https://github.com/droodman/boottest), R's [fwildcusterboot](https://github.com/s3alfisc/fwildclusterboot) or Julia's [WildBootTests.jl](https://github.com/droodman/WildBootTests.jl). It supports

-   The wild cluster bootstrap for OLS ([Cameron, Gelbach & Miller 2008](https://direct.mit.edu/rest/article-abstract/90/3/414/57731/Bootstrap-Based-Improvements-for-Inference-with),
    [Roodman et al (2019)](https://econpapers.repec.org/paper/qedwpaper/1406.htm)).
-   Multiple new versions of the wild cluster bootstrap as described in
    [MacKinnon, Nielsen & Webb (2022)](https://www.econ.queensu.ca/sites/econ.queensu.ca/files/wpaper/qed_wp_1485.pdf), including the WCR13, WCR31, WCR33,
    WCU13, WCU31 and WCU33.
-   CRV1 and CRV3 robust variance estimation, including the CRV3-Jackknife as 
    described in [MacKinnon, Nielsen & Webb (2022)](https://arxiv.org/pdf/2205.03288.pdf).
    
At the moment, `wildboottest` only computes wild cluster bootstrapped *p-values*, and no confidence intervals. 

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

You can install `wildboottest` from [PyPi](https://pypi.org/project/wildboottest/) by running 

```
pip install wildboottest
```

## Citation 

If you use wildboottest in your research, please consider citing it via

```
@Unpublished{wildboottest2022,
  Title  = {Fast Wild Cluster Bootstrap Inference in Python via wildboottest},
  Author = {Alexander Fischer and Aleksandr Michuda},
  Year   = {2022},
  Url    = {https://github.com/s3alfisc/wildboottest}
}
```

