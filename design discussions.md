Hi all, I thought I'd share some discussions here regarding the design of the functionality we'd like to add we'd be interested in your thoughts, @josef-pkt @bashtage . 

- First, as implemented in `boottest`, `fwildclusterboot` and `WildBootTests.jl`, we currently do not provide a bootstrapped variance covariance matrix. 
- The reason for this is that computing the 'full' vcov matrix is time consuming. Second, it is not possible to impose the null on the data generating process, as a 'general' covariance matrix is not tied to a specific hypothesis. In general, the literature on the WCB recommends to impose the null on the bootstrap data generating process. 
- The algorithm we implement is fairly optimized for testing scalar hypotheses for single coefficients (e.g. standard t-tests). In consequence, to compute p-values for all k regressors, we loop over k individual hypotheses tests. P-values are computed via a percentile-t approach
- Because no bootstrapped variance covariance matrix is explicitly computed, we currently do not report bootstrapped standard errors. This is also in alignment with the implementations in `boottest`, `fwildclusterboot` and `WildBootTests.jl`
- Confidence intervals are not yet supported, but we will add this feature at some point. CIs will be inverted by inverting tests based on the percentile-t pvalue.d
- In principle, it would be possible to compute standard errors based on the bootstrapped regression coefficients and to report them
- In general, I am against doing so, mostly because I believe that it might be confusing to compute p-values and confidence intervals without using the reported standard errors in their computation
- Optionally, we could offer a 