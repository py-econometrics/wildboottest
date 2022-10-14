import timeit 
import time

N = 100000
k = 100
G= 50
X = np.random.normal(0, 1, N * k).reshape((N,k))
beta = np.random.normal(0,1,k)
beta[0] = 0.005
u = np.random.normal(0,1,N)
y = 1 + X @ beta + u
cluster = np.random.choice(list(range(0,G)), N)
bootcluster = cluster
R = np.zeros(k)
R[0] = 1
B = 99999

start_time = timeit.default_timer()
wb = Wildboottest(X = X, Y = y, cluster = cluster, bootcluster = bootcluster, R = R, B = B, seed = 12341)
wb.get_scores(bootstrap_type = "11", impose_null = True)
wb.get_numer()
wb.get_denom()
wb.numer
wb.denom
wb.get_tboot()
wb.t_boot
wb.get_vcov()
wb.get_tstat()
wb.get_pvalue(pval_type = "two-tailed")
print("estimation time:", timeit.default_timer() - start_time)
# >>> 0.1981981981981982
print("p value:", wb.pvalue)
# >>> p value: 0.15258152581525816
