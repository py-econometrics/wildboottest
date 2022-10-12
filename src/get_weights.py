import numpy as np
from itertools import permutations

class WildDrawFunctionException(Exception):
    pass

def rademacher(n: int) -> np.ndarray:
    rng = np.random.default_rng()
    return rng.choice([-1,1],size=n, replace=True)

def mammen(n: int) -> np.ndarray:
    rng = np.random.default_rng()
    return rng.choice(
        a= np.array([-1, 1]) * (np.sqrt(5) + np.array([-1, 1])) / 2, #TODO: #10 Should this divide the whole expression by 2 or just the second part
        size=n,
        replace=True,
        p = (np.sqrt(5) + np.array([1, -1])) / (2 * np.sqrt(5))
    )
    
def norm(n):
    rng = np.random.default_rng()
    return rng.normal(size=n)

def webb(n):
    rng = np.random.default_rng()
    return rng.choice(
        a = [-np.sqrt(np.array([3,2,1]) / 2), np.sqrt(np.array([1,2,3]) / 2)],
        replace=True,
        size=n
    )
    
wild_draw_fun_dict = {
    'rademacher' : rademacher,
    'mammen' : mammen,
    'norm' : norm,
    'webb' : webb
}

  
def get_weights(t : str, full_enumeration: bool, N_G_bootcluster: int, boot_iter: int) -> np.ndarray:
    """draw bootstrap weights

    Args:
        t (str): the type of the weights distribution. Either 'rademacher', 'mammen', 'norm' or 'webb'
        full_enumeration (bool): should deterministic full enumeration be employed
        N_G_bootcluster (int): the number of bootstrap clusters
        boot_iter (int): the number of bootstrap iterations

    Returns:
        np.ndarray: a matrix of dimension N_G_bootcluster x (boot_iter + 1)
    """    
    
    #TODO: we can use the `case` feature in python, but that's only available in 3.10+ will do a 3.7 version for now
    # Will take out this and make separate functions for readability
    
    wild_draw_fun = wild_draw_fun_dict.get(t)
    
    if wild_draw_fun is None:
        raise WildDrawFunctionException("Function type specified is not supported or there is a typo.")
  
    # do full enumeration for rademacher weights if bootstrap iterations
    # B exceed number of possible permutations else random sampling

    # full_enumeration only for rademacher weights (set earlier)
    if full_enumeration: 
        t = 0
        #TODO: #12 Is this just a permutation function?
        # gtools_permutations(
        v0 = permutations( 
            [-1,1],
            r = N_G_bootcluster,
        )
        v = np.insert(v0, 0, 1)
    else:
        # else: just draw with replacement - by chance, some permutations
        # might occur more than once
        v = wild_draw_fun(n = N_G_bootcluster * (boot_iter + 1))
        #dim(v) <- c(N_G_bootcluster, boot_iter +1) don't think we need this
        v = np.insert(v, 0, 1)
        # v[, 1] <- 1
    
    return v
  
  
