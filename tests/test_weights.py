import pytest
from itertools import product
from wildboottest.weights import WildDrawFunctionException, draw_weights, wild_draw_fun_dict

ts = list(wild_draw_fun_dict.keys())
full_enum = [True, False]
ng_bootclusters = list(range(0,100, 10))
boot_iter = list(range(0,1000,400))

def test_none_wild_draw_fun():
    with pytest.raises(WildDrawFunctionException):
        draw_weights(None, True, 1,1)
        
