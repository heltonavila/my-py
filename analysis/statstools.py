import pandas as pd
import numpy as np
import scipy as sp

def boxplot_bounds(x):
    q1, q2 = np.percentil(x, [25,75], axis=0)
    bound_min = q1 - 1.5 * (q3 - q1)
    bound_max = q3 + 1.5 * (q3 - q1)
    return bound_min, bound_max

def std_bound(x, confiance=0.95):
    mean = np.mean(x)
    std = np.std(x)
    coef = sp.stats.norm.ppf( 1 - ( ( 1-confiance ) / 2 ) )
    bound_min = mean - coef * std
    bound_max = mean + coef * std
    return bound_min, bound_max

def pct_bounds(x, confiance=0.95):
    ite = 100 * (1 - confiance) / 2
    bound_min, bound_max = np.percentile( x, [ite, 100-ite] )
    return bound_min, bound_max

def is_outlier( x, method="boxplot", confiance=0.95 ):
    if method == "boxplot":
        bound_min, bound_max = boxplot_bounds(x)
    elif method == "std":
        bound_min, bound_max = std_bounds(x, confiance)
    elif method == "pct":
        bound_min, bound_max = pct_bounds(x, confiance)
    y = x.apply( lambda x: not( bound_min <= x <= bound_max) )
    return y



