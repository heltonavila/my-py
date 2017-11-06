from math import *
import numpy as np

def media_recursiva(μ, n, valor):
    return μ + ( valor - μ ) / (n+1)

def var_recursiva(σ2, μ_n, μ ,n, valor ):
    return σ2 + μ_n**2 - μ**2 + (valor**2 - σ2 - μ_n**2) / (n+1)

def media_var_rec( μ, σ2, n, valor ):
    μ_n1 = media_recursiva(μ, n, valor)
    σ2_n1 = var_recursiva(σ2, μ, μ_n1, n, valor )
    return (μ_n1, σ2_n1)

def calc_n(X):
    return np.apply_along_axis( np.count_nonzero, 0, ~np.isnan(X) )
