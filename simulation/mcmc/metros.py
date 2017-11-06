import numpy as np
from scipy.stats import norm
from scipy.stats import expon
from scipy.stats import poisson
import numpy.random as random
import matplotlib.pyplot as plt
from statsmodels.graphics import tsaplots


def metro_Normal(chute=[1], N=1000, sigma_aux=1):
    valores = chute
    taxa = []
    priori = norm(0,1)

    for i in range(N):
        normal_aux_ant = norm(valores[-1], sigma_aux)
        valor = normal_aux_ant.rvs()
        normal_aux_valor = norm(valor, sigma_aux)
        U = random.rand()

        teste = ( ( priori.pdf(valor) * normal_aux_valor.pdf(valores[-1]) ) /
                  ( priori.pdf(valores[-1]) * normal_aux_ant.pdf(valor) ) )

        if min([1,teste]) > U:
            valores.append(valor)
            taxa.append(1)
        else:
            valores.append(valores[-1])
            taxa.append(0)

    return {"valores":valores , "taxa":sum(taxa)/len(taxa)}

def metro_exp_poison(chute=[1], N=1000):
    valores = chute
    taxa = []

    priori = expon(1)

    for i in range(N):
        expo_aux = expon(1)
        valor = expo_aux.rvs()

        U = random.rand()

        x_dado_y = poisson(valores[-1])
        y_dado_x = poisson(valor)

        teste = ( priori.pdf(valor) * x_dado_y.pmf(int(valores[-1])) ) / ( priori.pdf(valores[-1]) * y_dado_x.pmf(int(valor)) )

        if min([teste,1]) > U:
            valores.append(valor)
            taxa.append(1)
        else:
            valores.append(valores[-1])
            taxa.append(0)

    return {"valores":valores , "taxa":sum(taxa)/len(taxa)}

def faz_plot(valores):
    fig, axes = plt.subplots(nrows=3,figsize=(8,12))
    axes[0].hist(valores)
    axes[1].plot(valores)
    tsaplots.plot_acf(valores, axes[2])
    for ax in axes.flat:
        ax.set(title='', xlabel='')
    plt.show()
    return None
