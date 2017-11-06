from models.functions import *
import numpy as np
import math
from math import pi
from scipy.stats import norm

class webayes:

    def __init__(self, nome=None):
        self.nome = nome

    ### DISTRIBUIÇÃO DE PROBABILIDADE BERNOULLI ###
    def fdpBernoulli(self, x ,theta):
        return ( theta ** x ) * ( (1-theta) ** (1 - x ) )

    ### DISTRIBUIÇÃO DE PROBABILDIADE NORMAL ( GAUSSIANA )
    def fdpNormal(self, x, theta):
        return norm(theta[0], theta[1]).pdf(x)

    ### ESTIMA OS PARAMETROS
    def estimar(self, X, Y, *stats ):
        return np.array([ np.array([stat(X[Y == c], axis=0) for stat in stats]).T for c in self.classes])

    ### ESTIMA DOS PARAMETROS DAS FEATURES
    def fit(self, XB=None, X=None, Y=None, priori=None):
        XN=X
        self.classes = np.unique(Y)
        self.n = [np.trim_zeros(np.bincount(Y))]
        if priori != None and len(priori)==len(self.classes):
            self.priori = priori
        else:
            self.priori = self.n[0] / np.sum(self.n[0])
        if XB is not None and XN is not None:
            self.theta = [ self.estimar( XB, Y, np.nanmean ), self.estimar(XN, Y, np.nanmean, np.nanstd) ]
            self.n +=[ np.array( [calc_n(XB[Y == i]) for i in self.classes] ) , np.array( [calc_n(XN[Y == i]) for i in self.classes] ) ]
        elif XB is not None:
            self.theta = [self.estimar( XB, Y, np.nanmean), None]
            self.n += [ np.array( [calc_n(XB[Y == i]) for i in self.classes] ), None]
        else:
            self.theta = [None, self.estimar( XN, Y, np.nanmean, np.nanstd)]
            self.n += [None, np.array( [calc_n(XN[Y == i]) for i in self.classes]) ]
        return None

    def update_fit_one(self, xB=None, xN=None, y=None):
        classe = list(self.classes).index(y)
        self.n[0][classe]+=1
        self.priori = self.n[0] / np.sum(self.n[0])
        if xB is not None:
            for i in range(len(xB)):
                self.theta[0][classe][i] = media_recursiva( self.theta[0][classe][i], self.n[1][classe][i], xB[i] )
                self.n[1][classe][i] += 1
        if xN is not None:
            for i in range(len(xN)):
                μ, σ2 = media_var_rec(self.theta[1][classe][i][0], self.theta[1][classe][i][1] ** 2, self.n[2][classe][i], xN[i])
                self.theta[1][classe][i] = [μ, math.sqrt(σ2)]
                self.n[2][classe][i] += 1

    ### LOG-VEROSSIMILHANÇA BERNOULLI
    def LogVeroBernoulli(self, x, theta):
        return np.nansum( np.log( np.apply_along_axis( self.fdpBernoulli, 0, x, theta ) ) )

    ### LOG VEROSSIMILHANÇA NORMAL
    def LogVeroNormal(self, x, theta):
        return np.nansum( np.log( np.apply_along_axis( self.fdpNormal, 0, x, theta.T ) ) )

    ### LOG-VEROSSIMILHANÇA BERNOULLI
    def LogVerossimilhanca(self, xB, xN, theta_b , theta_n):
        return self.LogVeroBernoulli(xB, theta_b) + self.LogVeroNormal(xN,  theta_n)

    def pred_log_prob_Bernoulli(self, xB):
        prob = np.array( [ np.log(self.priori[c]) + np.nansum(self.LogVeroBernoulli(xB, theta=self.theta[0][c])) for c in range(len(self.classes))] )
        return prob

    def pred_log_prob_Normal(self, xN):
        prob = np.array( [ np.log(self.priori[c]) + np.nansum(self.LogVeroNormal(xN, theta=self.theta[1][c] )) for c in range(len(self.classes))] )
        return prob

    def predicao_log_prob(self, xB, xN ):
        prob = np.array( [ np.log(self.priori[c]) + np.nansum( self.LogVerossimilhanca(xB, xN, theta_b=self.theta[0][c], theta_n=self.theta[1][c]) ) for c in range(len(self.classes)) ] )
        return  prob

    def predict_log_proba(self, XB=None, XN=None ):
        if XB is not None and XN is not None:
            return np.array([self.predicao_log_prob(XB[i], XN[i]) for i in range(XB.shape[0])])
        elif XB is not None:
            return np.apply_along_axis(self.pred_log_prob_Bernoulli, 1,XB )
        else:
            return np.apply_along_axis(self.pred_log_prob_Normal, 1, XN)

    def predict(self, XB=None, X=None):
            return self.classes[ np.argmax( self.predict_log_proba(XB, X), axis=1) ]
