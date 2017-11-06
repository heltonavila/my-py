from numpy.random import randint
from numpy.random import normal as rnorm
from numpy.random import multivariate_normal as mvrnorm
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import pandas as pd

### CLASSE DE BOOTSTRAP PARAMETRICO
class boot():
    def __init__(self, x, B, medidas, metodo = "NaoPar"):
        self.dados = x
        self.B = B
        self.medidas = medidas
        self.metodo = metodo
        self.n = self.dados.shape[0]
        self.m = self.dados.shape[1]
        self.est_dados = self.tira_estats(self.dados, self.medidas).T
        self.est_dados.columns = [ "Estatistica" ]
        self.amostras = pd.DataFrame(index=self.dados.index)
        self.roda_tudo()

    ### FUNCAO QUE GERA UMA AMOSTRA ALETATORIA PARAMETRICA
    def gera_parametrico(self):
        if self.m is 1:
            amostra = rnorm(self.est_dados["Estatistica"]["Media"],sqrt(self.est_dados["Estatistica"]["Variancia"]), self.n )
        else:
            amostra = mvrnorm(self.dados.mean(), self.dados.cov(), size=self.n  )
            amostra = pd.DataFrame(amostra)
        return amostra

    ### FUNCAO QUE GERA UMA AMOSTRA ALETATORIA JACKKNIFE
    def gera_jackknife(self):
        nomes = self.dados.columns
        if self.m is 1:
            for i in self.dados.index:
                self.amostras["Amostra" + str(i + 1)] = self.dados[nomes[0]][self.dados.index!=i]
        else:
            for i in self.dados.index:
                self.amostras["X" + str(i + 1)] = self.dados[nomes[0]][self.dados.index!=i]
                self.amostras["Y" + str(i + 1)] = self.dados[nomes[1]][self.dados.index!=i]
        return None

    ###FUNCAO QUE GERA UMA UNICA AMOSTRA ALEATÓRIA NAO PARAMETRICA
    def gera_nao_parametrico(self):
        valores = randint(self.n, size=self.n)
        amostra = self.dados.loc[ valores ].copy()
        amostra.index = range(amostra.shape[0])
        return amostra

    ### FUNCAO QUE GERA B AMOSTRAS DE TAMANHO n
    def todas_amostras(self):
        if self.metodo=="Par":
            if self.m == 1:
                for i in range(self.B):
                    self.amostras[ "Amostra" + str(i + 1) ] = self.gera_parametrico()
            else:
                for i in range(self.B):
                    amostra = self.gera_parametrico()
                    self.amostras[ "X" + str(i + 1) ] = amostra[amostra.columns[0]]
                    self.amostras[ "Y" + str(i + 1) ] = amostra[amostra.columns[1]]

        elif self.metodo=="NaoPar":
            if self.m == 1:
                for i in range(self.B):
                    self.amostras[ "Amostra" + str(i + 1) ] = self.gera_nao_parametrico()
            else:
                for i in range(self.B):
                    amostra = self.gera_nao_parametrico()
                    self.amostras[ "X" + str(i + 1) ] = amostra[amostra.columns[0]]
                    self.amostras[ "Y" + str(i + 1) ] = amostra[amostra.columns[1]]

        elif self.metodo=="Jack":
            self.gera_jackknife()
        return None

    ### FUNCAO QUE REALIZA AS ESTATÍSTICAS REQUISITADAS
    def tira_estats(self, x, medidas):
        res = pd.DataFrame()
        if "Media" in medidas:
            res[ "Media" ] = x.mean()
        if "Variancia" in medidas:
            res[ "Variancia" ] = x.var()
        if "Correlacao" in medidas:
            corr = []
            if x.shape[1] == 2:
                corr.append(x.corr()[x.columns[0]][x.columns[1]])
                res[ "Variancia" ] = x.cov()[x.columns[1]][x.columns[0]]
            else:
                col = self.dados.columns
                vari = []
                for j in range(int(self.amostras.shape[1]/2)):
                    colunas = [col[0]+str(j+1) , col[1]+str(j+1)]
                    corr.append(x[colunas].corr()[colunas[0]][colunas[1]])
                    vari.append(x[colunas].cov()[colunas[0]][colunas[1]])
                res["Variancia"] = vari
            res["Correlacao"] = corr
        return res

    ### FUNCAO QUE REALIZA AS ESTIMAÇÕES BOOTSTRAP
    def estimacao(self, estat_dado):
        self.resultados = pd.DataFrame()
        self.resultados[ "Media" ] = self.estats.mean()
        if self.metodo == "Jack":
            self.resultados[ "Erro Padrao" ] = self.estats[self.medidas].std()/sqrt(self.n)
        else:
            self.resultados[ "Erro Padrao" ] = self.estats[self.medidas].std()
        self.resultados[ "Vicio" ] = self.resultados[ "Media" ] - estat_dado
        self.resultados = self.resultados.loc[self.medidas]
        return None

    ### RODA TODAS FUNÇÕES NECESSÁRIAS
    def roda_tudo(self):
        self.todas_amostras()
        self.estats = self.tira_estats(self.amostras, self.medidas)
        self.estimacao(self.est_dados[ "Estatistica" ])
        return None

def varios_boots(x, B, estats, par):
    lista_boot = [ boot(x, b,estats, par) for b in B]
    res = pd.DataFrame()
    for i in lista_boot:
        res = res.append(i.resultados)
    bootstrap = { "boots":lista_boot , "resultados": res}
    return bootstrap

def intervalo_normal(Boot):
    Boot["resultados"]["IC90_Inf_Normal"] = Boot["resultados"]["Media"]-1.64*Boot["resultados"]["Erro Padrao"]
    Boot["resultados"]["IC90_Sup_Normal"] = Boot["resultados"]["Media"]+1.64*Boot["resultados"]["Erro Padrao"]
    Boot["resultados"]["IC95_Inf_Normal"] = Boot["resultados"]["Media"]-1.96*Boot["resultados"]["Erro Padrao"]
    Boot["resultados"]["IC95_Sup_Normal"] = Boot["resultados"]["Media"]+1.96*Boot["resultados"]["Erro Padrao"]
    return None

def intervalo_t(Boot):
    Boot["resultados"]["IC90_Inf_t"] = Boot["resultados"]["Media"]-1.83*Boot["resultados"]["Erro Padrao"]
    Boot["resultados"]["IC90_Sup_t"] = Boot["resultados"]["Media"]+1.83*Boot["resultados"]["Erro Padrao"]
    Boot["resultados"]["IC95_Inf_t"] = Boot["resultados"]["Media"]-2.26*Boot["resultados"]["Erro Padrao"]
    Boot["resultados"]["IC95_Sup_t"] = Boot["resultados"]["Media"]+2.26*Boot["resultados"]["Erro Padrao"]

def intervalo_boot_t(Boot, medidas):
    quartil90_Inf = []
    quartil90_Sup = []
    quartil95_Inf = []
    quartil95_Sup = []
    for b in Boot["boots"]:
        for m in medidas:
            valor = (b.estats[m] - b.resultados["Media"][m]).divide( b.estats["Variancia"].apply(np.sqrt) )
            valor.sort()
            valor.index=range(valor.shape[0])
            quartil90_Inf.append( valor.quantile(0.05) )
            quartil90_Sup.append( valor.quantile(0.95) )
            quartil95_Inf.append( valor.quantile(0.025) )
            quartil95_Sup.append( valor.quantile(0.975) )
    Boot["resultados"]["IC90_Sup_b-t"] = Boot["resultados"]["Media"]-quartil90_Inf*Boot["resultados"]["Erro Padrao"]
    Boot["resultados"]["IC90_Inf_b-t"] = Boot["resultados"]["Media"]-quartil90_Sup*Boot["resultados"]["Erro Padrao"]
    Boot["resultados"]["IC95_Sup_b-t"] = Boot["resultados"]["Media"]-quartil95_Inf*Boot["resultados"]["Erro Padrao"]
    Boot["resultados"]["IC95_Inf_b-t"] = Boot["resultados"]["Media"]-quartil95_Sup*Boot["resultados"]["Erro Padrao"]
    return None

def intervalo_perc(Boot, medidas):
    quartil90_Inf = []
    quartil90_Sup = []
    quartil95_Inf = []
    quartil95_Sup = []
    for b in Boot["boots"]:
        for m in medidas:
            valor = b.estats[m].copy()
            valor.sort()
            valor.index=range(valor.shape[0])
            quartil90_Inf.append(valor.quantile(0.05))
            quartil90_Sup.append(valor.quantile(0.95))
            quartil95_Inf.append(valor.quantile(0.025))
            quartil95_Sup.append(valor.quantile(0.975))
    Boot["resultados"]["IC90_Inf_Perc"] = quartil90_Inf
    Boot["resultados"]["IC90_Sup_Perc"] = quartil90_Sup
    Boot["resultados"]["IC95_Inf_Perc"] = quartil95_Inf
    Boot["resultados"]["IC95_Sup_Perc"] = quartil95_Sup
    return None

def todos_IC(Boot, medidas, nome):
    intervalo_normal(Boot)
    intervalo_t(Boot)
    intervalo_boot_t(Boot, medidas)
    intervalo_perc(Boot,medidas)
    Boot["resultados"].to_csv("IC_"+nome+".csv")

def faz_grafico(Boot, medidas, nome):
    valores = {i:[] for i in medidas}
    B=[]
    for b in Boot["boots"]:
        B.append(b.B)
        for m in medidas:
           valores[m].append(b.estats[m])
    texto = medidas
    fig, axes = plt.subplots( nrows=1 , ncols=len(medidas), figsize=(12,5) )
    if len(medidas) > 1:
        axes = axes.ravel()
        for ax in range(len(axes)):
            axes[ax].boxplot(valores[medidas[ax]])
            axes[ax].set_xlabel("Número de Amostras")
            axes[ax].set_title(texto[ax])
        plt.setp(axes, xticklabels=B)
    else:
        axes.boxplot(valores[medidas[0]])
        axes.set_xlabel("Número de Amostras")
        axes.set_title(texto)
        plt.setp(axes, xticklabels=B)
    plt.savefig("boxplot"+"".join(medidas) + nome+ ".png", format="png", dpi=600)
