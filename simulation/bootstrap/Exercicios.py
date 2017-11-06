# IMPORT BIBLIOTECAS NECESSÁRIAS
import Boot
import pandas as pd
import numpy as np
from scipy.stats import t

# EXERCÍCIO 1
path1 = "ex1.csv"
dados = pd.DataFrame.from_csv(path1)
B1 = [ 50, 200, 1000 ]
est1 = ["Media", "Variancia"]

# ITEM a)
Boot1 = Boot.varios_boots(dados, B1 , est1, "NaoPar")
Boot.todos_IC(Boot1,est1,"NaoPar")
Boot.faz_grafico(Boot1, est1,"NaoPar")

#ITEM b)
Boot2 = Boot.varios_boots(dados, B1, est1, "Par")
Boot.todos_IC(Boot2, est1, "Par")
Boot.faz_grafico(Boot2, est1,"Par")

#ITEM c)
Boot3 =Boot.varios_boots(dados, B1, est1, "Jack")
Boot.todos_IC(Boot3, est1, "Jack")
Boot.faz_grafico(Boot3, est1,"Jack")


# EXERCICIO 2
dados2 = pd.DataFrame.from_csv("ex2.csv")
B2 = [100,500,1500]
est2 = ["Correlacao"]

Boot4_par = Boot.varios_boots(dados2, B2 ,["Correlacao"],"Par")
Boot.todos_IC(Boot4_par,est2,"Cor_Par")
Boot.faz_grafico(Boot4_par, est2,"Cor_Par")

Boot4_Npar = Boot.varios_boots(dados2, B2 ,["Correlacao"],"NaoPar")
Boot.todos_IC(Boot4_Npar,est2,"Cor_NPar")
Boot.faz_grafico(Boot4_Npar, est2,"Cor_NPar")

Boot4_Jack = Boot.varios_boots(dados2, B2 ,["Correlacao"],"Jack")
Boot.todos_IC(Boot4_Jack,est2,"Cor_Jack")
Boot.faz_grafico(Boot4_Jack, est2,"Cor_Jack")


# EXERCICIO 3
dados3 = pd.DataFrame.from_csv("ex3.csv")
print(dados3)
n = dados3.shape[0]
B3 = [150,1500]
BootX3 = Boot.varios_boots(dados3,B3, ["Variancia"], "NaoPar")
Boot.todos_IC(BootX3,["Variancia"],"Exe3")

colunas = ["Media",           "Erro Padrao",
           "IC90_Inf_Normal", "IC90_Sup_Normal",
           "IC95_Inf_Normal", "IC95_Sup_Normal",
           "IC90_Sup_b-t",    "IC90_Inf_b-t",
           "IC95_Sup_b-t",    "IC95_Inf_b-t",
           "IC90_Inf_Perc",   "IC90_Sup_Perc",
           "IC95_Inf_Perc",   "IC95_Sup_Perc"  ]

#i) a=-5 e b=5
#n=150
a = -5 / BootX3["resultados"][colunas]
b =  5 / BootX3["resultados"][colunas]

p1 =  pd.DataFrame(t.cdf(b, df=n-1) - t.cdf(a, df=n-1), index=["N150","N1500"], columns=colunas )
print(p1)
p1.to_csv("Ex3_p1.csv")

#ii) a=-2 e b=2
a = -2 / BootX3["resultados"][colunas]
b =  2 / BootX3["resultados"][colunas]

p2 =  pd.DataFrame(t.cdf(b, df=n-1) - t.cdf(a, df=n-1), index=["N150","N1500"], columns=colunas )
print(p2)
p2.to_csv("Ex3_p2.csv")
