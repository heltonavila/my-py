import pandas as pd
from scipy.stats import chi2_contingency as qui2_test
from scipy.stats import fisher_exact as fisher_test
import itertools

def tabela(x):
    tabela = x.value_counts()
    total = tabela.sum()
    tabela.name = "Frequencia"
    tabela = pd.DataFrame( tabela )
    tabela["Porcentagem"] = tabela["Frequencia"]  / total
    index = list( zip(*[ [x.name] * tabela.shape[0] , list(tabela.index) ]) )
    index = pd.MultiIndex.from_tuples( index, names=["Variável", "Categoria"] )
    tabela = pd.DataFrame( tabela.get_values() , columns = tabela.columns, index = index)
    return tabela.sortlevel()

def todas_tabelas(X):
    minha_tabela = pd.DataFrame()
    for x in list(X.columns):
        minha_tabela = pd.concat( [minha_tabela , tabela( X[x] )], axis=0 )
    return minha_tabela

def tabela_cruzada(  x, y , porc=False, teste="qui-squad fisher"):
    tabela = pd.crosstab( x, y )
    index = list(zip(*[ [tabela.index.name]*len(tabela.index), list(tabela.index) ] ) )
    index = pd.MultiIndex.from_tuples( index, names=["Variável Auxiliar","Categoria"] )
    if porc:
        colunas = list(itertools.product( [tabela.columns.name], list(tabela.columns), ["n", "%"] ) )
        colunas = pd.MultiIndex.from_tuples( colunas, names=["Variável Resposta","Categoria", "Unidade"] )
        nova_tabela= pd.DataFrame(columns = colunas, index= index)
        for c in list(tabela.columns):
            nova_tabela[ tabela.columns.name , c , "n"  ] = tabela[c].get_values()
            nova_tabela[ tabela.columns.name , c , "%"  ] = tabela[c].get_values() / tabela[c].sum()
    else:
        colunas = list(itertools.product( [tabela.columns.name], list(tabela.columns) ) )
        colunas = pd.MultiIndex.from_tuples( colunas, names=["Variável Resposta","Categoria"] )
        nova_tabela = pd.DataFrame( tabela.get_values(),columns = colunas, index = index)
    if "qui-squad" in teste:
        nova_tabela["p-valor Qui-Quadrado"] = round( qui2_test( tabela )[1] , 3)
    if "fisher" in teste and tabela.shape == (2,2):
        nova_tabela["p-valor Fishe Exato"] = round( fisher_test( tabela )[1] , 3)
    return nova_tabela

def todas_tabelas_cruzadas( X, Y, porc = True, teste="qui-squad fisher"):
    tabela = pd.DataFrame()
    for x in X:
        tabela = tabela.append( tabela_cruzada(X[x], Y, porc, teste) )
    return tabela
