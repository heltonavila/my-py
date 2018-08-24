import numpy as np
import pandas as pd

def calc_pct( x , k ):
    '''Função realiza o cálculo do percentile, retornando um array com o perceil que cada observação pertence.

    inputs:
        x: array com os valores para se calcular os percentis, dimensão n.
        k: valor (int) do passo dos percentis, isso é, 5 em 5 ou 10 em 10.

    return:
        array de dimensão n com os valores de cada percentile referente a cada observação '''

    q = np.arange(k,100.1,k)
    p = np.percentile(x, q)
    return q[np.searchsorted( p, x )]

def agregate_pct( df, by="percentile", target="target" ):
    '''Função para calcular a quantidade de evento em cada um dos percentis.

    inputs: 
        df: Dataframe com ao menos duas colunas: percentile e target.
        by: Nome da coluna que contem a variável a ser utilizada para realização a agregação (percentile)
        target: Nome da coluna que contem a variável target (que será aplicada a função soma na agregação)

    return: Novo Dataframe sumarizado pela agregação realizada.'''
    
    group = df.groupby(by=by)
    
    # Calculo da quantidade de eventos em um percentile
    df_new = group[target].sum()
    
    # Realiza a ordenação correta dos percentis.
    df_new = df_new.reset_index()
    df_new[by] = df_new[by].sort_values(ascending=False).get_values()
    df_new.sort_values(by=by, inplace=True)
    df_new.reset_index(drop=True, inplace=True)

    # Calculo da quantidade de eventos acumulada
    df_new[target+"_cum"] = df_new[target].cumsum()
    
    # Cálculo da quantidade de observações por percentile e acumulado.
    df_new["N"] = group[by].count().get_values()
    df_new["N_cum"] = df_new["N"].cumsum()

    return df_new

def compute_taxes(df, target="target"):
    '''Funcao que realiza os cálculo das taxas de acerto em cada percentile com base no dataframe dos percentis agregados.

    inputs:
        df: Dataframe dos percentís agregados, contendo a quantidade de evento capturado em cada percentile bem como o numero de observações.
        target: Nome da coluna que corresponde a variável resposta (target)

    return: Dataframe com as colunas adiconais referente às taxas obtidas, tanto por percentile quanto acumuladas'''

    df_new = df.copy()
    df_new["taxe"] = df[target] / df["N"]
    df_new["taxe_cum"] = df[target+"_cum"] / df["N_cum"]

    return df_new

def compute_lift(df, percentile="percentile"):
    '''Funcao para realizar o cálculo do lift e lift acumulado a partir de um dataframe que contem as taxas por percetil.

    inputs:
        df: Dataframe contendo as taxas (e acumualdas) calculada em cada percentile.
        percentile: Nome da coluna que contem os percentis.

    return: Dataframe atualizado com as colunas de lift e lift acumulado geradas.'''

    df_new = df.copy()
    df_new["lift"] = df["taxe"] / df[ df[percentile]==100 ]["taxe_cum"].get_values()
    df_new["lift_cum"] = df["taxe_cum"] / df[ df[percentile]==100 ]["taxe_cum"].get_values()
    return df_new

def compute_target_capture(df, percentile="percentile"):
    '''Função para realizar o cálculo de quanto da variável resposta foi capturada em cada percentile, bem como a acumulada.

    inputs:
        df: Dataframe contendo a quantidade da resposta em cada percentile e a acumulada disso.
        percentile: Nome da coluna que contem os percentis.

    return: Novo dataframe contendo as taxas da variável respostas que foram capturadas em cada percentile (acumulado e absoluto)'''

    df_new = df.copy()
    df_new["target_catch"] = df["target"] / df[ df["percentile"]==100 ]["target_cum"].get_values()
    df_new["target_catch_cum"] = df["target_cum"] / df[ df["percentile"]==100 ]["target_cum"].get_values()

    return df_new

def compute_FullTable(model_prob, target,  k ):
    '''Funcao que cria a tabela completa para verificar lift e resposta capturada

    inputs:
        model_prob: array com as probabilidade estimadas do modelo
        real_target: array com o verdadeiro rótulo da observação
        k: valor de como os percentis serão quebrados, 5 em 5, 10 em 10, etc.

    return: dataframe com todas as métricas geradas percentile por percentile'''

    # Cria dataframe para realizar a agregação
    df = pd.DataFrame( np.matrix( [model_prob, target] ).T, columns=["model_prob", "target"])
    df["percentile"] = calc_pct( model_prob, k)

    # Realiza todos os procecimentos crian as colunas necessárias para analise.
    df_pct = agregate_pct( df )
    df_pct = compute_taxes( df_pct )
    df_pct = compute_lift( df_pct )
    df_pct = compute_target_capture( df_pct )

    return df_pct

def calc_sum_cph(cph, k):
    '''Funcao para calcular o coeficiente "sum_cph", que será utilizado para obter L-quality.

    inputs:
        cph: array com as taxas da respostas capturadas acumulada nos percentis.
        k: passo dos percentis, isto é, 5 em 5, 10 em 10, etc.

    return: Valor do sumCPH obtido'''

    return np.sum(cph - 0.5) * (k/100)

def calc_b(df, percentile="percentile"):
    ''' Funcao para calcular obter a proporção real da variável resposta na base.
    
    inputs:
        df: Dataframe dos percentis calculados
        percentile:  Nome da coluna que contem os percentis.

    return: Valor da real proporção de evento na base'''

    return df[ df[percentile]==100 ]["taxe_cum"].get_values()[0]

def calc_lQuality( sum_cph, b):
    '''Funcao para calcular o valor L-Quality linear

    inputs:
        sum_cph: Coeficiente gerado a partir da taxa de resposta capturada acumulada.
        b: Real proporção de evento na base

    return: valor do L-Quality linear'''

    return (sum_cph - 0.5) / ( 1 - (b/0.5) - 0.5)