'''
  PONTIFÍCIA UNIVERSIDADE CATÓLICA DE MINAS GERAIS
  NÚCLEO DE EDUCAÇÃO A DISTÂNCIA
  Pós-graduação Lato Sensu em Ciência de Dados e Big Data
  
  Título:  Segmentação de Vendedores do marketplace Olist Store em 2017
  
  Aluno: Alexandre Luis Nunes Cardiga

  Nome: minhas_funcoes.py
  
  Data: 22/10/2021
  
  Objetivo: Funções criadas  para atender este projeto

'''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
from scipy import stats
from scipy.stats    import normaltest, kstest, norm
from sklearn.feature_selection import VarianceThreshold
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
from math import isnan
from sklearn.cluster import KMeans, DBSCAN
from sklearn import metrics
from sklearn.metrics import silhouette_samples,silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans, DBSCAN
from random import randrange, uniform
from sklearn.cluster import MeanShift, estimate_bandwidth
from datetime import datetime as dt

#import datetime
#from datetime import time

#eliminando os espacos entre as tags
def Trata_HTML(input):
    return " ".join(input.split()).replace('> <','><')

#Separa cfe tamanho definido no prametro
def Separador(v,tamanho):
    return v.strip().zfill(tamanho)[0:tamanho]

def e_par(numero):
    resto=numero % 2
    if resto == 0:
        return 1
    else:
        return 0
    
#Algumas informações do dataFrame
def Resumo_DataFrame(df):
    df_sumario = pd.DataFrame({
    'Colunas': df.columns,
    'Nulos': [df[i].isnull().sum() for i in df.columns],
    'Total de unicos': [df[i].nunique() for i in df.columns],
    'Numero Registros': [df[i].shape[0] for i in df.columns],
    'Tipo': [df[i].dtypes for i in df.columns]})
    df_sumario['Total de nulos'] = round(df_sumario['Nulos'] / df_sumario['Numero Registros'] * 100, 2)
    df_sumario= df_sumario[['Colunas', 'Nulos', 'Total de nulos', 'Total de unicos', 'Numero Registros', 'Tipo']]
    
    return df_sumario

def trocar_caracteres(conteudo):
    nome = conteudo.replace('Á','A').replace('Â','A').replace('Ã','A').replace('À','A').replace('É','E').replace('Ê','E').replace('Í','I').\
    replace('Ó','O').replace('Ô','O').replace('Õ','O').replace('Ú','U').replace('Ü','U').replace('Ç', 'C')
    return nome

def is_float(element):
    try:
        float(element)
        return True
    except ValueError:
        return False

def atualiza_municipios_vendedores(df,feature):
    df[feature] = df[feature].replace('riberao preto','ribeirao preto')\
    .replace("santa barbara d'oeste",'santa barbara doeste')\
    .replace("santa barbara d´oeste", 'santa barbara doeste')\
    .replace('mogi guacu','mogi-guacu')\
    .replace("arraial dajuda",'porto seguro')\
    .replace('sao  paulo','sao paulo')\
    .replace('sao paulo - sp', 'sao paulo')\
    .replace('pirituba','sao paulo')\
    .replace('paincandu','paicandu') \
    .replace('cascavael','cascavel')\
    .replace("arraial d'ajuda (porto seguro)",'arraial dajuda') \
    .replace("vendas@creditparts.com.br", 'curitiba')\
    .replace("scao jose do rio pardo", "sao jose do rio pardo")\
    .replace("s jose do rio preto", "sao jose do rio preto")\
    .replace('balenario camboriu','balneario camboriu')\
    .replace('distrito federal', 'brasilia')\
    .replace('sao bernardo do capo','sao bernardo do campo')\
    .replace('portoferreira','porto ferreira')\
    .replace('embu guacu','embu-guacu')\
    .replace('ji parana','ji-parana')\
    .replace('aguas claras','agua clara')\
    .replace('vicente de carvalho','rio de janeiro')\
    .replace('centro','belo horizonte')\
    .replace('gama','distrito federal')
 
    return df

    
def atualiza_municipios_nomes(nome):
    nome_munic = nome.replace('AMPARO DA SERRA - MG','AMPARO DO SERRA - MG')\
    .replace('ASSU - RN','ACU - RN')\
    .replace('BALNEARIO DE PICARRAS - SC','BALNEARIO PICARRAS - SC')\
    .replace("OLHO D'AGUA DOS BORGES - RN","OLHO D'AGUA DO BORGES - RN")\
    .replace('SANTA CRUZ DO MONTE CASTELO - PR','SANTA CRUZ DE MONTE CASTELO - PR') \
    .replace('SAO VALERIO DA NATIVIDADE - TO','SAO VALERIO - TO')\
    .replace('SAO THOME DAS LETRAS-MG','SAO TOME DAS LETRAS-MG')\
    .replace('LAGOA DO ITAENGA - PE','LAGOA DE ITAENGA - PE') \
    .replace('SANTA TERESINHA - BA','SANTA TEREZINHA - BA') \
    .replace('ELDORADO DOS CARAJAS - PA','ELDORADO DO CARAJAS - PA') \
    .replace('MUQUEM DE SAO FRANCISCO - BA','MUQUEM DO SAO FRANCISCO - BA') \
    .replace('IGUARACI - PE','IGUARACY - PE')\
    .replace('SAO DOMINGOS DE POMBAL - PB','SAO DOMINGOS - PB') \
    .replace('FLORINIA - SP','FLORINEA - SP')\
    .replace('PRESIDENTE CASTELO BRANCO - SC','PRESIDENTE CASTELLO BRANCO - SC') \
    .replace('PASSA-VINTE - MG','PASSA VINTE - MG')\
    .replace('MUNHOZ DE MELLO - PR','MUNHOZ DE MELO - PR')\
    .replace('BOM JESUS - GO','BOM JESUS DE GOIAS - GO')\
    .replace('MOGI-MIRIM - SP','MOGI MIRIM - SP')\
    .replace("PINGO D'AGUA - MG","PINGO-D'AGUA - MG")\
    .replace('EMBU - SP','EMBU DAS ARTES - SP')\
    .replace('PACAEMBU DAS ARTES - SP','PACAEMBU - SP')\
    .replace("SANTANA DO LIVRAMENTO - RS","SANT'ANA DO LIVRAMENTO - RS")\
    .replace('BRASOPOLIS - MG','BRAZOPOLIS - MG') \
    .replace('BARAO DO MONTE ALTO - MG','BARAO DE MONTE ALTO - MG')\
    .replace('BOA SAUDE - RN','JANUARIO CICCO - RN')\
    .replace('BELEM DE SAO FRANCISCO - PE','BELEM DO SAO FRANCISCO - PE')\
    .replace('SANTA ISABEL DO PARA - PA','SANTA IZABEL DO PARA - PA')\
    .replace('TRAJANO DE MORAIS - RJ','TRAJANO DE MORAES - RJ')\
    .replace('BIRITIBA-MIRIM - SP','BIRITIBA MIRIM - SP') \
    .replace('POXOREO - MT','POXOREU - MT')\
    .replace('NOVA DO MAMORE - RO','NOVA MAMORE - RO')\
    .replace('PARATI - RJ','PARATY - RJ')
    return nome_munic

def resumo_categorias_produtos(df):
  df['product_category_name'] = df['product_category_name'].replace('telefonia_fixa', 'telefonia')\
  .replace('eletrodomesticos_2', 'eletrodomesticos'). replace('bebes', 'brinquedos_e_bebes')\
  .replace('fraldas_higiene', 'brinquedos_e_bebes').replace('brinquedos', 'brinquedos_e_bebes')\
  .replace('artes', 'artes_e_artesanato').replace('bebidas', 'alimentos_bebidas'). replace('alimentos', 'alimentos_bebidas')\
  .replace('casa_conforto_2', 'casa_conforto').replace('musica', 'musicas_cds_dvds_blu_ray')\
  .replace('cds_dvds_musicais', 'musicas_cds_dvds_blu_ray').replace('dvds_blu_ray', 'musicas_cds_dvds_blu_ray')\
  .replace('livros_importados', 'livros').replace('livros_tecnicos', 'livros').replace('livros_interesse_geral', 'livros')\
  .replace('fashion_bolsas_e_acessorios', 'moda_beleza_perfumaria').replace('fashion_calcados', 'moda_beleza_perfumaria')\
  .replace('fashion_underwear_e_moda_praia', 'moda_beleza_perfumaria')\
  .replace('fashion_roupa_masculina', 'moda_beleza_perfumaria').replace('fashion_roupa_feminina', 'moda_beleza_perfumaria')\
  .replace('fashion_esporte', 'moda_beleza_perfumaria').replace('fashion_roupa_infanto_juvenil', 'moda_beleza_perfumaria')\
  .replace('relogios_presentes', 'moda_beleza_perfumaria').replace('perfumaria', 'moda_beleza_perfumaria')\
  .replace('construcao_ferramentas_construcao', 'construção_ferramentas').replace('casa_construcao', 'construção_ferramentas')\
  .replace('construcao_ferramentas_seguranca', 'construção_ferramentas')\
  .replace('construcao_ferramentas_ferramentas', 'construção_ferramentas')\
  .replace('construcao_ferramentas_iluminacao', 'construção_ferramentas')\
  .replace('construcao_ferramentas_jardim', 'construção_ferramentas')\
  .replace('moveis_decoracao', 'moveis_decoracao')\
  .replace('moveis_escritorio', 'moveis_decoracao')\
  .replace('moveis_sala', 'moveis_decoracao')\
  .replace('moveis_cozinha_area_de_servico_jantar_e_jardim', 'moveis_decoracao')\
  .replace('moveis_quarto', 'moveis_decoracao')\
  .replace('moveis_colchao_e_estofado', 'moveis_decoracao')\
  .replace('informatica_acessorios', 'informatica_tablets')\
  .replace('pcs', 'informatica_tablets')\
  .replace('pc_gamer', 'informatica_tablets')\
  .replace('tablets_impressao_imagem', 'informatica_tablets')\
  .replace('agro_industria_e_comercio', 'industria_comercio')\
  .replace('industria_comercio_e_negocios', 'industria_comercio')\
  .replace('portateis_casa_forno_e_cafe', 'portateis_casa')\
  .replace('portateis_cozinha_e_preparadores_de_alimentos', 'portateis_casa')\
  .replace('artigos_de_festas', 'artigos_festas')\
  .replace('artigos_de_natal', 'artigos_festas')\
  .replace('eletronicos', 'eletronicos_games')\
  .replace('consoles_games', 'eletronicos_games')\
  .replace('eletronicos_games', 'eletronicos_games_livros')\
  .replace('livros', 'eletronicos_games_livros')\
  .replace('cine_foto', 'cine_foto_audio')\
  .replace('audio', 'cine_foto_audio')\
  .replace('sinalizacao_e_seguranca', 'sinalizacao_e_seguranca_servicos')\
  .replace('seguros_e_servicos', 'sinalizacao_e_seguranca_servicos')
  return df

def atualiza_municipios_PIB(df,feature):
    df[feature] = df[feature].apply(lambda x:x.replace('AUGUSTO SEVERO - RN','CAMPO GRANDE - RN') )
    df[feature] = df[feature].apply(lambda x:x.replace("PINGO D'AGUA - MG","PINGO-D'AGUA - MG") )
    df[feature] = df[feature].apply(lambda x:x.replace("SAO THOME DAS LETRAS - MG","SAO TOME DAS LETRAS - MG") )
    df[feature] = df[feature].apply(lambda x:x.replace("BIRITIBA MIRIM - SP","BIRITIBA-MIRIM - SP") )
    df[feature] = df[feature].apply(lambda x:x.replace("MOGI GUACU - SP","MOGI-GUACU - SP") )
    df[feature] = df[feature].apply(lambda x:x.replace("SANT'ANA DO LIVRAMENTO - RS","SANTANA DO LIVRAMENTO - RS") )
    df[feature] = df[feature].apply(lambda x:x.replace("SANTA BARBARA D'OESTE - SP","SANTA BARBARA DOESTE - SP") )
    df[feature] = df[feature].apply(lambda x:x.replace("PINDARE-MIRIM - MA","PINDARE MIRIM - MA") )
    df[feature] = df[feature].apply(lambda x:x.replace("BOM JESUS DE GOIAS - GO","BOM JESUS - GO") )
    df[feature] = df[feature].apply(lambda x:x.replace("SANTA IZABEL DO PARA - PA","SANTA ISABEL DO PARA - PA") )
    df[feature] = df[feature].apply(lambda x:x.replace("GRACHO CARDOSO - SE","GRACCHO CARDOSO - SE") ) 
    df[feature] = df[feature].apply(lambda x:x.replace("ESPIGAO D'OESTE - RO","ESPIGAO DO OESTE - RO") ) 
    df[feature] = df[feature].apply(lambda x:x.replace("AMPARO DO SERRA - MG","AMPARO DA SERRA - MG") ) 
    df[feature] = df[feature].apply(lambda x:x.replace("SANTA TEREZINHA - BA","SANTA TERESINHA - BA") ) 
    df[feature] = df[feature].apply(lambda x:x.replace("SANTA RITA DE IBITIPOCA - MG","SANTA RITA DO IBITIPOCA - MG") ) 
    df[feature] = df[feature].apply(lambda x:x.replace("COUTO MAGALHAES - TO","COUTO DE MAGALHAES - TO") ) 
    df[feature] = df[feature].apply(lambda x:x.replace("SAO TOME DAS LETRAS - MG","SAO THOME DAS LETRAS - MG") ) 
    df[feature] = df[feature].apply(lambda x:x.replace("DONA EUSEBIA - MG","DONA EUZEBIA - MG") ) 
    df[feature] = df[feature].apply(lambda x:x.replace("OLHOS-D'AGUA - MG","OLHOS D'AGUA - MG") ) 
    return df

def atualiza_municipios_nomes(df,feature):
    df[feature] = df[feature].replace('lages - sc','lages')\
    .replace('auriflama/sp','auriflama')\
    .replace('sp','sao paulo')\
    .replace('sao paulo - sp','sao paulo')\
    .replace('sao paulo / sao paulo','sao paulo') \
    .replace('são paulo','sao paulo')\
    .replace('cariacica / es','cariacica')\
    .replace("arraial d'ajuda (porto seguro)",'arraial dajuda') \
    .replace('santo andre/sao paulo','santo andre') \
    .replace('maua/sao paulo','maua') \
    .replace('mogi das cruzes / sp','mogi das cruzes') \
    .replace('rio de janeiro \\rio de janeiro','rio de janeiro')\
    .replace('rio de janeiro / rio de janeiro','rio de janeiro') \
    .replace('rio de janeiro, rio de janeiro, brasil','rio de janeiro')\
    .replace('barbacena/ minas gerais','barbacena') \
    .replace('andira-pr','andira')\
    .replace("santa barbara d´oeste",'santa barbara doeste')\
    .replace('ribeirao preto / sao paulo','ribeirao preto')\
    .replace('sao jose do rio pret','sao jose do rio preto')\
    .replace('carapicuiba / sao paulo','carapicuiba')\
    .replace('vendas@creditparts.com.br','Sem definicao')\
    .replace('04482255','Sem definicao')\
    .replace('aguas claras df','aguas claras')\
    .replace('ribeirao pretp','ribeirao preto')\
    .replace('jacarei / sao paulo','jacarei') \
    .replace('brasilia df','brasilia')\
    .replace('sp / sp','sao paulo')\
    .replace('sbc/sp','sao bernardo do campo')\
    .replace('scao jose do rio pardo','sao jose do rio pardo')\
    .replace('pinhais/pr','pinhais')\
    .replace('angra dos reis rj','angra dos reis')\
    .replace('robeirao preto','ribeirao preto')\
    .replace('sao paluo','sao paulo')\
    .replace('sao paulop','sao paulo')\
    .replace('sbc','sao bernando do campo')\
    .replace('belo horizont','belo horizonte')\
    .replace('santa barbara d oeste','santa barbara doeste')\
    .replace('parati','paraty')\
    .replace('sao sebastiao de grama','sao sebastiao de grama')
 
    return df

# https://stackoverflow.com/questions/40452759/pandas-latitude-longitude-to-distance-between-successive-rows
# vectorized haversine function
def haversine(lat1, lon1, lat2, lon2, to_radians=True, earth_radius=6371):
    """
    modificado: of http://stackoverflow.com/a/29546836/2901002

    Calcule a distância do grande círculo entre dois pontos
    na terra (especificado em graus decimais ou em radianos)
    lat, lon) coordenadas são numéricas e igual tamanho.

    """
    lat2 = float(lat2)
    lon2 = float(lon2)
    lat1 = float(lat1)
    lon1 = float(lon1)
    if to_radians:
        try:
            lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
        except:
            print([lat1, lon1, lat2, lon2])

    a = np.sin((lat2-lat1)/2.0)**2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1)/2.0)**2

    return earth_radius * 2 * np.arcsin(np.sqrt(a))

def atualiza_df_dataframes(df,nome_df, label_df,lista):
    numero_rows_nan = nome_df.isnull().sum(axis=1)
    numero_total_nan= len(numero_rows_nan[numero_rows_nan > 0])
    df = df.append(pd.DataFrame({'DataFrame' : [label_df],
                                        'Numero de registros' : [len(nome_df.isnull().sum(axis=1))],
                                        'Features' : [len(nome_df.columns)],
                                        'Duplicados' : [len(nome_df[nome_df.duplicated(subset=lista,keep='first') == True])] ,
                                        'Duplicados(%)' : [np.round((len(nome_df[nome_df.duplicated(subset=lista,keep='first') == True]) /                                           len(nome_df.isnull().sum(axis=1)))*100, 2)],
                                        'Ausentes' : [numero_total_nan],
                                        'Ausentes(%)' : [np.round((numero_total_nan / len(nome_df.isnull().sum(axis=1)))*100,                                                       2)]}),ignore_index=True)
    return df


def faixa_pib_per_capita(num):
    if num > 3285.00 and num <= 11000.00:
        return '(3285,11000]'
    elif num > 11000.00 and num <= 19000.00:
        return '(11000,19000]'
    elif num > 19000.00 and num <= 27000.00:
        return '(19000,27000]'
    elif num > 27000.00 and num <= 35000.00:
        return '(27000,35000]'  
    elif num > 35000.00 and num <= 43000.00:
        return '(35000,43000]'
    elif num > 43000.00 and num <= 51000.00:
        return '(43000,51000]'
    elif num > 51000.00 and num <= 344847.17:
        return '(51000,344847]'    
    
        
def exibe_estatistica(p_df_dataframe, p_df_sem_outliers, p_lista_features,p_plotar):
    """
    Descrição:
        Função que exibir: histograma, qq-plot e  boxplot. Gerar novo dataframe c/ estatisticas
    argumentos:
        p_df_dataframe     -- O pandas dataframe
        p_df_sem_outliers  -- dataframe sem outliers, se não for de interesse, este deve estar vazio
        p_lista_features   -- lista das features
        p_plotar           -- S=sim  N=nao
    Return:
        novo dataframe com estatisticas    
    Exceção:
        Nenhum
    """
    from pandas import DataFrame, Series 
    
    ##  Estatistica registrado no dataframe
    df = DataFrame(columns=['feature','count','mean', 'std','var', 'min', '25%', '50%',           
                        '75%','90%','95%','97%','99.99%','max','skew','kurtosis'])
  
    for var in p_lista_features:
        #print(var)
        reg={}
        reg["feature"]=var
        reg["count"]  = (round(p_df_dataframe[var].describe().loc['count'], 2))        
        reg["mean"]   = round(p_df_dataframe[var].describe().loc['mean'], 2)        
        reg["std"]    = round(p_df_dataframe[var].describe().loc['std'], 2)       
        reg["var"]    = str(round(p_df_dataframe[var].var(),2))        
        reg["min"]    = round(p_df_dataframe[var].describe().loc['min'], 2)         
        reg["25%"]    = round(p_df_dataframe[var].describe().loc['25%'], 2)        
        reg["50%"]    = round(p_df_dataframe[var].describe().loc['50%'], 2)        
        reg["75%"]    = round(p_df_dataframe[var].describe().loc['75%'], 2)        
        reg["90%"]    = round(p_df_dataframe[var].describe(percentiles=[0.90]).loc['90%'], 2)        
        reg["95%"]    = round(p_df_dataframe[var].describe(percentiles=[0.95]).loc['95%'], 2)
        reg["97%"]    = round(p_df_dataframe[var].describe(percentiles=[0.97]).loc['97%'], 2)
        reg["99.99%"] = round(p_df_dataframe[var].describe(percentiles=[0.9999]).loc['99.99%'], 2)
        reg["max"]    = round(p_df_dataframe[var].describe().loc['max'], 2)    
        reg["skew"]   = round(np.abs(p_df_dataframe[var].skew()),2)
        reg["kurtosis"] = round(np.abs(p_df_dataframe[var].kurtosis()),2)
 
        df=df.append(reg, ignore_index=True)
    
        #verifica outliers
        if len(p_df_sem_outliers)>0:
            #print('s/outliers_'+var)
            reg={}
            reg["feature"]='s/outliers_'+var
            reg["count"]  = (round(p_df_sem_outliers[var].describe().loc['count'], 2))        
            reg["mean"]   = round(p_df_sem_outliers[var].describe().loc['mean'], 2)        
            reg["std"]    = round(p_df_sem_outliers[var].describe().loc['std'], 2)       
            reg["var"]    = str(round(p_df_sem_outliers[var].var(),2))        
            reg["min"]    = round(p_df_sem_outliers[var].describe().loc['min'], 2)         
            reg["25%"]    = round(p_df_sem_outliers[var].describe().loc['25%'], 2)        
            reg["50%"]    = round(p_df_sem_outliers[var].describe().loc['50%'], 2)        
            reg["75%"]    = round(p_df_sem_outliers[var].describe().loc['75%'], 2)        
            reg["90%"]    = round(p_df_sem_outliers[var].describe(percentiles=[0.90]).loc['90%'], 2)        
            reg["95%"]    = round(p_df_sem_outliers[var].describe(percentiles=[0.95]).loc['95%'], 2)
            reg["97%"]    = round(p_df_sem_outliers[var].describe(percentiles=[0.97]).loc['97%'], 2)
            reg["99.99%"] = round(p_df_sem_outliers[var].describe(percentiles=[0.9999]).loc['99.99%'], 2)
            reg["max"]    = round(p_df_sem_outliers[var].describe().loc['max'], 2)    
            reg["skew"]   = round(np.abs(p_df_sem_outliers[var].skew()),2)
            reg["kurtosis"] = round(np.abs(p_df_sem_outliers[var].kurtosis()),2)
 
            df=df.append(reg, ignore_index=True)             
        
        if p_plotar == 'S':
            plt.style.use('seaborn')
            plt.figure(figsize=(22, 4))
        
            # histograma
            plt.subplot(1, 3, 1)
            sns.distplot(p_df_dataframe[var], fit=norm,kde=False)
            #sns.histplot( data=p_df_dataframe, x=var, bins=30)
            plt.title('Distplot '+var)

            # Q-Q plot
            plt.subplot(1, 3, 2)
            stats.probplot(p_df_dataframe[var], dist="norm", plot=plt)
            #plt.ylabel('RM quantiles')
            plt.title('Q-Q plot '+var)
 
            # boxplot
            plt.subplot(1, 3, 3)
            sns.boxplot(y=p_df_dataframe[var])
            plt.title('Boxplot '+var)
            
            if len(p_df_sem_outliers)>0:
                plt.figure(figsize=(22, 4))
                
                # histograma
                plt.subplot(1, 4, 1)
                #sns.histplot( data=p_df_sem_outliers, x=var, bins=30)
                sns.distplot(p_df_sem_outliers[var], fit=norm,kde=False)
                plt.title('Distplot s/Outliers '+var)

                # Q-Q plot
                plt.subplot(1, 4, 2)
                stats.probplot(p_df_sem_outliers[var], dist="norm", plot=plt)
                #plt.ylabel('RM quantiles')
                plt.title('Q-Q plot s/Outliers '+var)
 
                # boxplot
                plt.subplot(1, 4, 3)
                sns.boxplot(y=p_df_sem_outliers[var])
                plt.title('Boxplot s/Outliers '+var)                
           
            plt.show()           

    return df 

def localiza_outliers(p_df, p_desvios):   
    """
        Descrição:
        Calcula mean, standard deviation, limites inferior/superior e numero de outliers e porcentagem de outliers 
    argumentos:
        p_df_dataframe     -- O pandas dataframe
        p_desvios          -- numero de desvios-padrão
    Return:
        df_outliers (dict): Dicionario de outliers indicados pelas colunas   
    Exceção:
        Nenhum
    """
    
    df_outliers = {}
    print(f'Outliers maiores que {p_desvios} desvios-padrão')
    print('----------------------------------------------')
    for coluna in p_df:
        outliers = []
        p_df[coluna].astype(float)
    
        # Seta limite superior e inferior
        series_std  = np.std(p_df[coluna])
        series_mean = np.mean(p_df[coluna])
        outlier_corte    = series_std * p_desvios
        limite_inferior  = series_mean - outlier_corte 
        limite_superior  = series_mean + outlier_corte 

        # Gera os outliers
        for indice, valor in p_df[coluna].iteritems():
            if valor > limite_superior or valor < limite_inferior:
                outliers.append({'indice': indice, 'valor': valor})
            
        print(f'{coluna} | mean {round(series_mean,1)} | std {round(series_std,1)} | ll={round(limite_inferior,1)} | '
              f'ul={round(limite_superior,1)} | outliers={len(outliers)} |'
              f'{len(p_df[coluna])} outlier % {len(outliers)/len(p_df[coluna]):.2%}')
        df_outliers[coluna]= outliers
        
    return df_outliers

def univariate_num(p_df,p_lista):
    """
    Descrição:
        Plotar Distplot e Boxplot
    argumentos:
        p_df      -- O pandas dataframe
        p_lista   -- lista das features
    Return:
        Nenhum    
    Exceção:
        Nenhum
    """ 
    for coluna in p_lista:
        #sns.set(style="whitegrid")
        plt.style.use('seaborn')
        plt.figure(figsize=(11,4))
        plt.subplot(121)
        plt.title(f'Distplot de {coluna}')
        sns.distplot(p_df[coluna], bins = 40,kde=True,hist_kws=dict(edgecolor='k', linewidth=0.5))
        #sns.distplot(p_df[coluna],fit=norm,kde=False)
    
        plt.subplot(122)
        plt.title(f'Boxplot de {coluna}')
#        sns.boxplot(y=df[columns].dropna())
        sns.boxplot(y=p_df[coluna])
        plt.show()


def adiciona_features(p_df,p_lista_features):
    """
        Descrição:
        Criar novas features transformadas em log e yeojohnson 
    argumentos:
        p_df             -- O pandas dataframe
        p_lista_features -- features que servem de base para criar novas com: feature+'_log' e feature+'_yeojohnson'
    Return:
        nenhum   
    Exceção:
        Nenhum
    """
    
    for feature in p_lista_features:
        campo = feature+'_log'
        print(campo + ' Ok!')
        X = np.abs(p_df[feature])
        p_df[campo] = np.log(X)
        #np.log(0.0000001)-> -16.11809565095832
        p_df[campo]= np.where(p_df[feature] > 0, np.log(p_df[feature]), 0)
        xt, lo = stats.yeojohnson(X)
        campo2 = feature+'_yeojohnson'
        print(campo2 + ' Ok!')
        p_df[campo2]=xt
    return 

def Pontuacao_R(p_x,p_p,p_d):
    if p_x <= p_d[p_p][0.25]:
        return 4
    elif p_x <= p_d[p_p][0.50]:
        return 3
    elif p_x <= p_d[p_p][0.75]: 
        return 2
    else:
        return 1

def Pontuacao_FM(p_x,p_p,p_d):
    if p_x <= p_d[p_p][0.25]:
        return 1
    elif p_x <= p_d[p_p][0.50]:
        return 2
    elif p_x <= p_d[p_p][0.75]: 
        return 3
    else:
        return 4

def mais_frequente(p_x):
    conta = p_x.value_counts()
    if len(conta): 
        return conta.index[0]
    return 0 

def faixa_plano(p_feature):
    if p_feature <= 30:
        return 'Lite'
    elif p_feature <= 60:
        return 'Basic'
    else:
        return 'Pro'   

def plano_valor(p_feature):
    if p_feature == 'Lite':
        return 18.99
    elif p_feature == 'Basic':
        return 68.99
    else:
        return 173.99   

def plano_comissao(p_feature):
    if p_feature == 'Lite':
        return 0.21
    elif p_feature == 'Basic':
        return 0.20
    else:
        return 0.19   
    
def plot_inertia_sillhoutte(p_df_score,p_df,p_max_cluster,p_n_init,p_max_iter,p_figsize,p_subtitulo, p_algoritmo, p_transf,p_redu):
    """
    Descrição:
        Grafico score sillhoutte e inertia comparativo no mesmo grafico
    argumentos:
        p_df           -- O pandas dataframe
        p_max_cluster  -- numero maximo de clusters
        p_n_init       -- algoritmo tem q rodar n vezes e retornar o mesmo valor
        p_max_iter     -- número maximo de iterações do KMeans
        p_figsize      -- tamanho do grafico
        p_subtitulo    -- subtitulo
        p_transf       -- transformação-técnica
        p_redu         -- redução de dimensões-técnica
    Return:
        dataframe anterior + novos e dataframe somente c/ novos   
    Exceção:
        Nenhum
    """
     
    lista_score_silhouette=[]
    lista_score_davies_bouldin=[]
    lista_score_calinski_harabasz=[]
    lista_score_inertia=[] 
    df_score_local= pd.DataFrame(columns= ['Algor.','Transf.', 'Red.Dim.','Features','ClustersCalc',\
                                    'Clusters', 'inertia','silhouette', 'davies', 'calinski'])
    
    for k in range(2,p_max_cluster):
        kmean = KMeans(n_clusters=k,n_init=p_n_init, max_iter=p_max_iter)
        labels= kmean.fit_predict(p_df)
        reg={}
        reg['Algor.']             = p_algoritmo
        reg['Transf.']            = p_transf        
        reg['Red.Dim.']           = p_redu        
        reg['Features']           = p_df.shape[1]       
        reg['Clusters']            =  k         
        reg['inertia']             =  kmean.inertia_
        lista_score_inertia.append(kmean.inertia_)
        reg['silhouette']          =  silhouette_score(p_df, labels) 
        lista_score_silhouette.append(silhouette_score(p_df, labels))
        reg['davies']              =  davies_bouldin_score(p_df, labels)
        lista_score_davies_bouldin.append(davies_bouldin_score(p_df, labels))
        reg['calinski']            =  calinski_harabasz_score(p_df, labels)
        lista_score_calinski_harabasz.append(calinski_harabasz_score(p_df, labels))        
        df_score_local=df_score_local.append(reg, ignore_index=True)          
   
  
    ## Número ótimo dos clusters 
    ## Métrica inertia:o ponto mais distante da linha que liga os pontos extremos da curva
    #A inertia pode ser reconhecida como uma medida de quão internamente os clusters são coerentes
    
    # lista_score_inertia=soma dos quadrados para as quantidade de clusters
    
    x1, y1 = 2, lista_score_inertia[0]
    x2, y2 = p_max_cluster, lista_score_inertia[len(lista_score_inertia)-1]

    distancias = []
    for i in range(len(lista_score_inertia)):
        x0 = i+2
        y0 = lista_score_inertia[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distancias.append(numerator/denominator)
    calculo= distancias.index(max(distancias)) + 2 
    df_score_local['ClustersCalc']=calculo 
    
    p_df_score=pd.concat([p_df_score,df_score_local],axis=0)
    
    #plotar graficos 
    plt.style.use('seaborn')
    plt.figure(figsize= p_figsize)

    plt.subplot(1, 4, 1)
    sns.lineplot(x= 'Clusters', y= 'inertia', data= df_score_local)
    plt.title('Inertia', fontsize= 20)

    plt.subplot(1, 4, 2)
    sns.lineplot(x= 'Clusters', y= 'silhouette', data= df_score_local)
    plt.title('Coeficiente de Silhouette', fontsize= 20)

    plt.subplot(1, 4, 3)
    sns.lineplot(x= 'Clusters', y= 'davies', data= df_score_local)
    plt.title('Davies-Bouldin', fontsize= 20)

    plt.subplot(1, 4, 4)
    sns.lineplot(x= 'Clusters', y= 'calinski', data= df_score_local)
    plt.title('Calinski-Harabasz', fontsize= 20)

    plt.suptitle(p_subtitulo, fontsize = 20, fontname = 'monospace', weight = 'bold')

    #print("Número ótimo de clusters(cálculo matemático)  : {}".format( calculo))
    display(df_score_local)
    
    return p_df_score, df_score_local     

def plot_inertia_sillhoutte_dbscan(p_df_score, p_df, p_eps_min, p_eps_max,  p_figsize, p_subtitulo, p_algoritmo, p_transf,p_redu):
    """
    Descrição:
        Grafico score sillhoutte e inertia comparativo no mesmo grafico
    argumentos:
        p_df           -- O pandas dataframe
        p_eps_min      -- eps
        p_eps_max      -- eps
        p_figsize      -- tamanho do grafico
        p_subtitulo    -- subtitulo
        p_transf       -- transformação-técnica
        p_redu         -- redução de dimensões-técnica
    Return:
        dataframe anterior + novos e dataframe somente c/ novos   
    Exceção:
        Nenhum
    """
     
    lista_score_silhouette=[]
    lista_score_davies_bouldin=[]
    lista_score_calinski_harabasz=[]
    
    df_score_local= pd.DataFrame(columns= ['Algor.','Transf.', 'Red.Dim.','Features','eps','clusters',\
                    'silhouette', 'davies', 'calinski'])
    
    for e in range(p_eps_min,p_eps_max):
        #converte p/ float
        calculo = e/10
        #lista_epsilon.append(calculo)
        #lista_min_samples.append(m)
        dbs_1 = DBSCAN(eps = calculo, min_samples=2, n_jobs=-1).fit(p_df)
        labels=dbs_1.labels_ 
        #kmean = KMeans(n_clusters=k,n_init=p_n_init, max_iter=p_max_iter
        reg={}
        reg['Algor.']             = p_algoritmo
        reg['Transf.']            = p_transf        
        reg['Red.Dim.']           = p_redu        
        reg['Features']           = p_df.shape[1]       
        reg['eps']                = calculo
        label_DBS=pd.DataFrame(dbs_1.labels_) 

        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        reg['clusters']              = n_clusters_
        lista_score_davies_bouldin.append(davies_bouldin_score(p_df, labels))
        reg['davies']              =  davies_bouldin_score(p_df, labels)
            
        reg['calinski']            =  calinski_harabasz_score(p_df, labels)
        lista_score_calinski_harabasz.append(calinski_harabasz_score(p_df, labels)) 
            
        reg['silhouette']          =  silhouette_score(p_df, labels)
        lista_score_silhouette.append(silhouette_score(p_df, labels)) 
            
        df_score_local=df_score_local.append(reg, ignore_index=True)
        
    p_df_score=pd.concat([p_df_score,df_score_local],axis=0)
    
    #plotar graficos 
    plt.style.use('seaborn')
    plt.figure(figsize= p_figsize)

    plt.subplot(1, 3, 1)
    sns.lineplot(x= 'eps', y= 'silhouette', data= df_score_local)
    plt.title('Coeficiente de Silhouette', fontsize= 20)

    plt.subplot(1, 3, 2)
    sns.lineplot(x= 'eps', y= 'davies', data= df_score_local)
    plt.title('Davies-Bouldin', fontsize= 20)

    plt.subplot(1, 3, 3)
    sns.lineplot(x= 'eps', y= 'calinski', data= df_score_local)
    plt.title('Calinski-Harabasz', fontsize= 20)

    plt.suptitle(p_subtitulo, fontsize = 20, fontname = 'monospace', weight = 'bold')
   
    display(df_score_local)
    
    return p_df_score, df_score_local   


def plot_inertia_sillhoutte_meanshift(p_df_score, p_df,p_quantile_min,p_quantile_max, p_figsize, p_subtitulo, p_algoritmo, p_transf,p_redu):
    """
    Descrição:
        Grafico score sillhoutte e inertia comparativo no mesmo grafico
    argumentos:
        p_df           -- O pandas dataframe
        p_quantile_min -- quantile min
        p_quantile_max -- quantile max
        p_figsize      -- tamanho do grafico
        p_subtitulo    -- subtitulo
        p_transf       -- transformação-técnica
        p_redu         -- redução de dimensões-técnica
    Return:
        dataframe anterior + novos e dataframe somente c/ novos   
    Exceção:
        Nenhum
    """
     
    lista_score_silhouette=[]
    lista_score_davies_bouldin=[]
    lista_score_calinski_harabasz=[]
    
    df_score_local= pd.DataFrame(columns= ['Algor.','Transf.', 'Red.Dim.','Features','quantile','clusters',\
                    'silhouette', 'davies', 'calinski'])
    
    for e in range(1,21):
        #converte p/ float
        #if e_par(e) == 1:
        #    calculo = (e/14)
        #else:
        #    calculo=math.sqrt(e/30)
        #if calculo<1:    
        #    calculo=round(calculo,2)
        
        calculo=round(uniform(p_quantile_min, p_quantile_max),3)

        bandwidth = estimate_bandwidth(p_df, quantile=calculo, n_samples=500)  
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(p_df)
        labels = ms.labels_
        centro_clusters = ms.cluster_centers_
     
        reg={}
        reg['Algor.']             = p_algoritmo
        reg['Transf.']            = p_transf        
        reg['Red.Dim.']           = p_redu        
        reg['Features']           = p_df.shape[1]       
        reg['quantile']           = calculo
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        reg['clusters']              = n_clusters_
    
        lista_score_davies_bouldin.append(davies_bouldin_score(p_df, labels))
        reg['davies']              =  davies_bouldin_score(p_df, labels)
            
        reg['calinski']            =  calinski_harabasz_score(p_df, labels)
        lista_score_calinski_harabasz.append(calinski_harabasz_score(p_df, labels)) 
            
        reg['silhouette']          =  silhouette_score(p_df, labels)
        lista_score_silhouette.append(silhouette_score(p_df, labels))            
        df_score_local=df_score_local.append(reg, ignore_index=True)
        
    p_df_score=pd.concat([p_df_score,df_score_local],axis=0)
    
    #plotar graficos 
    plt.style.use('seaborn')
    plt.figure(figsize= p_figsize)

    plt.subplot(1, 3, 1)
    sns.lineplot(x= 'quantile', y= 'silhouette', data= df_score_local)
    plt.title('Coeficiente de Silhouette', fontsize= 20)

    plt.subplot(1, 3, 2)
    sns.lineplot(x= 'quantile', y= 'davies', data= df_score_local)
    plt.title('Davies-Bouldin', fontsize= 20)

    plt.subplot(1, 3, 3)
    sns.lineplot(x= 'quantile', y= 'calinski', data= df_score_local)
    plt.title('Calinski-Harabasz', fontsize= 20)

    plt.suptitle(p_subtitulo, fontsize = 20, fontname = 'monospace', weight = 'bold')
   
    display(df_score_local)
    
    return p_df_score, df_score_local, centro_clusters     

def teste_kruskal(p_df,*p_args):
    #Teste de Kruskal Wallis
    df_0 = p_df[p_df['cluster'] == 0]
    df_1 = p_df[p_df['cluster'] == 1]
    df_2 = p_df[p_df['cluster'] == 2]
    df_3 = p_df[p_df['cluster'] == 3]
    df_4 = p_df[p_df['cluster'] == 4]
    df_5 = p_df[p_df['cluster'] == 5]
    df_6 = p_df[p_df['cluster'] == 6]
    df_7 = p_df[p_df['cluster'] == 7]
    
    df = pd.DataFrame(columns=['med_0', 'std_0', 
                            'med_1', 'std_1', 
                            'med_2', 'std_2', 
                            'med_3', 'std_3',
                            'med_4', 'std_4',
                            'med_5', 'std_5',
                            'med_6', 'std_6',
                            'med_7', 'std_7',
                            't', 'p'])

    for var in p_args:

        med_0 = round(df_0[var].describe().loc['50%'], 2)
        std_0 = round(df_0[var].describe().loc['std'], 2)
        
        med_1 = round(df_1[var].describe().loc['50%'], 2)
        std_1 = round(df_1[var].describe().loc['std'], 2)
        
        med_2 = round(df_2[var].describe().loc['50%'], 2)
        std_2 = round(df_2[var].describe().loc['std'], 2)
        
        med_3= round(df_3[var].describe().loc['50%'], 2)
        std_3= round(df_3[var].describe().loc['std'], 2)
        
        med_4= round(df_4[var].describe().loc['50%'], 2)
        std_4= round(df_4[var].describe().loc['std'], 2)
        
        med_5= round(df_5[var].describe().loc['50%'], 2)
        std_5= round(df_5[var].describe().loc['std'], 2)
        
        med_6= round(df_6[var].describe().loc['50%'], 2)
        std_6= round(df_6[var].describe().loc['std'], 2)
        
        med_7= round(df_7[var].describe().loc['50%'], 2)
        std_7= round(df_7[var].describe().loc['std'], 2)

        test = stats.kruskal(df_0[var], df_1[var], df_2[var],df_3[var],df_4[var],df_5[var],df_6[var],df_7[var])

        t = round(test[0], 2)
        p = round(test[1], 6)

        ser = pd.Series({'med_0': med_0, 'std_0': std_0, 
                      'med_1': med_1, 'std_1': std_1, 
                      'med_2': med_2, 'std_2': std_2, 
                      'med_3': med_3, 'std_3': std_3, 
                      'med_4': med_4, 'std_4': std_4,
                      'med_5': med_5, 'std_5': std_5,
                      'med_6': med_6, 'std_6': std_6,
                      'med_7': med_7, 'std_7': std_7,
                      't': t, 'p': p})
        ser.name = var
        df = df.append(ser)

    return df

def calcula_lucro_total(p_df):
    lucro = round(p_df['lucro'].sum(), 2)
    return lucro

def verifica_correlacao(p_df_abt,p_intervalo_ini, p_intervalo_fim):
    #Todas as features sem autocorrelação e que têm muita correlação(1) c/ outras.
    df_corr=p_df_abt.corr(method='spearman').abs().unstack().sort_values(ascending= False).reset_index()
    
    df_corr_1=df_corr[(df_corr.level_0!=df_corr.level_1)&((df_corr.iloc[:, 2]>p_intervalo_ini)&(df_corr.iloc[:, 2]<=p_intervalo_fim))] 
    for k in range(df_corr_1.shape[0]):
        campo1=df_corr_1["level_0"].iloc[k]
        campo2=df_corr_1["level_1"].iloc[k]
        test_statistics, pvalue = stats.spearmanr(p_df_abt[campo1], p_df_abt[campo2])
        if(pvalue < 0.05):
            print("Para  {} {}  HÁ CORRELAÇÃO. p-value={} estatistica={:.4f}".format(campo1,campo2, pvalue,test_statistics))
        else:
            print("Para  {} {} NÃO HÁ CORRELAÇÃO. p-value={} estatistica={:.4f}".format(campo1,campo2, pvalue,test_statistics))
    
    return

def conferindo_correlacao(p_df_abt,p_intervalo_ini, p_intervalo_fim):
    #del p_df
    df_corr=p_df_abt.corr(method='spearman').abs().unstack().sort_values(ascending= False).reset_index()
    df_corr_1=df_corr[(df_corr.level_0!=df_corr.level_1)&((df_corr.iloc[:, 2]>p_intervalo_ini)&(df_corr.iloc[:, 2]<=p_intervalo_fim))] 
    #display(df_corr_1)
    print("{} correlações de {} à {} \nNúmero features agora {}.".format(len(df_corr_1),p_intervalo_ini,p_intervalo_fim,p_df_abt.shape[1]))
    return 

def transforma_tlc(p_df_abt, p_df_abt_tlc,p_features_corr):
    #106 features
    for var in p_features_corr:
        #Aplicando Teorem do Limite Central(TLC)
        dist_auxiliar = amostra_means(p_df_abt[var],750,350)        
        df_dist_normal=pd.DataFrame(dist_auxiliar)
        df_dist_normal.columns = [var]
        p_df_abt_tlc[var]=df_dist_normal[var]

    return

def analise_multivariada(p_categoria_1,p_titulo,p_label_x, p_var_fonte):
    plt.style.use('seaborn')
    fig,ax = plt.subplots(1,1,figsize = (18,5))
    font_size = 17
    cat_agrupado_pela_var_fonte = pd.crosstab(index = [p_categoria_1],columns = p_var_fonte, normalize = "index")*100                            
    cat_agrupado_pela_var_fonte.plot.bar(color = ['green', 'red','blue','yellow','orange'],ax=ax)
    #ax.set_title("Comparativo "+p_titulo, size=14, color='black')
    ax.set_xlabel(p_label_x, fontsize = font_size)
    ax.set_ylabel("Frequencia Relativa(%)", fontsize = font_size)
    ax.tick_params(axis="x", labelsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)
   
    plt.suptitle(p_titulo, fontsize = 20, fontname = 'monospace', weight = 'bold')
    plt.legend(loc = "best")
    return plt.show()

def teste_estatistico(p_df,p_feature,p_numero_amostras,p_tamanho_amostra, p_len_dataframe, p_titulo, p_filtro, p_tlc, p_feature_filtro):
    #Aplicando Teorem do Limite Central(TLC)-população
    #dist_total = amostra_means(p_df[p_feature],p_numero_amostras,p_tamanho_amostra)
    #df_dist_total=pd.DataFrame(dist_total)
    #df_dist_total.columns = [p_feature]
    
    #Teorem do Limite Central(TLC)-população
    #p_tlc
    
    if p_filtro == 'Brasil':
        print('Comparativo Original X TLC')
        print('')
        print('')
        plt.style.use('seaborn')
        plt.figure(figsize=(16,4))
        plt.subplot(121)
        plt.title('Original - curtosi= {:.3f}'.format(round(stats.kurtosis(p_df[p_feature], fisher=False),3)))
        sns.distplot(p_df[p_feature], bins = 20,kde=True,hist_kws=dict(edgecolor='k', linewidth=0.5))
   
        plt.subplot(122)
        plt.title('Teorema do Limite Central - curtosi= {:.3f}'.format(round(stats.kurtosis(p_tlc[p_feature], fisher=False),3)))
        sns.distplot(p_tlc[p_feature], bins = 20,kde=True,hist_kws=dict(edgecolor='k', linewidth=0.5))
        plt.suptitle(p_titulo, fontsize = 18, fontname = 'monospace', weight = 'bold')
        plt.show()
        #plt.subplot( 1, 1, 1 )
    else:            
        #Aplicando Teorem do Limite Central(TLC)-amostra
        #df_aux=p_df_abt.loc[p_df_abt[p_feature_filtro] == p_filtro]
        #df_aux=pd.DataFrame(p_df[[p_feature]])
        
        df_aux=p_df.loc[p_df[p_feature_filtro] == p_filtro]
        df_aux=pd.DataFrame(df_aux)
        dist_parcial = amostra_means(df_aux[p_feature],p_numero_amostras,p_tamanho_amostra)
        df_dist_parcial=pd.DataFrame(dist_parcial)
        df_dist_parcial.columns = [p_feature]       
        #executa_t_teste_1_amostra(0.95, df_dist_parcial, df_dist_total, p_titulo, p_feature,p_len_dataframe)
        executa_t_teste_1_amostra(0.95, df_dist_parcial, p_tlc, p_titulo, p_feature,p_len_dataframe)

        
        
def executa_t_teste_1_amostra(p_alfa, p_dist, p_pop_dist, p_titulo,p_feature, p_tamanho_amostra):
    #média de 1 amostra independente
    #Este é um teste para a hipótese nula de que o valor esperado (média) de uma amostra de observações independentes a 
    #é igual à média populacional dada 
    t_teste, pvalue = stats.ttest_1samp(p_dist, p_pop_dist[p_feature].mean())

    if pvalue < 1-p_alfa:
        resultado = ' A amostra é estatisticamente diferente da população! (rejeitamos a hipótese nula).'
    else:
        resultado = 'As amostras não são estatisticamente diferente da população! (não rejeitamos a hipótese nula).'

    #df=grau de liberdade
    #loc=média da amostra
    #scale=erro standard da distribuição
    
    std_erro = p_pop_dist[p_feature].std()/np.sqrt(p_tamanho_amostra)
    media = p_pop_dist[p_feature].mean()
    grau_lib= p_pop_dist.shape[0]-1  

    intervalo = stats.t.interval(alpha=p_alfa, df=grau_lib, loc=media, scale=std_erro)
    print('>>>>>>> ' + p_titulo+':')
    print('Média amostra: {0:.3f}'.format(p_dist[p_feature].mean()))
    print('Média população: {0:.3f}'.format(p_pop_dist[p_feature].mean()))
    print( 'Intervalo de confiança: {0:.3f} <------------> {1:.3f}'.format(intervalo[0], intervalo[1]))
    print('p-valor='+str(pvalue.astype(float)))
   
    print('Resultado: '+resultado)        
    
def vif_calculo(p_x):
    '''
     Descrição:
       Calcula estatistica e p-valor das features do dataframe informado     
    argumentos:  
      p_x:    Datafraame, 
    Return:
      estatistica e p-valor    
    Exceção:
        Nenhum
    
    '''
    vif = pd.DataFrame()
    vif["features"] = p_x.columns
    vif["VIF"] = [variance_inflation_factor(p_x.values, i)  for i in range(p_x.shape[1])]
    return(vif)

def amostragem(p_df, p_amostra_tamanho):
    #amostra_tamanho = 500
    indice = np.random.choice(range(0, p_df.shape[0]), size=p_amostra_tamanho)
    df_amostra = p_df.iloc[indice]
    return df_amostra


def amostra_means( p_df, p_n_amostras, p_amostra_tamanho):
    
    #n_amostras = 15000
    #amostra_tamanho = 500
    amostras_means = []

    for amostra in range(p_n_amostras):
        amostra = amostragem(p_df, p_amostra_tamanho)
        amostras_means.append(amostra.mean())
    return amostras_means

def countplot_com_perc(p_st,p_dados,p_titulo,p_label_x,p_legenda_tit,p_legenda_lista):
    plt.style.use('seaborn')
    countplt, ax = plt.subplots(figsize = (10,7))
    if bool(p_legenda_tit):
        ax =sns.countplot(x = p_st, hue=p_st, data=p_dados, orient='h', dodge=False )     
    else:
        ax =sns.countplot(x = p_st, data=p_dados, orient='h', dodge=False )
        
    #ax.set_title(p_titulo, fontsize=20)
    ax.set_xlabel(p_label_x, fontsize=17)
    ax.set_ylabel("Contagem", fontsize=17)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
    total = len(p_dados)
    
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height()/total)
        x = p.get_x() + p.get_width()
        y = p.get_height()
        ax.annotate(percentage, (x, y),ha='right',size=10)  

    if bool(p_legenda_tit):
        plt.legend(p_legenda_lista, title = p_legenda_tit, bbox_to_anchor = (1.141,1))
        
    plt.suptitle(p_titulo, fontsize = 20, fontname = 'monospace', weight = 'bold')

    countplt
    return

def estatistica_quanti(p_data, p_significancia,p_lista_features):
    """
    Descrição:
        Estatistica Descritiva c/ verificação de testes de normalidade
    argumentos:
        p_data            dataframe a ser analisado
        p_significancia   nivel de significancia, geralmente 0.05
        p_lista_features  features do dataframe a serem analisadas
    Return:
        dataframe com informações    
    Exceção:
        Nenhum
    """  
  
    df = pd.DataFrame(columns=['feature','count','mean', 'median','std','var','Coeficiente_Variacao_%','mod', 'min', '25%', '50%', '75%',\
                'max','range', 'skew','Assimetria',\
                'kurtosis','curtose','NormalTeste','NormalTeste_p','Rejeitar H0','Teste_Kolmogorov','Teste_Kolmogorov_p',\
                'Rejeitar H0-Kolmogorov'])
  
    for var in p_lista_features:
    
        conta = round(p_data[var].describe().loc['count'], 2)
        media = round(p_data[var].describe().loc['mean'], 2)
        mediana = round(p_data[var].median(), 2)
        variancia = round(p_data[var].var(), 2)
        desvio_padrao = round(p_data[var].describe().loc['std'], 2)
        moda = p_data[var].mode()[0]
        minimo = round(p_data[var].describe().loc['min'], 2)
        q1 = round(p_data[var].describe().loc['25%'], 2)
        med = round(p_data[var].describe().loc['50%'], 2)
        q3 = round(p_data[var].describe().loc['75%'], 2)
        maximo = round(p_data[var].describe().loc['max'], 2)
        range = maximo - minimo
        skew = round(p_data[var].skew(), 2)
        kurtosis = round(p_data[var].kurtosis(), 2)  
        
        #H0=é normal
        #Ha=não é normal
        normaltest_teste,normaltest_valor_p = normaltest(p_data[var])
        resposta_H0 = normaltest_valor_p <= p_significancia 
 
        #Teste de Kolmogorov-Smirnov
        #normal_kolmogorov_teste, normal_kolmogorov_valor_p = stats.shapiro(p_data[var])
        normal_kolmogorov_teste, normal_kolmogorov_valor_p = kstest(p_data[var], 'norm')
        resposta_H0_kolmogorov = normal_kolmogorov_valor_p <= p_significancia
        
        #Uso o teste Shapiro-Wilk tem precisão para amostras até 5000, que é o caso(1658) deve ter dist. normal
        #normal_shapiro_teste, normal_shapiro_valor_p = stats.shapiro(p_data[var])        
        #resposta_H0_shapiro = normal_shapiro_valor_p <= p_significancia  
        
        coe_var = round(desvio_padrao / media,2)*100
        
        if media > mediana:
            assimetria='Direita'
        elif mediana > media:
            assimetria='Esquerda'
        elif mediana == media:
            assimetria='Zero'
            
        if kurtosis > 3:
            curtose ='leptocúrtica-cauda fina e pontiaguda'
        elif kurtosis == 3:
            curtose='dist.normal'
        elif kurtosis < 3:
            curtose='platicúrtica-cauda grossa e espessa'       
 
        ser= pd.Series({'feature':var,'count':conta , 'mean':media ,'median':mediana,\
                 'var':variancia, 'std':desvio_padrao ,'Coeficiente_Variacao_%':coe_var,'mod':moda,\
                 'min': minimo, '25%':q1 ,'50%':med ,'75%':q3 ,'max':maximo ,
                 'range':range, 'skew':skew,  'Assimetria':assimetria, 'kurtosis':kurtosis,'curtose':curtose,\
                 'NormalTeste':normaltest_teste,  'NormalTeste_p':normaltest_valor_p, 'Rejeitar H0':resposta_H0,\
                 'Teste_Kolmogorov':normal_kolmogorov_teste, 'Teste_Kolmogorov_p':normal_kolmogorov_valor_p, \
                 'Rejeitar H0-Kolmogorov':resposta_H0_kolmogorov    })
        ser.name=var
        df=df.append(ser)
        #df.reset_index(drop=True, inplace=True)
        df.set_index('feature')
    return df    

def visual_4(p_data, p_lista):
    '''
     Descrição:
       Imprime:histograma, boxplot, violinplot e distibuição cumulativa 
       p/ cada feature passada como parametro.
    
    argumentos:  
      p_data:    Datafraame, 
      p_lista    lista das features
    Return:
        Nenhum    
    Exceção:
        Nenhum
    
    '''
    for coluna in p_lista:
        plt.style.use('seaborn')
        fig = plt.figure(figsize=(16, 8))
        plt.subplots_adjust(hspace = 0.6)
        #sns.set_palette('Pastel1')
    
        plt.subplot(221, frameon=True)                
        ax1 = sns.distplot(p_data[coluna], bins = 20,kde=True,hist_kws={'edgecolor':'k','linewidth':0.5,'density': True} )
        #ax1 = sns.distplot(p_data[coluna], bins=20, hist_kws={"density": True})
        #moda
        ax1.axvline(p_data[coluna].mode()[0], label='Moda',color="green", linestyle=":")      
        # media
        ax1.axvline(np.mean(p_data[coluna]), label='Media',color="red", linestyle="--")  
        # mediana
        ax1.axvline( np.median(p_data[coluna]),label='Mediana', color="blue", linestyle="-" )
       
        plt.title(' Distribuição - '+ coluna+ ' - variancia: '+str(round(p_data[coluna].var(),2)))
        plt.legend() 
        
        plt.subplot(222, frameon=True, sharex=ax1)
        #ax2 = sns.kdeplot(p_data[coluna], cumulative=True)
        ax2 = sns.kdeplot(data=p_data, x=p_data[coluna], cumulative=True)
        plt.title('Distribuição Cumulativa - '+ coluna)   
        
 
        plt.subplot(223, frameon=True)
        stats.probplot(p_data[coluna], dist='norm', plot=plt)
        plt.title('Verificar normalidade - '+coluna)                       
 
        plt.subplot(224, frameon=True, sharex=ax2)
        ax4 = sns.boxplot(x=p_data[coluna], palette = 'cool', width=0.7, linewidth=0.6, showmeans=True)
        plt.title('Outliers - '+ coluna)    
    
        plt.show() 
    return
 



        
