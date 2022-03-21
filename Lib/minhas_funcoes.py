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
import matplotlib.pyplot as plt
import time
from scipy import stats
from scipy.stats    import normaltest, kstest, norm



#import datetime
#from datetime import time

#eliminando os espacos entre as tags
def Trata_HTML(input):
    return " ".join(input.split()).replace('> <','><')

#Separa cfe tamanho definido no prametro
def Separador(v,tamanho):
    return v.strip().zfill(tamanho)[0:tamanho]


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
