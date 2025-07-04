import sys
import seaborn as sns
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
import pandas as pd
import os
from factor_analyzer import FactorAnalyzer
import numpy as np
import matplotlib.pyplot as plt

def load():
    try:
        dataset = pd.read_csv('../Dataset/2019/inep_saeb_merge_2019.csv', delimiter='\t')
        print(dataset.shape)
        print(dataset.columns)

        return dataset;

    except:
        print("Oops!", sys.exc_info()[0], "occurred.")

def reduceNoise(data,threshould,indicator):
    a = np.array(data.values.tolist())
    return np.where(a >= threshould, indicator, a).tolist()

def preprocessamento(df):
    # Dropping unnecessary columns
    # Pegar 10 valores mais frequentes
    n = 10
    data = df['QT_PROF_PSICOLOGO']
    mode = data.value_counts()[:n].index.tolist()
    df['QT_PROF_PSICOLOGO'] = reduceNoise(data, data.mean(), max(mode))

    df['IN_ACESSO_INTERNET_COMPUTADOR'] = reduceNoise(df['IN_ACESSO_INTERNET_COMPUTADOR'], data.mean(), data.mean())


    for parametro in ['NECESSIDADE_ESPECIAL','IN_BIBLIOTECA']:
        data = df[parametro]
        threshould = getValue_upper_whisker_quarile(data)[0]
        indicator = getValue_upper_whisker_quarile(data)[1]
        df[parametro] = reduceNoise(data,threshould,indicator)

    for parametro in ['BAIXA_VISAO', 'DEF_AUDITIVA', 'DEF_FISICA', 'DEF_INTELECTUAL', 'TRANSPORTE_PUBLICO',
                      'TRANSP_ONIBUS']:
        data = df[parametro]
        mode = data.value_counts()[:n].index.tolist()
        threshould = max(mode)
        df[parametro] = reduceNoise(data, threshould, threshould)

    columns_drop = ['IN_REGULAR', 'IN_SERIE_ANO', 'REGULAR', 'IN_FUNDAMENTAL_CICLOS',
                    'IN_BANHEIRO_FUNCIONARIOS', 'IN_COMUM_FUND_AI', 'IN_DORMITORIO_ALUNO',
                    'IN_COMUM_PRE', 'IN_REDES_SOCIAIS', 'IN_PERIODOS_SEMESTRAIS', 'IN_PROFISSIONALIZANTE', 'IN_EJA',
                    'IN_QUADRA_ESPORTES_COBERTA','IN_MEDIACAO_PRESENCIAL',
                    'PROFISSIONALIZANTE', 'IN_RESERVA_PPI', 'IN_RESERVA_PUBLICA', 'IN_COMUM_MEDIO_INTEGRADO',
                    'IN_ESGOTO_FOSSA_COMUM','IN_ESGOTO_FOSSA',
                    'IN_INTERNET', 'IN_INTERNET_ADMINISTRATIVO', 'IN_INTERNET_APRENDIZAGEM',
                    'IN_BIBLIOTECA',
                    #'IN_COZINHA', 'IN_DESPENSA','IN_ALMOXARIFADO','IN_BIBLIOTECA',
                    #'IN_PATIO_COBERTO','IN_SALA_PROFESSOR','IN_ACESSIBILIDADE_PISOS_TATEIS', 'IN_INTERNET',
                    #'IN_INTERNET_ADMINISTRATIVO', 'IN_INTERNET_APRENDIZAGEM',
                    #'IN_TRATAMENTO_LIXO_SEPARACAO', 'IN_TRATAMENTO_LIXO_REUTILIZA', 'IN_TRATAMENTO_LIXO_RECICLAGEM','IN_TRATAMENTO_LIXO_INEXISTENTE',
                    #'AEE_LIBRAS', 'AEE_BRAILLE',
                    'IN_EQUIP_IMPRESSORA_MULT',
                    'IN_COMUM_MEDIO_MEDIO',
                    'EDUCACAO_INDIGENA', 'IN_EDUCACAO_INDIGENA',
                    'IN_ORGAO_ASS_PAIS', 'IN_ORGAO_ASS_PAIS_MESTRES', 'IN_ORGAO_CONSELHO_ESCOLAR','IN_ORGAO_GREMIO_ESTUDANTIL', 'IN_ORGAO_OUTROS', 'IN_ORGAO_NENHUM'
                    ]

    df.drop(columns_drop, axis=1, inplace=True)
    return df


def getValue_upper_whisker_quarile(data):
    median = np.median(data)
    upper_quartile = np.percentile(data, 75)
    lower_quartile = np.percentile(data, 25)

    iqr = upper_quartile - lower_quartile
    upper_whisker = data[data <= upper_quartile + 1.5 * iqr].max()
    lower_whisker = data[data >= lower_quartile - 1.5 * iqr].min()

    return upper_whisker,upper_quartile

def checkFeasibility(dataset):

    columns_numeric = pd.DataFrame(dataset._get_numeric_data()).columns
    columns_categorical = list(pd.DataFrame(dataset.select_dtypes(['object'])).columns)
    columns_categorical.append('CO_ENTIDADE')
    columns_categorical.append('ID_REGIAO')
    columns_categorical.append('ID_UF')

    print("Categorical Columns")
    print(columns_categorical)

    #Remove Categorical Columns
    dataset_reduce = dataset.drop(columns=columns_categorical, axis=1)
    dataset_reduce = dataset_reduce.astype(float)

    #Remove as colunas com linhas acima 10% NA
    result = dataset_reduce.isna().mean()
    dataset_reduce = dataset_reduce.loc[:, result < .1]

    #Remove colunas com valores iguais em todas as linhas
    nunique = dataset_reduce.nunique()
    cols_to_drop = nunique[nunique == 1].index
    dataset_reduce.drop(cols_to_drop, axis=1,inplace=True)

    #Remove as columnas relacionadas a códigos
    columns = dataset_reduce.columns;
    filtered = filter(lambda name: name.find("CO_") != -1, columns);

    dataset_reduce.drop(filtered, axis=1, inplace=True)

    print("Dimensionality reduced from {} to {}.".format(dataset.shape, dataset_reduce.shape))

    #Teste de Factorability
    chi_square_value, p_value = calculate_bartlett_sphericity(dataset_reduce)
    print(chi_square_value, p_value);

    #Teste de Adequacidade dos dados
    kmo_all, kmo_model = calculate_kmo(dataset_reduce)
    print('kmo: ', kmo_model)

    # # Criamos objeto factor_analysis, sem rotação e usando 5 fatores (tentativamente)
    # fa = FactorAnalyzer(n_factors=40, rotation="promax")
    #
    # # Aplicamos o método fit (ajuste) desse objeto no dataframe
    # fa.fit(dataset_reduce)
    #
    # # Depois desse ajuste podemos coletar os autovetores e autovalores
    # ev, v = fa.get_eigenvalues()
    # print('São ' + str(len(ev)) + ' autovalores:\n', ev)
    #
    # # Create scree plot using matplotlib
    # plt.scatter(range(1, dataset_reduce.shape[1] + 1), ev)
    # plt.plot(range(1, dataset_reduce.shape[1] + 1), ev)
    # plt.title('Scree Plot')
    # plt.xlabel('Factors')
    # plt.ylabel('Eigenvalue')
    # plt.grid()
    # plt.axhline(y=1, c='k')
    # plt.show()

    #eigenvalues, _ = fa.get_eigenvalues()
    # count eigenvalues > 1
    #number_of_factors = sum(eigenvalues > 1)
    #print('Numero de Fatores = {}'.format(number_of_factors))

    #Apos saber Numero de Fatores com eigenvalues > 1
    fa = FactorAnalyzer(12, rotation="varimax")

    # o objeto tem o método fit para análise do dataframe
    fa.fit(dataset_reduce)

    # Desse extraimos as cargas fatoriais (factor loadings)
    # Observe que fa.loadings_ é um numpy.array com shape . Usamos o método
    # do pandas pd.DataFrame.from_records para convertê-lo em um dataframe
    factorLoadings = pd.DataFrame.from_records(fa.loadings_)

    # Para ver a dataframe gerado:
    # factorLoadings.head(4)
    # Substitue as linhas pelo nomes dos itens
    factorLoadings.index = dataset_reduce.columns

    threshould = 0.55
    drop_factos = [c for c in factorLoadings.columns if len(factorLoadings[abs(factorLoadings[c]) > threshould]) == 0]
    factorLoadings.drop(drop_factos, axis=1, inplace=True)
    for c in factorLoadings.columns:
        data = factorLoadings[abs(factorLoadings[c])>threshould]
        print("Fator {} formado por {}.".format(c,data[c].index.values.tolist()))

    #Selecionar as linhas acima de threshould e abaixo -threshould
    factorLoadings = factorLoadings[factorLoadings.gt(threshould).any(axis=1) | factorLoadings.lt(-1*threshould).any(axis=1)]
    print(factorLoadings.shape)

    plt.figure(figsize=(8, 6))
    sns.set(font_scale=.9)
    sns.heatmap(factorLoadings, linewidths=1, linecolor='#ffffff', cmap="RdYlGn", xticklabels=1, yticklabels=1)
    plt.show()
if __name__ == '__main__':
    dataset = load()
    dataset = preprocessamento(dataset)
    checkFeasibility(dataset)