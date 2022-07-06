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
                    'IN_ESGOTO_FOSSA_COMUM']

    df.drop(columns_drop, axis=1, inplace=True)
    return df


def getValue_upper_whisker_quarile(data):
    median = np.median(data)
    upper_quartile = np.percentile(data, 75)
    lower_quartile = np.percentile(data, 25)

    iqr = upper_quartile - lower_quartile
    upper_whisker = data[data <= upper_quartile + 1.5 * iqr].max()
    lower_whisker = data[data >= lower_quartile - 1.5 * iqr].min()
    print(upper_whisker)
    return upper_whisker,upper_quartile

def checkFeasibility(dataset):
    columns_numeric = pd.DataFrame(dataset._get_numeric_data()).columns
    columns_categorical = list(pd.DataFrame(dataset.select_dtypes(['object'])).columns)
    columns_categorical.append('CO_ENTIDADE')

    print("Categorical Columns")
    print(columns_categorical)
    dataset_reduce = dataset.drop(columns=columns_categorical, axis=1)
    dataset_reduce = dataset_reduce.astype(float)

    result = dataset_reduce.isna().mean()

    dataset_reduce = dataset_reduce.loc[:, result < .1]
    #Remove colunas com valores iguais em todas as linhas
    nunique = dataset_reduce.nunique()
    cols_to_drop = nunique[nunique == 1].index
    dataset_reduce.drop(cols_to_drop, axis=1,inplace=True)

    columns = dataset_reduce.columns;
    filtered = filter(lambda name: name.find("CO_") != -1, columns);
    dataset_reduce.drop(filtered, axis=1, inplace=True)

    print("Dimensionality reduced from {} to {}.".format(dataset.shape, dataset_reduce.shape))
    print((dataset_reduce.values < 0).any())
    chi_square_value, p_value = calculate_bartlett_sphericity(dataset_reduce)
    print(chi_square_value, p_value);

    kmo_all, kmo_model = calculate_kmo(dataset_reduce)
    print('kmo: ', kmo_model)

    # Criamos objeto factor_analysis, sem rotação e usando 5 fatores (tentativamente)
    #fa = FactorAnalyzer(40, rotation=None)

    # Aplicamos o método fit (ajuste) desse objeto no dataframe
    #fa.fit(dataset_reduce)

    # Depois desse ajuste podemos coletar os autovetores e autovalores
    #ev, v = fa.get_eigenvalues()
    #print('São ' + str(len(ev)) + ' autovalores:\n', ev)

    # Create scree plot using matplotlib
    #plt.scatter(range(1, dataset_reduce.shape[1] + 1), ev)
    #plt.plot(range(1, dataset_reduce.shape[1] + 1), ev)
    #plt.title('Scree Plot')
    #plt.xlabel('Factors')
    #plt.ylabel('Eigenvalue')
    #plt.grid()
    #plt.axhline(y=1, c='k')
    #plt.show()

    # 6 fatores
    fa = FactorAnalyzer(17, rotation="varimax")

    # o objeto tem o método fit para análise do dataframe
    fa.fit(dataset_reduce)

    # Desse extraimos as cargas fatoriais (factor loadings)
    # Observe que fa.loadings_ é um numpy.array com shape (25,6). Usamos o método
    # do pandas pd.DataFrame.from_records para convertê-lo em um dataframe
    factorLoadings = pd.DataFrame.from_records(fa.loadings_)

    # Para ver a dataframe gerado:
    #factorLoadings.head(4)
    # Substitue as linhas pelo nomes dos itens
    factorLoadings.index = dataset_reduce.columns
    #print(factorLoadings.describe())
    data_filtered = factorLoadings.copy()

    for c in data_filtered.columns:
        data = data_filtered[abs(data_filtered[c])>0.6]
        print("Fator {} formado por {}.".format(c,data[c].index.values.tolist()))

    plt.figure(figsize=(8, 6))
    sns.set(font_scale=.9)
    sns.heatmap(factorLoadings, linewidths=1, linecolor='#ffffff', cmap="YlGnBu", xticklabels=1, yticklabels=1)
    plt.show()
if __name__ == '__main__':
    dataset = load()
    dataset = preprocessamento(dataset)
    checkFeasibility(dataset)