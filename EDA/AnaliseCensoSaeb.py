import sys
import seaborn as sns
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
import pandas as pd
import os
from factor_analyzer import FactorAnalyzer
import numpy as np

def load():
    try:
        dataset = pd.read_csv('../Dataset/inep_sabe_merge_2019_new.csv', delimiter='\t')
        print(dataset.shape)

        return dataset;

    except:
        print("Oops!", sys.exc_info()[0], "occurred.")

def checkFeasibility(dataset):
    columns_numeric = pd.DataFrame(dataset._get_numeric_data()).columns
    columns_categorical = list(pd.DataFrame(dataset.select_dtypes(['object'])).columns)
    columns_categorical.append('ID_SAEB')
    columns_categorical.append('CO_ENTIDADE')

    print("Categorical Columns")
    print(columns_categorical)
    drop_columns = columns_numeric
    dataset_reduce = dataset.drop(columns=columns_categorical, axis=1)

    result = dataset_reduce.isna().mean()
    dataset_reduce = dataset_reduce.loc[:, result < .1]

    nunique = dataset_reduce.nunique()
    cols_to_drop = nunique[nunique == 1].index
    dataset_reduce = dataset_reduce.drop(cols_to_drop, axis=1)
    print("Dimensionality reduced from {} to {}.".format(dataframe.shape, dataset_reduce.shape))


    chi_square_value, p_value = calculate_bartlett_sphericity(dataset_reduce)
    print(chi_square_value, p_value);

    kmo_all, kmo_model = calculate_kmo(dataset_reduce)
    print('kmo: ', kmo_model)

if __name__ == '__main__':
    dataframe = load()
    checkFeasibility(dataframe)