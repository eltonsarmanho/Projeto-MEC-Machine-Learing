import numpy as np
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt

def loadData():
    try:
        dataset = pd.read_csv("../Dataset/inep_sabe_merge_all.csv",low_memory=False);

        #for key, value in dict_uf.items():
        #    dataset.loc[dataset["ID_UF"] == str(key) , "ID_UF"] = value
        #dataset.to_csv('../Dataset/inep_sabe_merge_all4.csv', index=False)
        return dataset;
    except:
         print("Oops!", sys.exc_info()[0], "occurred.");

def plotDiscriminate(dataset):
    dataset_uf = dataset[dataset['ID_REGIAO'] == 'Norte'].copy()

    # Encontrar a média
    #media = dataset_uf.NECESSIDADE_ESPECIAL.mean()
    # Make a column that denotes which day had highest O3
    dataset_uf['Suporte'] = ['EXISTE' if value1 >0 or value2 >0 or value3 >0 else 'NAO EXISTE'
                                          for value1, value2,value3 in zip(dataset_uf.IN_ESPECIAL_EXCLUSIVA,
                                                           dataset_uf.IN_ACESSIBILIDADE_RAMPAS,
                                                           dataset_uf.IN_BANHEIRO_PNE)]



    # Encode the hue of the points with the O3 generated column
    sns.scatterplot(x='NECESSIDADE_ESPECIAL',
                    y='DEF_FISICA',
                    hue='Suporte',
                    data=dataset_uf)
    plt.show()



    pass;
if __name__ == '__main__':
    df = loadData()
    plotDiscriminate(df)