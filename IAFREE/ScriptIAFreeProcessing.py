import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(color_codes=True)
def loadData():
    df = pd.read_csv('Data/Coleta_Piloto_Colunas.csv')
    # dataframe para os indices fatores medio baixo e medio alto
    d = {'Fator': ['E-ESC1', 'E-ESC2', 'E-PROF1', 'E-PROF2', 'E-FAM1', 'E-FAM2', 'E-COM1', 'E-COM2', 'E-COM3', 'E-EST1',
                   'E-EST2', 'E-EST3'],
         'Medio_Baixo': [3.66, 2.01, 2.34, 2.34, 3.01, 3.34, 2.01, 2.01, 2.01, 2.01, 2.34, 2.01],
         'Medio_Alto': [5.33, 3.33, 4.33, 4.00, 5.00, 5.00, 3.66, 3.66, 3.66, 4.00, 4.00, 3.33]}
    fatores = pd.DataFrame(data=d)
    print(df.columns)

    # dataframe para os indices dimensoes medio baixo e medio alto
    d = {'Dimensao': ['E-ESC', 'E-PROF', 'E-FAM', 'E-COM', 'E-EST'],
         'Medio_Baixo': [2.84, 2.67, 3.34, 2.45, 2.44],
         'Medio_Alto': [4, 3.66, 4.33, 3.55, 3.41]}
    dimensoes = pd.DataFrame(data=d)

    # dataframe fatores uma coluna com o id do estudante, uma
    # coluna pro valor e outra pra classificacao totalizando 24 colunas
    fatores_Est = pd.DataFrame(
        columns=['IDALUNO', 'E-ESC1V', 'E-ESC2V', 'E-PROF1V', 'E-PROF2V', 'E-FAM1V', 'E-FAM2V', 'E-COM1V',
                 'E-COM2V', 'E-COM3V', 'E-EST1V', 'E-EST2V', 'E-EST3V', 'E-ESC1C', 'E-ESC2C', 'E-PROF1C',
                 'E-PROF2C', 'E-FAM1C', 'E-FAM2C', 'E-COM1C', 'E-COM2C', 'E-COM3C', 'E-EST1C', 'E-EST2C',
                 'E-EST3C'])
    Id = list(df.IDALUNO)
    fatores_Est = fatores_Est.append(pd.DataFrame(Id,
                                                  columns=['IDALUNO']),
                                     ignore_index=True)

    # dataframe dimensoes uma coluna com o id do estudante,
    # uma coluna pro valor e outra pra classificacao totalizando 10 colunas
    dimensoes_Est = pd.DataFrame(
        columns=['IDALUNO', 'E-ESCV', 'E-PROFV', 'E-FAMV', 'E-COMV', 'E-ESTV', 'E-ESCC', 'E-PROFC',
                 'E-FAMC', 'E-COMC', 'E-ESTC'])
    Id = list(df.IDALUNO)
    dimensoes_Est = dimensoes_Est.append(pd.DataFrame(Id,
                                                      columns=['IDALUNO']),
                                         ignore_index=True)

    return df,fatores_Est,fatores,dimensoes_Est,dimensoes;
def processing(df,fatores_Est,fatores,dimensoes_Est,dimensoes):
    # for pra computar indices e valores
    for index, row in df.iterrows():
        eesc1 = row[["QA1", "QA26", 'QA6']].mean()
        eesc2 = row[["QA3", "QE1", 'QE6']].mean()

        eprof1 = row[["QA17", "QA7", 'QA12']].mean()
        eprof2 = row[["QA22", "QA27", 'QA31']].mean()

        efam1 = row[["QA23", "QE12", 'QE3']].mean()
        efam2 = row[["QA13", "QA5", 'QA18']].mean()

        ecom1 = row[["QE9", "QE16", 'QE4']].mean()
        ecom2 = row[["QA4", "QA19", 'QA24']].mean()
        ecom3 = row[["QA18", 'QA11', 'QE15']].mean()

        eest1 = row[["QA32", "QA9", 'QA25']].mean()
        eest2 = row[["QA10", "QA25", 'QA15']].mean()
        eest3 = row[["QE10", "QE12", 'QE4']].mean()

        fatores_Est.loc[index, 'E-ESC1V'] = eesc1
        fatores_Est.loc[index, 'E-ESC2V'] = eesc2

        fatores_Est.loc[index, 'E-PROF1V'] = eprof1
        fatores_Est.loc[index, 'E-PROF2V'] = eprof2

        fatores_Est.loc[index, 'E-FAM1V'] = efam1
        fatores_Est.loc[index, 'E-FAM2V'] = efam2

        fatores_Est.loc[index, 'E-COM1V'] = ecom1
        fatores_Est.loc[index, 'E-COM2V'] = ecom2
        fatores_Est.loc[index, 'E-COM3V'] = ecom3

        fatores_Est.loc[index, 'E-EST1V'] = eest1
        fatores_Est.loc[index, 'E-EST2V'] = eest2
        fatores_Est.loc[index, 'E-EST3V'] = eest3

        for index, row in fatores.iterrows():
            medB = row['Medio_Baixo']
            medA = row['Medio_Alto']
            col = row['Fator']
            colC = col + 'C'
            colV = col + 'V'
            # print(colC)
            for i, r in fatores_Est.iterrows():
                v = r[colV]
                if (v >= medB and v <= medA):
                    # print('Risco Médio')
                    # print("Fator: ", col, ", Média Baixa: ", medB, ', Media Alta: ', medA, ', Valor: ', v)
                    sit = 'Risco Médio'
                elif (v < medB):
                    # print('Risco Baixo')
                    # print("Fator: ", col, ", Média Baixa: ", medB, ', Media Alta: ', medA, ', Valor: ', v)
                    sit = 'Risco Baixo'
                elif (v > medA):
                    # print('Risco Alto')
                    # print("Fator: ", col, ", Média Baixa: ", medB, ', Media Alta: ', medA, ', Valor: ', v)
                    sit = 'Risco Alto'
                fatores_Est.loc[i, colC] = sit


    print(fatores_Est)

    # for pra computar indices e valores
    for index, row in fatores_Est.iterrows():
        eesc = row[['E-ESC1V', "E-ESC2V"]].mean()

        eprof = row[["E-PROF1V", "E-PROF2V"]].mean()

        efam = row[["E-FAM1V", "E-FAM2V"]].mean()

        ecom = row[["E-COM1V", "E-COM2V", 'E-COM3V']].mean()

        eest = row[["E-EST1V", "E-EST2V", 'E-EST3V']].mean()

        dimensoes_Est.loc[index, 'E-ESCV'] = eesc

        dimensoes_Est.loc[index, 'E-PROFV'] = eprof

        dimensoes_Est.loc[index, 'E-FAMV'] = efam

        dimensoes_Est.loc[index, 'E-COMV'] = ecom

        dimensoes_Est.loc[index, 'E-ESTV'] = eest

    for index, row in dimensoes.iterrows():
        medB = row['Medio_Baixo']
        medA = row['Medio_Alto']
        col = row['Dimensao']
        colC = col + 'C'
        colV = col + 'V'
        # print(colC)
        for i, r in dimensoes_Est.iterrows():
            v = r[colV]
            if (v >= medB and v <= medA):
                # print('Risco Médio')
                # print("Fator: ", col, ", Média Baixa: ", medB, ', Media Alta: ', medA, ', Valor: ', v)
                sit = 'Risco Médio'
            elif (v < medB):
                # print('Risco Baixo')
                # print("Fator: ", col, ", Média Baixa: ", medB, ', Media Alta: ', medA, ', Valor: ', v)
                sit = 'Risco Baixo'
            elif (v > medA):
                # print('Risco Alto')
                # print("Fator: ", col, ", Média Baixa: ", medB, ', Media Alta: ', medA, ', Valor: ', v)
                sit = 'Risco Alto'
            dimensoes_Est.loc[i, colC] = sit

    print(dimensoes_Est)

    EESCC = pd.DataFrame({'Risco': dimensoes_Est.groupby('E-ESCC')['IDALUNO'].count()}).reset_index()
    # EPROF['E-PROFC'] = 'Risco'
    EESCC = EESCC.rename(columns={"Risco": "Total", 'E-ESCC': 'Risco'})

    EFAMC = pd.DataFrame({'Risco': dimensoes_Est.groupby('E-FAMC')['IDALUNO'].count()}).reset_index()
    # EPROF['E-PROFC'] = 'Risco'
    EFAMC = EFAMC.rename(columns={"Risco": "Total", 'E-FAMC': 'Risco'})

    ECOMC = pd.DataFrame({'Risco': dimensoes_Est.groupby('E-COMC')['IDALUNO'].count()}).reset_index()
    # EPROF['E-PROFC'] = 'Risco'
    ECOMC = ECOMC.rename(columns={"Risco": "Total", 'E-COMC': 'Risco'})

    EESTC = pd.DataFrame({'Risco': dimensoes_Est.groupby('E-ESTC')['IDALUNO'].count()}).reset_index()
    # EPROF['E-PROFC'] = 'Risco'
    EESTC = EESTC.rename(columns={"Risco": "Total", 'E-ESTC': 'Risco'})

    EPROF = pd.DataFrame({'Risco': dimensoes_Est.groupby('E-PROFC')['IDALUNO'].count()}).reset_index()
    # EPROF['E-PROFC'] = 'Risco'
    EPROF = EPROF.rename(columns={"Risco": "Total", 'E-PROFC': 'Risco'})

    plt.figure(figsize=(10, 15))
    # set_style("whitegrid")
    # fig, axs = plt.subplots(ncols=2)
    # plt.style.use('seaborn-paper')
    f, axes = plt.subplots(3, 2, figsize=(10, 15))
    sns.barplot(x="Risco", y="Total", data=EPROF, ci=68, ax=axes[0, 0])
    sns.barplot(x="Risco", y="Total", data=EESTC, ci=68, ax=axes[0, 1])
    sns.barplot(x="Risco", y="Total", data=EESCC, ci=68, ax=axes[1, 0])
    sns.barplot(x="Risco", y="Total", data=EFAMC, ci=68, ax=axes[1, 1])
    sns.barplot(x="Risco", y="Total", data=ECOMC, ci=68, ax=axes[2, 0])

    axes[0, 0].title.set_text('E-EPROF')
    axes[0, 1].title.set_text('E-EST')
    axes[1, 0].title.set_text('E-ESC')
    axes[1, 1].title.set_text('E-FAM')
    axes[2, 0].title.set_text('E-COM')

    # sns.boxplot(x='education',y='wage', data=df_melt, ax=axs[2])
    plt.suptitle('Risco por dimenstões')


if __name__ == '__main__':
    df,fatores_Est,fatores,dimensoes_Est,dimensoes = loadData()
    processing(df,fatores_Est,fatores,dimensoes_Est,dimensoes)