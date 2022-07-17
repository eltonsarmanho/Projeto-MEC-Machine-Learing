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

        efam1 = row[["QA23", "QE12", 'QE3']].mean()#QA23,QE12,QE03
        efam2 = row[["QA13", "QA5", 'QA18']].mean()#QA13,QA05,QA18

        ecom1 = row[["QE9", "QE16", 'QE4']].mean()#QE09,QE16,QE04
        ecom2 = row[["QA4", "QA19", 'QA24']].mean()#QA04,QA19,QA24
        ecom3 = row[["QA18", 'QA11', 'QE15']].mean()#QA18,QA11,QE15

        eest1 = row[["QA32", "QA9", 'QA25']].mean()#QA32, QA09, QA25
        eest2 = row[["QA10", "QA25", 'QA15']].mean()#QA10, QA25, QA15
        eest3 = row[["QE10", "QE12", 'QE4']].mean()#QE10, QE18, QE11



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

        fatores_Est.loc[(fatores_Est[colV] >= medB) & (fatores_Est[colV] <= medA), colC] = 'Risco Médio'
        fatores_Est.loc[fatores_Est[colV] < medB, colC] = 'Risco Baixo'
        fatores_Est.loc[fatores_Est[colV] > medA, colC] = 'Risco Alto'

    print("Print Fatores")
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
        dimensoes_Est.loc[(dimensoes_Est[colV] >= medB) & (dimensoes_Est[colV] <= medA), colC] = 'Risco Médio'
        dimensoes_Est.loc[dimensoes_Est[colV] < medB, colC] = 'Risco Baixo'
        dimensoes_Est.loc[dimensoes_Est[colV] > medA, colC] = 'Risco Alto'

    print("Print Dimensões")
    print(dimensoes_Est)

    #Escreve CSV
    #fatores_Est.to_csv('')

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

    return EPROF,EESTC,EESCC,EFAMC,ECOMC

def piePlot(EPROF,EESTC,EESCC,EFAMC,ECOMC):

    #title = "Habilidade e receptividade do Professor "
    dimensao = ['E-EPROF','E-EST','E-ESC','E-FAM','E-COM']
    title = "Risco por dimenstões"

    fig, axes = plt.subplots(ncols=2,nrows=3, figsize=(4, 2), dpi=100)

    plt.suptitle(title)
    axe = axes.ravel()

    ax1 = plt.subplot2grid(shape=(2, 6), loc=(0, 0), colspan=2)
    ax2 = plt.subplot2grid((2, 6), (0, 2), colspan=2)
    ax3 = plt.subplot2grid((2, 6), (0, 4), colspan=2)
    ax4 = plt.subplot2grid((2, 6), (1, 1), colspan=2)
    ax5 = plt.subplot2grid((2, 6), (1, 3), colspan=2)
    axes = [ax1,ax2,ax3,ax4,ax5]
    for ax,titulo,feature in zip(axes,dimensao,[EPROF,EESTC,EESCC,EFAMC,ECOMC]):

        ax.pie(feature['Total'], labels=feature['Risco'], startangle=90, autopct='%1.0f%%', textprops={'fontsize': 14})
        ax.set_title(titulo + ' em % ',fontsize=10, bbox={'facecolor': '0.8', 'pad': 5})


    plt.show()

if __name__ == '__main__':
    df,fatores_Est,fatores,dimensoes_Est,dimensoes = loadData()
    EPROF,EESTC,EESCC,EFAMC,ECOMC = processing(df,fatores_Est,fatores,dimensoes_Est,dimensoes)
    piePlot(EPROF,EESTC,EESCC,EFAMC,ECOMC)