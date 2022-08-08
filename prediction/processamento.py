import pandas as pd
import numpy as np
from prediction.conecta import get_respostas
from prediction.models import FatorEST, DimensaoEST, Fator, Dimensao

def loadData():
    #df = pd.read_csv('Dataset/Coleta_Piloto_Colunas.csv')
    df = pd.read_json(get_respostas())
    # dataframe para os indices fatores medio baixo e medio alto
    d = {'Fator': ['E-ESC1', 'E-ESC2', 'E-PROF1', 'E-PROF2', 'E-FAM1', 'E-FAM2', 'E-COM1', 'E-COM2', 'E-COM3', 'E-EST1',
                   'E-EST2', 'E-EST3'],
         'Medio_Baixo': [3.66, 2.01, 2.34, 2.34, 3.01, 3.34, 2.01, 2.01, 2.01, 2.01, 2.34, 2.01],
         'Medio_Alto': [5.33, 3.33, 4.33, 4.00, 5.00, 5.00, 3.66, 3.66, 3.66, 4.00, 4.00, 3.33]}
    fatores = pd.DataFrame(data=d)
    for index, fator in fatores.iterrows():
        fator_db, created = Fator.objects.update_or_create(
            fator=fatores.loc[index, 'Fator'],
            defaults={
                'medio_baixo': fatores.loc[index, 'Medio_Baixo'],
                'medio_alto': fatores.loc[index, 'Medio_Alto'],
            }
        )
    print(df.columns)

    # dataframe para os indices dimensoes medio baixo e medio alto
    d = {'Dimensao': ['E-ESC', 'E-PROF', 'E-FAM', 'E-COM', 'E-EST'],
         'Medio_Baixo': [2.84, 2.67, 3.34, 2.45, 2.44],
         'Medio_Alto': [4, 3.66, 4.33, 3.55, 3.41]}
    dimensoes = pd.DataFrame(data=d)
    for index, dimensao in dimensoes.iterrows():
        dimensao_db, created = Dimensao.objects.update_or_create(
            dimensao=dimensoes.loc[index, 'Dimensao'],
            defaults={
                'medio_baixo': dimensoes.loc[index, 'Medio_Baixo'],
                'medio_alto': dimensoes.loc[index, 'Medio_Alto'],
            }
        )

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
        eesc1 = row[["QA1", "QA26", 'QA6']].mean(skipna=True)
        eesc2 = row[["QA3", "QE1", 'QE6']].mean(skipna=True)

        eprof1 = row[["QA17", "QA7", 'QA12']].mean(skipna=True)
        eprof2 = row[["QA22", "QA27", 'QA31']].mean(skipna=True)

        efam1 = row[["QA23", "QE12", 'QE3']].mean(skipna=True)#QA23,QE12,QE03
        efam2 = row[["QA13", "QA5", 'QA18']].mean(skipna=True)#QA13,QA05,QA18

        ecom1 = row[["QE9", "QE16", 'QE4']].mean(skipna=True)#QE09,QE16,QE04
        ecom2 = row[["QA4", "QA19", 'QA24']].mean(skipna=True)#QA04,QA19,QA24
        ecom3 = row[["QA18", 'QA11', 'QE15']].mean(skipna=True)#QA18,QA11,QE15

        eest1 = row[["QA32", "QA9", 'QA25']].mean(skipna=True)#QA32, QA09, QA25
        eest2 = row[["QA10", "QA25", 'QA15']].mean(skipna=True)#QA10, QA25, QA15
        eest3 = row[["QE10", "QE12", 'QE4']].mean(skipna=True)#QE10, QE18, QE11



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

        fatores_Est.loc[(fatores_Est[colV] >= medB) & (fatores_Est[colV] <= medA), colC] = '2'
        fatores_Est.loc[fatores_Est[colV] < medB, colC] = '1'
        fatores_Est.loc[fatores_Est[colV] > medA, colC] = '3'
        
    fatores_Est = fatores_Est.fillna(0)
    fatores_est_banco = []
    count = 0
    for index, row in df.iterrows():
        fator_est, created = FatorEST.objects.update_or_create(
            id_aluno=fatores_Est.loc[index, 'IDALUNO'],
            defaults={
                'E_ESC1V': fatores_Est.loc[index, 'E-ESC1V'],
                'E_ESC2V': fatores_Est.loc[index, 'E-ESC2V'],
                'E_PROF1V': fatores_Est.loc[index, 'E-PROF1V'],
                'E_PROF2V': fatores_Est.loc[index, 'E-PROF2V'],
                'E_FAM1V': fatores_Est.loc[index, 'E-FAM1V'],
                'E_FAM2V': fatores_Est.loc[index, 'E-FAM2V'],
                'E_COM1V': fatores_Est.loc[index, 'E-COM1V'],
                'E_COM2V': fatores_Est.loc[index, 'E-COM2V'],
                'E_COM3V': fatores_Est.loc[index, 'E-COM3V'],
                'E_EST1V': fatores_Est.loc[index, 'E-EST1V'],
                'E_EST2V': fatores_Est.loc[index, 'E-EST2V'],
                'E_EST3V': fatores_Est.loc[index, 'E-EST3V'],
                'E_ESC1C': fatores_Est.loc[index, 'E-ESC1C'],
                'E_ESC2C': fatores_Est.loc[index, 'E-ESC2C'],
                'E_PROF1C': fatores_Est.loc[index, 'E-PROF1C'],
                'E_PROF2C': fatores_Est.loc[index, 'E-PROF2C'],
                'E_FAM1C': fatores_Est.loc[index, 'E-FAM1C'],
                'E_FAM2C': fatores_Est.loc[index, 'E-FAM2C'],
                'E_COM1C': fatores_Est.loc[index, 'E-COM1C'],
                'E_COM2C': fatores_Est.loc[index, 'E-COM2C'],
                'E_COM3C': fatores_Est.loc[index, 'E-COM3C'],
                'E_EST1C': fatores_Est.loc[index, 'E-EST1C'],
                'E_EST2C': fatores_Est.loc[index, 'E-EST2C'],
                'E_EST3C': fatores_Est.loc[index, 'E-EST3C'],
            }
        )
        if created:
            count += 1
        fatores_est_banco.append(fator_est)

    print("Print Fatores")
    print(fatores_Est)

    # for pra computar indices e valores
    for index, row in fatores_Est.iterrows():
        eesc = row[['E-ESC1V', "E-ESC2V"]].mean(skipna=True)

        eprof = row[["E-PROF1V", "E-PROF2V"]].mean(skipna=True)

        efam = row[["E-FAM1V", "E-FAM2V"]].mean(skipna=True)

        ecom = row[["E-COM1V", "E-COM2V", 'E-COM3V']].mean(skipna=True)

        eest = row[["E-EST1V", "E-EST2V", 'E-EST3V']].mean(skipna=True)

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
        dimensoes_Est.loc[(dimensoes_Est[colV] >= medB) & (dimensoes_Est[colV] <= medA), colC] = '2'
        dimensoes_Est.loc[dimensoes_Est[colV] < medB, colC] = '1'
        dimensoes_Est.loc[dimensoes_Est[colV] > medA, colC] = '3'
    print("Print Dimensões")
    print(dimensoes_Est)

    dimensoes_Est = dimensoes_Est.fillna(0)
    dimensoes_est_banco = []
    count = 0
    for index, row in dimensoes_Est.iterrows():
        dimensoes_est, created = DimensaoEST.objects.update_or_create(
            id_aluno=dimensoes_Est.loc[index, 'IDALUNO'],
            defaults={
                'E_ESCV': dimensoes_Est.loc[index, 'E-ESCV'],
                'E_PROFV': dimensoes_Est.loc[index, 'E-PROFV'],
                'E_FAMV': dimensoes_Est.loc[index, 'E-FAMV'],
                'E_COMV': dimensoes_Est.loc[index, 'E-COMV'],
                'E_ESTV': dimensoes_Est.loc[index, 'E-ESTV'],
                'E_ESCC': dimensoes_Est.loc[index, 'E-ESCC'],
                'E_PROFC': dimensoes_Est.loc[index, 'E-PROFC'],
                'E_FAMC': dimensoes_Est.loc[index, 'E-FAMC'],
                'E_COMC': dimensoes_Est.loc[index, 'E-COMC'],
                'E_ESTC': dimensoes_Est.loc[index, 'E-ESTC'],
            }
        )
        if created:
            count += 1
        dimensoes_est_banco.append(dimensoes_est)
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

def processa_from_dict(dados):
    df = pd.DataFrame(dados)
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
    fatores_Est = fatores_Est.append(pd.DataFrame(Id, columns=['IDALUNO']), ignore_index=True)

    # dataframe dimensoes uma coluna com o id do estudante,
    # uma coluna pro valor e outra pra classificacao totalizando 10 colunas
    dimensoes_Est = pd.DataFrame(
        columns=['IDALUNO', 'E-ESCV', 'E-PROFV', 'E-FAMV', 'E-COMV', 'E-ESTV', 'E-ESCC', 'E-PROFC', 'E-FAMC', 'E-COMC', 'E-ESTC'])
    Id = list(df.IDALUNO)
    dimensoes_Est = dimensoes_Est.append(pd.DataFrame(Id, columns=['IDALUNO']), ignore_index=True)

    # for pra computar indices e valores
    for index, row in df.iterrows():
        eesc1 = row[["QA1", "QA26", 'QA6']].mean(skipna=True)
        eesc2 = row[["QA3", "QE1", 'QE6']].mean(skipna=True)

        eprof1 = row[["QA17", "QA7", 'QA12']].mean(skipna=True)
        eprof2 = row[["QA22", "QA27", 'QA31']].mean(skipna=True)

        efam1 = row[["QA23", "QE12", 'QE3']].mean(skipna=True)#QA23,QE12,QE03
        efam2 = row[["QA13", "QA5", 'QA18']].mean(skipna=True)#QA13,QA05,QA18

        ecom1 = row[["QE9", "QE16", 'QE4']].mean(skipna=True)#QE09,QE16,QE04
        ecom2 = row[["QA4", "QA19", 'QA24']].mean(skipna=True)#QA04,QA19,QA24
        ecom3 = row[["QA18", 'QA11', 'QE15']].mean(skipna=True)#QA18,QA11,QE15

        eest1 = row[["QA32", "QA9", 'QA25']].mean(skipna=True)#QA32, QA09, QA25
        eest2 = row[["QA10", "QA25", 'QA15']].mean(skipna=True)#QA10, QA25, QA15
        eest3 = row[["QE10", "QE12", 'QE4']].mean(skipna=True)#QE10, QE18, QE11


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
        
    fatores_Est = fatores_Est.fillna(0)
    print("Print Fatores")
    print(fatores_Est)

    # for pra computar indices e valores
    for index, row in fatores_Est.iterrows():
        eesc = row[['E-ESC1V', "E-ESC2V"]].mean(skipna=True)

        eprof = row[["E-PROF1V", "E-PROF2V"]].mean(skipna=True)

        efam = row[["E-FAM1V", "E-FAM2V"]].mean(skipna=True)

        ecom = row[["E-COM1V", "E-COM2V", 'E-COM3V']].mean(skipna=True)

        eest = row[["E-EST1V", "E-EST2V", 'E-EST3V']].mean(skipna=True)

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
        dimensoes_Est.loc[(dimensoes_Est[colV] >= medB) & (dimensoes_Est[colV] <= medA), colC] = 'Risco Médio'
        dimensoes_Est.loc[dimensoes_Est[colV] < medB, colC] = 'Risco Baixo'
        dimensoes_Est.loc[dimensoes_Est[colV] > medA, colC] = 'Risco Alto'

    dimensoes_Est = dimensoes_Est.fillna(0)
    print("Print Dimensões")
    print(dimensoes_Est)

    # EESCC = pd.DataFrame({'Risco': dimensoes_Est.groupby('E-ESCC')['IDALUNO'].count()}).reset_index()
    # EESCC = EESCC.rename(columns={"Risco": "Total", 'E-ESCC': 'Risco'})

    # EFAMC = pd.DataFrame({'Risco': dimensoes_Est.groupby('E-FAMC')['IDALUNO'].count()}).reset_index()
    # EFAMC = EFAMC.rename(columns={"Risco": "Total", 'E-FAMC': 'Risco'})

    # ECOMC = pd.DataFrame({'Risco': dimensoes_Est.groupby('E-COMC')['IDALUNO'].count()}).reset_index()
    # ECOMC = ECOMC.rename(columns={"Risco": "Total", 'E-COMC': 'Risco'})

    # EESTC = pd.DataFrame({'Risco': dimensoes_Est.groupby('E-ESTC')['IDALUNO'].count()}).reset_index()
    # EESTC = EESTC.rename(columns={"Risco": "Total", 'E-ESTC': 'Risco'})

    # EPROF = pd.DataFrame({'Risco': dimensoes_Est.groupby('E-PROFC')['IDALUNO'].count()}).reset_index()
    # EPROF = EPROF.rename(columns={"Risco": "Total", 'E-PROFC': 'Risco'})

    # return [EPROF, EESTC, EESCC, EFAMC, ECOMC]
    result = [fatores_Est, dimensoes_Est]  
    return map(df_to_dict, result)

def df_to_dict(df):
    return df.to_dict(orient='records')