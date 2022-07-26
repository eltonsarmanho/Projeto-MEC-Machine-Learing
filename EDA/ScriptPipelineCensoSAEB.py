
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import ListedColormap

pd.set_option("display.max.columns", None)
import sys
import matplotlib.pyplot as plt
import time
from dask import dataframe as dd
import glob
import os
from tqdm import tqdm
import itertools
import csv
import os
import re
import gc

class ScriptPipelineCensoSAEB:


    def splitFileWithDask(self,file, ano):
        file_path = "../Dataset/" + str(ano) + "/Regioes/" + file + ".CSV"
        dataframe = dd.read_csv(file_path, sep='|', dtype={'CO_MUNICIPIO_END': 'float64',
                                                           'CO_MUNICIPIO_NASC': 'float64',
                                                           'TP_LOCAL_RESID_DIFERENCIADA': 'float64',
                                                           'CO_UF_END': 'float64',
                                                           'CO_UF_NASC': 'float64',
                                                           'IN_EJA': 'float64',
                                                           'IN_ESPECIAL_EXCLUSIVA': 'float64',
                                                           'IN_PROFISSIONALIZANTE': 'float64',
                                                           'IN_REGULAR': 'float64',
                                                           'NU_DURACAO_TURMA': 'float64',
                                                           'NU_DUR_AEE_MESMA_REDE': 'float64',
                                                           'NU_DUR_AEE_OUTRAS_REDES': 'float64',
                                                           'NU_DUR_ATIV_COMP_MESMA_REDE': 'float64',
                                                           'NU_DUR_ATIV_COMP_OUTRAS_REDES': 'float64',
                                                           'TP_ETAPA_ENSINO': 'float64',
                                                           'IN_TRANSPORTE_PUBLICO': 'float64',
                                                           'NU_DIAS_ATIVIDADE': 'float64',
                                                           'TP_TIPO_LOCAL_TURMA': 'float64',
                                                           'TP_ZONA_RESIDENCIAL': 'float64',
                                                           'TP_UNIFICADA': 'float64'})
        print("Dimensionality total {}.".format(dataframe.shape))

        # set how many file you would like to have
        # in this case 10
        dataframe = dataframe.repartition(npartitions=20)
        dataframe.to_csv("../Dataset/" + str(ano) + "/" + file + "_file_*.csv", sep='\t')

    def runDimensionReduction(self,url, nameNewFile=None,ano=None):
        try:
            drop_columns = ['NU_ANO_CENSO', 'NU_ANO', 'NU_IDADE_REFERENCIA', 'NU_IDADE',
                            'TP_NACIONALIDADE', 'CO_PAIS_ORIGEM', 'CO_UF_NASC',
                            'CO_MUNICIPIO_NASC', 'CO_UF_END', 'CO_MUNICIPIO_END',
                            'IN_RECURSO_CD_AUDIO', 'IN_RECURSO_NENHUM', 'IN_AEE_OPTICOS_NAO_OPTICOS',
                            'IN_AEE_ENRIQ_CURRICULAR', 'TP_ETAPA_ENSINO', 'TP_MEDIACAO_DIDATICO_PEDAGO',
                            'CO_REGIAO', 'CO_MICRORREGIAO', 'CO_UF', 'CO_MUNICIPIO',
                            'CO_DISTRITO', 'TP_DEPENDENCIA', 'TP_LOCALIZACAO', 'TP_CATEGORIA_ESCOLA_PRIVADA',
                            'IN_CONVENIADA_PP', 'TP_CONVENIO_PODER_PUBLICO', 'IN_MANT_ESCOLA_PRIVADA_EMP',
                            'IN_MANT_ESCOLA_PRIVADA_ONG', 'IN_MANT_ESCOLA_PRIVADA_OSCIP',
                            'IN_MANT_ESCOLA_PRIV_ONG_OSCIP', 'IN_MANT_ESCOLA_PRIVADA_SIND',
                            'IN_MANT_ESCOLA_PRIVADA_SIST_S', 'IN_MANT_ESCOLA_PRIVADA_S_FINS',
                            'IN_RECURSO_TRANSCRICAO', 'IN_RECURSO_LEDOR',
                            'IN_SUPERDOTACAO', 'IN_DEF_MULTIPLA', 'IN_SURDOCEGUEIRA', 'IN_RECURSO_PROVA_PORTUGUES',
                            'TP_LOCAL_RESID_DIFERENCIADA', 'IN_RECURSO_AMPLIADA_18', 'IN_RECURSO_AMPLIADA_24',
                            'IN_AEE_LINGUA_PORTUGUESA', 'IN_AEE_INFORMATICA_ACESSIVEL', 'IN_AEE_CAA',
                            'IN_AEE_SOROBAN', 'IN_AEE_VIDA_AUTONOMA', 'IN_AEE_DESEN_COGNITIVO',
                            'IN_AEE_MOBILIDADE', 'TP_OUTRO_LOCAL_AULA', 'TP_RESPONSAVEL_TRANSPORTE',
                            'TP_REGULAMENTACAO', 'TP_LOCALIZACAO_DIFERENCIADA', 'TP_UNIFICADA', 'NU_DIAS_ATIVIDADE',
                            'TP_TIPO_ATENDIMENTO_TURMA', 'TP_TIPO_LOCAL_TURMA',
                            'NU_DUR_ATIV_COMP_OUTRAS_REDES', 'NU_DUR_ATIV_COMP_MESMA_REDE', 'NU_DURACAO_TURMA',
                            'NU_DUR_AEE_OUTRAS_REDES', 'NU_DUR_AEE_MESMA_REDE', 'CO_CURSO_EDUC_PROFISSIONAL',]

            #dataframe = dd.read_csv(url, sep='|', usecols=lambda x: x not in drop_columns, )
            dataframe = dd.read_csv(url, sep='\t',usecols=lambda x: x not in drop_columns,dtype='object')
            print("Dimensionality Reduzida {}.".format(dataframe.shape))
            #dataset_reduce = transformData(dataframe)
            dataframe.to_csv('../Dataset/'+str(ano)+'/'+nameNewFile+'.csv',sep='\t', encoding='utf-8', index_label='indice',index=False, single_file=True)
            #return dataframe
        except:
            print("Oops!", sys.exc_info(), "occurred.");

    def removeColumnWithDask(self,url, separate=None, newFile=None, ano=None):
        if (separate != None):
            dataset = dd.read_csv(url, dtype='object', sep=separate)
        else:
            dataset = dd.read_csv(url, dtype='object', blocksize="10MB")

        if newFile != None:
            dataset_reduce = dataset.drop('Unnamed: 0', axis=1)
            dataset_reduce.to_csv('../Dataset/' + str(ano) + '/' + newFile + '.csv', single_file=True)

        print("Dimensionality reduced from {} to {}.".format(dataset.shape, dataset_reduce.shape))

    def concatCSVbyRows(self,filefind, file, ano):
        files = os.path.join("../Dataset/" + str(ano) + "", filefind + '*.csv')

        all_files = glob.glob(files)
        print(all_files)
        out_file = '../Dataset/' + str(ano) + '/' + file + '.csv';
        with open(out_file, 'w') as outfile:
            for i, filename in enumerate(all_files):
                print(i, filename)
                with open(filename, 'r') as infile:
                    for rownum, line in enumerate(infile):
                        if (i != 0) and (rownum == 0):  # Only write header once
                            continue
                        outfile.write(line)

    def concatCSVWithDask(self,filefind, file, ano):
        # list of merged files returned
        files = os.path.join("../Dataset/" + ano, filefind + '*.csv')

        all_files = glob.glob(files)
        print(all_files)
        out_file = '../Dataset/' + ano + '/' + file + '.csv';
        with open(out_file, 'w') as outfile:
            for i, filename in enumerate(all_files):
                print(i, filename)
                with open(filename, 'r') as infile:
                    for rownum, line in enumerate(infile):
                        if (i != 0) and (rownum == 0):  # Only write header once
                            continue
                        outfile.write(line)

    def runDimensionReductionSAEB(self, url, nameNewFile=None, ano=None):
        try:
            drop_columns = ['NU_ANO_CENSO',
                            'CO_ORGAO_REGIONAL', 'IN_VINCULO_OUTRO_ORGAO',
                            'IN_CONVENIADA_PP', 'TP_CONVENIO_PODER_PUBLICO',
                            'IN_MANT_ESCOLA_PRIVADA_EMP', 'IN_MANT_ESCOLA_PRIVADA_ONG',
                            'IN_MANT_ESCOLA_PRIVADA_OSCIP', 'IN_MANT_ESCOLA_PRIV_ONG_OSCIP',
                            'IN_MANT_ESCOLA_PRIVADA_SIND', 'IN_MANT_ESCOLA_PRIVADA_SIST_S',
                            'IN_MANT_ESCOLA_PRIVADA_S_FINS', 'TP_REGULAMENTACAO',
                            'TP_RESPONSAVEL_REGULAMENTACAO', 'TP_OCUPACAO_PREDIO_ESCOLAR',
                            'IN_LOCAL_FUNC_SOCIOEDUCATIVO', 'IN_LOCAL_FUNC_UNID_PRISIONAL',
                            'IN_LOCAL_FUNC_PRISIONAL_SOCIO', 'IN_LOCAL_FUNC_GALPAO', 'TP_OCUPACAO_GALPAO',
                            'IN_LOCAL_FUNC_SALAS_OUTRA_ESC', 'IN_LOCAL_FUNC_OUTROS', 'IN_PREDIO_COMPARTILHADO',
                            'TP_INDIGENA_LINGUA', 'CO_LINGUA_INDIGENA_1', 'CO_LINGUA_INDIGENA_2',
                            'CO_LINGUA_INDIGENA_3',
                            # Variaveis Quantitativas
                            'QT_SALAS_UTILIZADAS_DENTRO', 'QT_SALAS_UTILIZADAS_FORA', 'QT_SALAS_UTILIZADAS',
                            'QT_SALAS_UTILIZA_CLIMATIZADAS', 'QT_SALAS_UTILIZADAS_ACESSIVEIS',
                            'CO_DISTRITO',
                            'TP_DEPENDENCIA', 'TP_LOCALIZACAO', 'TP_LOCALIZACAO_DIFERENCIADA',
                            'IN_VINCULO_SECRETARIA_EDUCACAO', 'IN_VINCULO_SEGURANCA_PUBLICA',
                            'IN_VINCULO_SECRETARIA_SAUDE', 'TP_CATEGORIA_ESCOLA_PRIVADA', 'CO_ESCOLA_SEDE_VINCULADA',
                            'CO_IES_OFERTANTE', 'IN_LOCAL_FUNC_PREDIO_ESCOLAR',
                            'IN_ESP_EXCLUSIVA_MEDIO_INTEGR', 'IN_ESP_EXCLUSIVA_MEDIO_NORMAL',
                            'IN_COMUM_EJA_FUND', 'IN_COMUM_EJA_MEDIO', 'IN_COMUM_EJA_PROF',
                            'IN_ESP_EXCLUSIVA_EJA_FUND', 'IN_ESP_EXCLUSIVA_EJA_MEDIO',
                            'IN_ESP_EXCLUSIVA_EJA_PROF', 'IN_COMUM_PROF', 'IN_ESP_EXCLUSIVA_PROF',
                            'TP_REDE_LOCAL', 'TP_SITUACAO_FUNCIONAMENTO',
                            # Variaveis Temporais
                            'DT_ANO_LETIVO_INICIO', 'DT_ANO_LETIVO_TERMINO',
                            # Variaveis geograficas
                            'CO_ORGAO_REGIONAL', 'DT_ANO_LETIVO_INICIO', 'DT_ANO_LETIVO_TERMINO',
                            'CO_DISTRITO',
                            'TP_DEPENDENCIA', 'TP_LOCALIZACAO',
                            'TP_LOCALIZACAO_DIFERENCIADA', 'TP_LOCALIZACAO',
                            'CO_ESCOLA_SEDE_VINCULADA', 'CO_IES_OFERTANTE', 'TP_REGULAMENTACAO',
                            'TP_RESPONSAVEL_REGULAMENTACAO', 'TP_PROPOSTA_PEDAGOGICA',
                            'TP_AEE', 'TP_ATIVIDADE_COMPLEMENTAR'
                            ]
            # dataframe = dd.read_csv(url, sep='|',  dtype='object')
            # dataset_reduce = pd.read_csv(url, sep='|', usecols=lambda x: x not in drop_columns )
            # Abrir arquivo setar utf-8
            dataset_reduce = pd.read_csv(url, sep='|', encoding="utf-8", usecols=lambda x: x not in drop_columns)

            # print("Dimensionality reduced from {} to {}.".format(dataframe.shape, dataset_reduce.shape))
            dataset_reduce.update(dataset_reduce[['NO_ENTIDADE']].applymap('"{}"'.format))

            print(dataset_reduce.head(5))
            print(dataset_reduce.shape)
            # dataset_reduce = transformData(dataframe)
            dataset_reduce.to_csv('../Dataset/' + ano + '/' + nameNewFile, sep='\t', encoding='utf-8', index=False)
            del (dataset_reduce)
            gc.collect()
        except:
            print("Oops!", sys.exc_info(), "occurred.");

    def splitFileWithDaskSAEB(self,ano):
        file_path = "../Dataset/" + str(ano) + "/matricula_reduzido_all_" + str(ano) + ".csv"
        df = dd.read_csv(file_path,
                         dtype={'CO_ENTIDADE': 'float64',
                                'CO_MESORREGIAO': 'float64',
                                'ID_MATRICULA': 'float64',
                                'ID_TURMA': 'float64',
                                'IN_EDUCACAO_INDIGENA': 'float64',
                                'IN_EJA': 'float64',
                                'IN_ESPECIAL_EXCLUSIVA': 'float64',
                                'IN_NECESSIDADE_ESPECIAL': 'float64',
                                'IN_PROFISSIONALIZANTE': 'float64',
                                'IN_REGULAR': 'float64',
                                'IN_TRANSPORTE_PUBLICO': 'float64',
                                'NU_MES': 'float64',
                                'TP_COR_RACA': 'float64',
                                'TP_SEXO': 'float64',
                                'TP_ZONA_RESIDENCIAL': 'float64'})
        print(df.shape)
        # set how many file you would like to have
        # in this case 10
        df = df.repartition(npartitions=35)
        df.to_csv("../Dataset/" + str(ano) + "/matricula_reduzido_all_" + str(ano) + "_*.csv", sep='\t',
                  encoding='utf-8', index=False)
        del (df)
        gc.collect()

    def loadData(self,indice, ano):

        start = time.time()
        dataset_escola = pd.read_csv('../Dataset/' + str(ano) + '/escola_update.csv', sep='\t', )
        dataset_matricula = pd.read_csv(
            '../Dataset/' + str(ano) + '/matricula_reduzido_all_' + str(ano) + '_' + str(indice) + '.csv',
            low_memory=True, sep='\t')

        end = time.time()
        print("Read csv: ", (end - start), "sec")
        print(dataset_escola.shape)
        print(dataset_matricula.shape)

        # Tratamento dos dados
        # dataset_matricula = dataset_matricula.fillna(0)
        # dataset_escola.fillna(0)
        # calculateMissingValues(dataset_matricula)
        # calculateMissingValues(dataset_escola)
        return dataset_matricula, dataset_escola;

    def calculateIndicators(self,dataset_escola, dataset_matricula, ano, indice=None):

        columns = dataset_matricula.columns;
        filtered = filter(lambda name: name.find("IN_") != -1, columns);

        colunas = list(filtered)
        for coluna in colunas:
            dataset_matricula[coluna] = pd.to_numeric(dataset_matricula[coluna], errors="coerce")
        colunas.insert(0, 'CO_ENTIDADE')
        # print(colunas)
        # print(dataset_matricula.dtypes)

        # estatistica_alunos_por_escola = dataset_matricula[colunas].groupby(['CO_ENTIDADE']).agg({"IN_DEF_INTELECTUAL": ["sum"]},split_out=4)
        estatistica_alunos_por_escola = dataset_matricula[colunas].groupby(['CO_ENTIDADE'], as_index=False).agg(["sum"],
                                                                                                                split_out=8)
        # estatistica_alunos_por_escola = dataset_matricula[colunas].groupby(['CO_ENTIDADE'],as_index=False).agg({lambda x: (x == 1).sum()},split_out=8)
        estatistica_alunos_por_escola = estatistica_alunos_por_escola.reset_index()
        # print(estatistica_alunos_por_escola.head(5))

        # Padronizar colunas
        colunas_rename = [w.replace('IN_', '') for w in colunas]
        estatistica_alunos_por_escola.columns = colunas_rename

        df_result = pd.merge(estatistica_alunos_por_escola, dataset_escola, on='CO_ENTIDADE')

        # Check duplicidade
        print("Check duplicidade: ", df_result['CO_ENTIDADE'].duplicated().any())
        print('Shape dataframe estatistica Escola ', df_result.shape)

        df_result.to_csv('../Dataset/' + str(ano) + '/dataset_escola_filtered_' + str(indice) + '.csv', sep='\t',
                         encoding='utf-8', index=False)

        del (df_result)
        gc.collect()

    def concatCSV(self,ano):
        # setting the path for joining multiple files
        files = os.path.join("../Dataset/" + ano,"dataset_escola_filtered_*.csv")

        CHUNK_SIZE = 1024
        # list of merged files returned
        files = glob.glob(files)
        print(files)
        print("Resultant CSV after joining all CSV files at a particular location...");

        # reading files using read_csv with chunk
        # chunk_container = [pd.read_csv(f, chunksize=CHUNK_SIZE, sep='\t') for f in files]

        # joining files using chunk
        # combined_csv = pd.concat(map(pd.DataFrame, chunk_container), ignore_index=True)
        # combined_csv = [pd.concat(chunk,ignore_index=True) for chunk in chunk_container]

        # joining files using read_csv withou chunk
        # combined_csv = pd.concat(map(pd.read_csv(sep='\t'), files), ignore_index=True)
        combined_csv = pd.concat([pd.read_csv(f, sep='\t') for f in files], ignore_index=True)
        combined_csv.to_csv('../Dataset/' + ano + '/dataset_escola_filtered_parcial.csv', sep='\t', encoding='utf-8',
                            index=False)

        del (combined_csv)
        gc.collect()

    def runDimensionReductionTS_ESCOLA(self,url):
        try:
            # dataframe = dd.read_csv(url, sep='|',  dtype='object')
            # dataset_reduce = pd.read_csv(url, sep='|', usecols=lambda x: x not in drop_columns )
            dataset = pd.read_csv(url, delimiter=',', encoding="utf-8", )

            columns = dataset.columns;
            for c in columns:
                print(c)
            # columns_drop_1 = filter(lambda name: name.find("NIVEL_") != -1, columns);
            # columns_drop_2 = filter(lambda name: name.find("TAXA_") != -1, columns);
            # columns_drop_3 = filter(lambda name: name.find("PC_") != -1, columns);

            columns_selected = list(filter(lambda name: name.find("MEDIA") != -1, columns));
            columns_selected.append('ID_ESCOLA')
            columns_selected.append('ID_SAEB')
            columns_selected.append('ID_REGIAO')
            columns_selected.append('ID_UF')

            print(columns_selected)
            print(dataset.columns.difference(columns_selected))

            dataset_reduce = dataset.drop(dataset.columns.difference(columns_selected), axis=1)

            dataset_reduce.rename(columns={'ID_ESCOLA': 'CO_ENTIDADE'}, inplace=True)

            print("Dimensionality reduced from {} to {}.".format(dataset.shape, dataset_reduce.shape))
            # dataset_reduce.update(dataset_reduce[['NO_ENTIDADE']].applymap('"{}"'.format))

            print(dataset_reduce.columns)
            return dataset_reduce
        except:
            print("Oops!", sys.exc_info(), "occurred.");

    def merge(self,censo, dataset_escola_saeb, ano):
        print('Merge Method')
        dataset_censo_escola_matricula = pd.read_csv(censo, delimiter='\t', encoding="utf-8", )
        print(dataset_censo_escola_matricula.shape)
        print("Check duplicidade: ", dataset_censo_escola_matricula['CO_ENTIDADE'].duplicated().any())
        df_result = pd.merge(dataset_escola_saeb, dataset_censo_escola_matricula, on='CO_ENTIDADE')
        print(df_result.shape)
        # Check duplicidade
        print("Check duplicidade: ", df_result['CO_ENTIDADE'].duplicated().any())
        df_result.to_csv('../Dataset/' + str(ano) + '/inep_saeb_merge_' + str(ano) + '.csv', sep='\t', encoding='utf-8',index=False)
        del (df_result)
        gc.collect()

    def calculateMissingValues(self, nyc_data_raw):
        print("Detect missing values.")
        start = time.time()
        missing_values = nyc_data_raw.isnull().sum()
        percent_missing = ((missing_values / nyc_data_raw.index.size) * 100).compute()
        print(percent_missing)
        end = time.time()
        print("Read csv with Dask: ", (end - start), "sec")

if __name__ == '__main__':

        obj = ScriptPipelineCensoSAEB()
        regioes = ['sul', 'co', 'nordeste', 'norte']
        ano = str(2019)
        # for region in regioes:
        #
        #     file = 'MATRICULA_' + region.upper()
        #     file_out_split = 'MATRICULA_' + region.upper() + '_REDUZIDO_'
        #     file_out_split_remove = 'MATRICULA_' + region.upper() + '_REDUZIDO_REMOVE_'
        #     file_out_merge = 'matricula_reduzido_' + region
        #     start_global = time.time()
        #
        #     # Passo 1: Quebrar os arquivos
        #     # Arquivos da base de dados
        #     print("Inicia Split File")
        #     start = time.time()
        #     obj.splitFileWithDask(file, ano)
        #     end = time.time()
        #     print("Split csv: ", (end - start), "sec")
        #     print("Finaliza Split File")
        #
        #     # Passo 2: Reduz numero de dimensões
        #     print("Inicia Redução de Dimensão")
        #     start = time.time()
        #     files = os.path.join("../Dataset/" + str(ano), file + "_file_*.csv")
        #     # list of merged files returned
        #     files = glob.glob(files)
        #     for f in files:
        #         n = re.findall(r'\d+', f)[1]
        #         obj.runDimensionReduction(f, file_out_split + str(n), ano)
        #
        #     end = time.time()
        #     print("Finaliza redução em : ", (end - start), "sec")
        #
        #     # Passo 3: remove index
        #     print("Inicia Remocao de indice")
        #     start = time.time()
        #     files = os.path.join("../Dataset/" + str(ano), file_out_split + "*.csv")
        #     files = glob.glob(files)
        #     for f in files:
        #         n = re.findall(r'\d+', f)[1]
        #         obj.removeColumnWithDask(f, '\t', file_out_split_remove + str(n), ano)
        #     end = time.time()
        #     print("Finaliza remoção indice em : ", (end - start), "sec")
        #
        #     # Passo 4: Realiza Merge dos arquivos - Se caso separou os arquivos
        #     print("Inicia Merge de Linhas")
        #     start = time.time()
        #     obj.concatCSVbyRows(file_out_split_remove, file_out_merge, ano)
        #     end = time.time()
        #     print("Finaliza Merge em : ", (end - start), "sec")
        #     end_global = time.time()
        #     print("Total Time: ", (end_global - start_global), "sec")

########################################################################################################################
        # #Passo 5: Concatenar os arquivos de cada regiao
        #
        # print("Inicia processo concatenao das regioes")
        # ano = str(2019)
        # start = time.time()
        # # loadMatriculaWithDask('../Dataset/2017/matricula_reduzido_nordeste.csv',)
        # obj.concatCSVWithDask('matricula_reduzido_', 'matricula_reduzido_all_' + ano, ano)
        # end = time.time()
        # print("Total time:", (end - start), "sec")

        # #Passo 6: Reduzir a dimensão do dataset SAEB_ESCOLA
        # start = time.time()
        # print("Inicia processo Redução SAEB")
        # url_csv = '../Dataset/' + ano + '/ESCOLAS.CSV'
        # obj.runDimensionReductionSAEB(url_csv, 'escola_update.csv', ano)
        # end = time.time()
        # print("Running process of Reducao: ", (end - start), "sec")
        #
        # #Script Merge Censo
        # # Passo 7: Split File
        # start = time.time()
        # print("Inicia processo Split File")
        # obj.splitFileWithDaskSAEB(ano)
        # end = time.time()
        # print("Running process of split: ", (end - start), "sec")
        #
        # #Passo 8: Calculator Indicator
        # start = time.time()
        # print("Inicia processo Calculator")
        # for i in range(35):
        #     indice = "%.2d" % i
        #     dataset_matricula, dataset_escola = obj.loadData(indice, ano)
        #     obj.calculateIndicators(dataset_escola, dataset_matricula, ano, indice)
        #
        # end = time.time()
        # print("Running process of Calculator: ", (end - start), "sec")
        #
        # # 9°Passo: Concat Files
        # start = time.time()
        # print("Inicia processo Concat")
        # obj.concatCSV(ano)
        # end = time.time()
        # print("Running process of Concat: ", (end - start), "sec")

########################################################################################################################
        # 10° Passo: Novo Agrupamento e novo arquivo
        # start = time.time()
        # print("Inicia processo Agrupamento")
        # dataset_escola_matricula = pd.read_csv('../Dataset/' + ano + '/dataset_escola_filtered_parcial.csv', sep='\t', )
        # colunas = dataset_escola_matricula.columns
        # print(dataset_escola_matricula.shape)
        #
        # estatistica_alunos_por_escola = dataset_escola_matricula.groupby(['CO_ENTIDADE'], as_index=False).agg(["sum"],split_out=8)
        # del (dataset_escola_matricula)
        # gc.collect()
        #
        # estatistica_alunos_por_escola = estatistica_alunos_por_escola.reset_index()
        # # Padronizar colunas devido agrupamento
        # estatistica_alunos_por_escola.columns = colunas
        # # Check duplicidade
        # print("Check duplicidade: ", estatistica_alunos_por_escola['CO_ENTIDADE'].duplicated().any())
        #
        # estatistica_alunos_por_escola.to_csv('../Dataset/' + ano + '/dataset_escola_filtered.csv', sep='\t',encoding='utf-8', index=False)
        # end = time.time()
        # print("Running process of Novo Agrupamento: ", (end - start), "sec")
        # del (estatistica_alunos_por_escola)
        # gc.collect()

########################################################################################################################
        #Passo 11:
        start = time.time()
        url_csv = '../Dataset/' + ano + '/TS_ESCOLA.csv'
        dataset_escola_saeb = obj.runDimensionReductionTS_ESCOLA(url_csv)
        censo = '../Dataset/' + ano + '/dataset_escola_filtered.csv'
        obj.merge(censo, dataset_escola_saeb, ano)
        end = time.time()
        print("Total Time: ", (end - start), "sec")