import pandas as pd
import numpy as np
import time
import timeit

if __name__ == '__main__':
    df = pd.DataFrame()

    for x in range(1, 33):
        nomeV = "Q_Estudante_" + str(x)
        df[nomeV] = np.random.randint(1, 8, 10000)

    for x in range(1, 19):
        nomeV = "Q_Escola_" + str(x)
        df[nomeV] = np.random.randint(1, 8, 10000)
        # print(nomeV)

    df['score_estudante'] = 0
    df['score_escola'] = 0
    # df

    mapping = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', }

    # dados = dados.replace({'Aprobo': mapping, 'Reprobo': mapping, 'SienNota': mapping})

    df = df.applymap(lambda s: mapping.get(s) if s in mapping else s)

    mapping = {'a': 3, 'b': 2, 'c': 1, 'd': 0, 'e': -1, 'f': -2, 'g': -3}

    # dados = dados.replace({'Aprobo': mapping, 'Reprobo': mapping, 'SienNota': mapping})

    df = df.applymap(lambda s: mapping.get(s) if s in mapping else s)

    nRows = 0
    inicio = timeit.default_timer()
    for index, row in df.iterrows():
        score = 0
        nRows = nRows + 1
        if ((nRows % 100000) == 0):
            fimP = timeit.default_timer()
            tempoP = (fimP - inicio) / 60
            # print ('duracao: %f' % tempoP)
            print("Linhas Processadas: ", nRows, " Duração: ", tempoP)

        for x in range(1, 33):
            nomeV = "Q_Estudante_" + str(x)
            valor = row[nomeV]
            if (nomeV != "Q32"):
                score = score + valor

            elif ():
                score = score - valor
        # print(score)
        df.loc[index, 'score_estudante'] = score

    fim = timeit.default_timer()
    tempo = (fim - inicio) / 60
    print('Duracao Total: %f' % tempo)

    nRows = 0
    inicio = timeit.default_timer()
    for index, row in df.iterrows():
        score = 0
        nRows = nRows + 1
        if ((nRows % 100000) == 0):
            fimP = timeit.default_timer()
            tempoP = (fimP - inicio) / 60
            # print ('duracao: %f' % tempoP)
            print("Linhas Processadas: ", nRows, " Duração: ", tempoP)

        for x in range(1, 19):
            nomeV = "Q_Escola_" + str(x)
            valor = row[nomeV]
            if (nomeV != "Q32"):
                score = score + valor

            elif ():
                score = score - valor
        # print(score)
        df.loc[index, 'score_escola'] = score

    fim = timeit.default_timer()
    tempo = (fim - inicio) / 60
    print('Duracao Total: %f' % tempo)

    df['score_escola'].mean()
    df['score_estudante'].mean()
    df['score_total'] = df['score_estudante'] + df['score_escola']
    df['score_total'].mean()

    df['Risco_Escola'] = np.where(df['score_escola'] <= df['score_escola'].mean(), 'S', 'N')
    df['Risco_Estudante'] = np.where(df['score_estudante'] <= df['score_estudante'].mean(), 'S', 'N')
    df['Risco_Geral'] = np.where(df['score_total'] <= df['score_total'].mean(), 'S', 'N')

    print(df)