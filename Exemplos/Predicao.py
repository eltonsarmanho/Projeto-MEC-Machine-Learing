import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

def substring(palavra, inicio=0, parada=0):
    if parada == len(palavra):
        inicio += 1
        parada = inicio
    if inicio >= len(palavra):
        return
    i = 0
    while inicio + i <= parada:
        print(palavra[inicio+i], end='')
        i += 1
    print()
    return substring(palavra, inicio, parada+1)

if __name__ == '__main__':
    substring('dog')
    #Criar dados
    nsample = 50
    sig = 0.25
    x1 = np.linspace(0, 20, nsample)
    X = np.column_stack((x1, np.sin(x1), (x1 - 5) ** 2))
    X = sm.add_constant(X)
    beta = [5., 0.5, 0.5, -0.02]
    y_true = np.dot(X, beta)
    y = y_true + sig * np.random.normal(size=nsample)

    #Treinar Modelo
    olsmod = sm.OLS(y, X)
    olsres = olsmod.fit()

    #Predizer

    ypred = olsres.predict(X)
    print(ypred)

    #Criar uma nova amostra exploratória
    x1n = np.linspace(20.5, 25, 10)
    Xnew = np.column_stack((x1n, np.sin(x1n), (x1n - 5) ** 2))
    Xnew = sm.add_constant(Xnew)
    ynewpred = olsres.predict(Xnew)  # predict out of sample
    print(ynewpred)


    fig, ax = plt.subplots()
    ax.plot(x1, y, 'o', label="Dados Observados")
    ax.plot(x1, y_true, 'b-', label="Tendência dos dados")
    ax.plot(np.hstack((x1, x1n)), np.hstack((ypred, ynewpred)), 'r', label="Modelo de Predição")
    ax.legend(loc="best");
    plt.show()


