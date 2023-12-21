import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta

def getData():
    # Lendo o dataframe
    url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.24363/dados?formato=json"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data)
    df['data'] = pd.to_datetime(df['data'], dayfirst=True)
    df.set_index('data', inplace=True)
    return df

def predict(df, periodos=100):
    # Configurando a predicao dos próximos periodos
    x = np.arange(len(df)).reshape(-1, 1)
    y = df['valor'].values
    model = LinearRegression()
    model.fit(x, y)
    predicao = model.predict(np.arange(len(df), len(df) + periodos).reshape(-1, 1))

    # Calculando os limites superiores e inferiores
    mse = mean_squared_error(y, model.predict(x))
    n = len(x)
    conf_interval = 1.96 * np.sqrt(mse)  # Intervalo de confianca de 95%

    upper_bound = predicao + conf_interval
    lower_bound = predicao - conf_interval

    return predicao, upper_bound, lower_bound

def temporalSeries(df, periodos = 100):
    predicao, upper_bound, lower_bound = predict(df, periodos)
    # Configurando a data da série temporal de 5 anos
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5 * 365)
    df = df[start_date:end_date]
    df.loc[:, 'valor'] = pd.to_numeric(df['valor'], errors='coerce', downcast='integer') # arrumando os valores

    # Juntando a predicao com a serie temporal
    df_predicao = pd.DataFrame(data={'data': pd.date_range(start=end_date, periods=periodos), 'valor': predicao,
                                     'upper_bound': upper_bound, 'lower_bound': lower_bound})
    df_predicao.set_index('data', inplace=True)
    df = pd.concat([df, df_predicao])

    return df, df_predicao

def showGraphic(df, periodos=100):
    df, df_predicao= temporalSeries(df, periodos)

    # Apresentando o gráfico
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['valor'], label='Histórico')  # valores sa série temporal de 5 anos
    plt.plot(df_predicao.index, df_predicao['valor'], color='pink', label='Predição')  # predição
    plt.plot(df_predicao.index, df_predicao['upper_bound'], color='red', label='Limite Superior (95%)')  # predição em 95% pra cima
    plt.plot(df_predicao.index, df_predicao['lower_bound'], color='purple', label='Limite Inferior (95%)')  # predição em 95% pra baixo
    plt.title('Série Temporal Histórica com Predição dos Próximos 6 Períodos')
    plt.xlabel('Ano')
    plt.ylabel('Índice de Confiança do Consumidor')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)  # linhas cinzas pq quis deixar parecido com o modelo

    plt.show()

# Funcoes sendo utilizadas
df = getData()
mostrar_grafico = showGraphic(df,100 )




