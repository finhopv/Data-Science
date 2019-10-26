from keras.models import Sequential
from keras.layers import Dense, Dropout,LSTM
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

base= pd.read_csv('poluicao.csv')
base= base.dropna()
base=base.drop(['year','No','month','day','hour','cbwd'], axis=1)

base_treinamento=base.iloc[:, 1:7].values # pegando valores de todas as linhas e colunas de 1 a 7

# Busca dos valores que será feita a previsão, ou seja o primeiro atributo pm2.5
poluicao = base.iloc[:, 0].values

# agora normalizando os dados

normalizador= MinMaxScaler(feature_range=(0,1))
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)

normalizador_previsao= MinMaxScaler(feature_range=(0,1))
normalizador_previsao.fit_transform(base_treinamento)

poluicao = poluicao.reshape(-1, 1)
poluicao_normalizado = normalizador.fit_transform(poluicao)


previsores=[]
poluicao_real = []
for i in range(10,41757):
    previsores.append(base_treinamento_normalizada[i-10:i, 0:6]) # até o 6 pra todas as colunas, não apenas  primeira
    poluicao_real.append(poluicao_normalizado[i, 0])
    
previsores, polluicao_real= np.array(previsores), np.array(polluicao_real)  

regressor= Sequential()
regressor.add(LSTM(units=100, return_sequences= True, input_shape=(previsores.shape[1],6))) # só passa o input_shape na primeira camada
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=50, return_sequences= True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences= True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences= True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences= True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))


regressor.add(Dense(units=1, activation='sigmoid'))

regressor.compile(optimizer='adam', loss='mean_squared_error',
                  metrics=['mean_absolute_error'])
  
regressor.fit(previsores, poluicao_real, epochs = 100, batch_size = 64)

# Neste exemplo não utilizaremos uma base de dados específica para teste, ou seja, 
# faremos as previsões diretamente na base de dados de treinamento
previsoes = regressor.predict(previsores)
previsoes = normalizador.inverse_transform(previsoes)



# Verificação da média nos resultados das previsões e nos resultados reais
previsoes.mean()
poluicao.mean()

# Geração do gráfico. Será gerado um gráfico de barras porque temos muitos registros
plt.plot(poluicao, color = 'red', label = 'Poluição real')
plt.plot(previsoes, color = 'blue', label = 'Previsões')
plt.title('Previsão poluição')
plt.xlabel('Horas')
plt.ylabel('Valor poluição')
plt.legend()
plt.show()