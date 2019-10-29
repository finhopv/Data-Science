from minisom import MiniSom
import pandas as pd

baseEntra = pd.read_csv('breastCancer.csv')
baseSai = pd.read_csv('saidasBreast.csv')
X= baseEntra.iloc[:,1:30].values
y= baseSai.iloc[:,0].values # esta é classe ou saída em outras hipoteses

from sklearn.preprocessing import MinMaxScaler
normalizador= MinMaxScaler(feature_range=(0,1))
X= normalizador.fit_transform(X)

# para encontrar a quantidade de linhas e colunas do mapa podemos
# multiplicar a raiz quadrada da quantidade de registros por 5

som= MiniSom(x=10, y=10, input_len=29, sigma=1.0, learning_rate=0.5,random_seed=2)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=1000)
som._weights
som._activation_map
q=som.activation_response(X)

from pylab import pcolor, colorbar,plot
pcolor(som.distance_map().T)
colorbar()


markers=['o','s']
color=['r','g']

for i, x in enumerate(X):
    w= som.winner(x)
    plot(w[0]+0.5, w[1] +0.5, markers[y[i]],
         markerfacecolor='None', markersize=10,
         markeredgecolor=color[y[i]], markeredgewidth=2)