from minisom import MiniSom
import pandas as pd

base = pd.read_csv('wines.csv')
X= base.iloc[:,1:14].values
y= base.iloc[:,0].values # esta é classe ou saída em outras hipoteses

from sklearn.preprocessing import MinMaxScaler
normalizador= MinMaxScaler(feature_range=(0,1))
X= normalizador.fit_transform(X)

# para encontrar a quantidade de linhas e colunas do mapa podemos
# multiplicar a raiz quadrada da quantidade de registros por 5

som= MiniSom(x=8, y=8, input_len=13, sigma=1.0, learning_rate=0.5,random_seed=2)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)
som._weights
som._activation_map
q=som.activation_response(X)

from pylab import pcolor, colorbar,plot
pcolor(som.distance_map().T)
colorbar()


markers=['o','s','D']
color=['r','g','b']
y[y==1]=0
y[y==2]=1
y[y==3]=2

for i, x in enumerate(X):
    w= som.winner(x)
    plot(w[0]+0.5, w[1] +0.5, markers[y[i]],
         markerfacecolor='None', markersize=10,
         markeredgecolor=color[y[i]], markeredgewidth=2)