import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, 1:].values  # все кроме айдишника


#закодируем мцужчин и женщин
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
X[:,0]=le.fit_transform(X[:,0])

#переопределим зависимые переменные для визуализации
X = dataset.iloc[:, [3,4]].values  # это только для визуализации
#мы не создаем обучающую и тестовую выборку так как мы хотим создать зависимую переменную Y

# рисуем дендограмму
import scipy.cluster.hierarchy as sch
#это функция
#method='ward' значит обьеденять кластеры с наименьшим расстоянием по эвклиду
#linkage Выполняет иерархическую / агломеративную кластеризацию.
dendogram=sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('dendogram')
plt.xlabel('customers')
plt.ylabel('Euclidian distance')
plt.show()

from sklearn.cluster import AgglomerativeClustering
# подготовим модель
#affinity вид дистанции который применяем
#linkage то как мы обьеденяем кластеры
model=AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')# избегаем ошибки рандомной инициализации
result=model.fit_predict(X)#тренируем модель и генерируем зависимую переменную
colors=['red','green','blue','cyan','magenta']
# визуализация
for touple_index in set(result):
    plt.scatter(X[result==touple_index,0],X[result==touple_index,1],s=50,c=colors[touple_index],label='Cluster {}'.format(touple_index))
#plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1],s=100,c='yellow',label='centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual income')
plt.title('spending score (1-100)')
plt.legend()
plt.show()
