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

#найдем оптимальное количество кластеров
# elbow method
from sklearn.cluster import KMeans

# создадим кучу кластеров и оуеним метрику
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i, init='k-means++', random_state=42)# избегаем ошибки рандомной инициализации
    result=kmeans.fit(X)#тренируем модель
    wcss.append(kmeans.inertia_)# метрика wcss

plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('number  of clusters')
plt.ylabel('WCSS')
plt.show()


# подготовим модель
kmeans=KMeans(n_clusters=5, init='k-means++', random_state=42)# избегаем ошибки рандомной инициализации
result=kmeans.fit_predict(X)#тренируем модель и генерируем зависимую переменную
colors=['red','green','blue','cyan','magenta']
# визуализация
for touple_index in set(result):
    plt.scatter(X[result==touple_index,0],X[result==touple_index,1],s=50,c=colors[touple_index],label='Cluster {}'.format(touple_index))
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='yellow',label='centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual income')
plt.title('spending score (1-100)')
plt.legend()
plt.show()

