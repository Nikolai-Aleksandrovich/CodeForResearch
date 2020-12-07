# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from sklearn import metrics
from math import radians
from math import tan, atan, acos, sin, cos, asin, sqrt
from scipy.spatial.distance import pdist, squareform

sns.set()


def haversine(lonlat1, lonlat2):
    lat1, lon1 = lonlat1
    lat2, lon2 = lonlat2
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r * 1000


df = pd.read_csv('00-首页数据.csv')

df = df[['lat_Amap', 'lng_Amap']].dropna(axis=0, how='all')
# df['lon_lat'] = df.apply(lambda x: [x['lng_Amap'], x['lat_Amap']], axis=1)
# df = df['lon_lat'].to_frame()
# data = np.array(data)
# plt.figure(figsize=(10, 10))
# plt.scatter(df['lat_Amap'], df['lng_Amap'])
distance_matrix = squareform(pdist(df, (lambda u, v: haversine(u, v))))
db = DBSCAN(eps=500, min_samples=10, metric='precomputed').fit_predict(distance_matrix)

'''
db = DBSCAN(eps=0.038, min_samples=3).fit(data)
'''
labels = db
raito = len(labels[labels[:] == -1]) / len(labels)  # 计算噪声点个数占总数的比例
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # 获取分簇的数目
# score = metrics.silhouette_score(distance_matrix, labels)
df['label'] = labels
sns.lmplot('lat_Amap', 'lng_Amap', df, hue='label', fit_reg=False)

''' 
df['label'] = labels
sns.lmplot('lat_Amap', 'lng_Amap', df, hue='label', fit_reg=False)
'''
map_all = folium.Map(location=[31.574729, 120.301663], zoom_start=12,
                     tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=7&x={x}&y={y}&z={z}',
                     attr='default')

# colors = ['#DC143C', '#FFB6C1', '#DB7093', '#C71585', '#8B008B', '#4B0082', '#7B68EE',
#          '#0000FF', '#B0C4DE', '#708090', '#00BFFF', '#5F9EA0', '#00FFFF', '#7FFFAA',
#          '#008000', '#FFFF00', '#808000', '#FFD700', '#FFA500', '#FF6347', '#000000']

colors = ['#DC143C', '#FFB6C1', '#DB7093', '#C71585', '#8B008B', '#4B0082', '#7B68EE',
          '#0000FF', '#B0C4DE', '#708090', '#00BFFF', '#5F9EA0', '#00FFFF', '#7FFFAA',
          '#008000', '#FFFF00', '#808000', '#FFD700', '#FFA500', '#FF6347', '#DC143C',
          '#FFB6C1', '#DB7093', '#C71585', '#8B008B', '#4B0082', '#7B68EE',
          '#0000FF', '#B0C4DE', '#708090', '#00BFFF', '#5F9EA0', '#00FFFF', '#7FFFAA',
          '#008000', '#FFFF00', '#808000', '#FFD700', '#FFA500', '#FF6347',
          '#DC143C', '#FFB6C1', '#DB7093', '#C71585', '#8B008B', '#4B0082', '#7B68EE',
          '#0000FF', '#B0C4DE', '#708090', '#00BFFF', '#5F9EA0', '#00FFFF', '#7FFFAA',
          '#008000', '#FFFF00', '#808000', '#FFD700', '#FFA500', '#FF6347',
          '#DC143C', '#FFB6C1', '#DB7093', '#C71585', '#8B008B', '#4B0082', '#7B68EE',
          '#0000FF', '#B0C4DE', '#708090', '#00BFFF', '#5F9EA0', '#00FFFF', '#7FFFAA',
          '#008000', '#FFFF00', '#808000', '#FFD700', '#FFA500', '#FF6347', '#DC143C',
          '#FFB6C1', '#DB7093', '#C71585', '#8B008B', '#4B0082', '#7B68EE',
          '#0000FF', '#B0C4DE', '#708090', '#00BFFF', '#5F9EA0', '#00FFFF', '#7FFFAA',
          '#008000', '#FFFF00', '#808000', '#FFD700', '#FFA500', '#FF6347',
          '#DC143C', '#FFB6C1', '#DB7093', '#C71585', '#8B008B', '#4B0082', '#7B68EE',
          '#0000FF', '#B0C4DE', '#708090', '#00BFFF', '#5F9EA0', '#00FFFF', '#7FFFAA',
          '#008000', '#FFFF00', '#808000', '#FFD700', '#FFA500', '#FF6347', '#000000']

for i in range(len(df)):
    if labels[i] == -1:
        continue
    else:
        folium.CircleMarker(location=[df.iloc[i, 0], df.iloc[i, 1]],
                            radius=4, popup='popup',
                            color=colors[labels[i]], fill=True,
                            fill_color=colors[labels[i]]).add_to(map_all)

map_all.save('all_cluster.html')