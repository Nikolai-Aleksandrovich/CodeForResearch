


import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import metrics

from geopy.distance import great_circle
from shapely.geometry import MultiPoint
import matplotlib.pyplot as plt, time

def get_centermost_point(cluster):
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
    return tuple(centermost_point)

df = pd.read_csv("E:/data/ExprimentField/manhatan/Poi_NYC_Manhatan.csv", encoding='utf-8')
df.head()
coords = df[['latitude', 'longitude']].values
kms_per_radian = 6371.0088
epsilon = 2/ kms_per_radian
start_time = time.time()
db = DBSCAN(eps=epsilon, min_samples=10, algorithm='ball_tree', metric="haversine").fit(np.radians(coords))
cluster_labels = db.labels_
num_clusters = len(set(cluster_labels))
# turn the clusters in to a pandas series, where each element is a cluster of points
clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])


message = 'Clustered {:,} points down to {:,} clusters, for {:.1f}% compression in {:,.2f} seconds'
print(message.format(len(df), num_clusters, 100*(1 - float(num_clusters) / len(df)), time.time()-start_time))
print('Silhouette coefficient: {:0.03f}'.format(metrics.silhouette_score(coords, cluster_labels)))
print('Number of clusters: {}'.format(num_clusters))
clusters = clusters[0:-1]

cluster_list = list(set(cluster_labels))
cluster_list = cluster_list[0:-1]
num_clusters = len(cluster_list)
print(num_clusters)

centermost_points = clusters.map(get_centermost_point)

lats, lons = zip(*centermost_points)

rep_points = pd.DataFrame({'lon':lons, 'lat':lats})
rep_points.tail()

rs = rep_points.apply(lambda row: df[(df['latitude']==row['lat']) & (df['longitude']==row['lon'])].iloc[0], axis=1)
rs.to_csv('E:/data/ExprimentField/manhatan/Poi_NYC_Manhatan_DBSCAN.csv', encoding='utf-8')
rs.tail()


fig, ax = plt.subplots(figsize=[10, 6])


rs_scatter = ax.scatter(rs['longitude'], rs['latitude'], c='#99cc99', edgecolor='None', alpha=0.7, s=120)
df_scatter = ax.scatter(df['longitude'], df['latitude'], c='k', alpha=0.9, s=3)
ax.set_title('Full data set vs DBSCAN reduced set')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.legend([df_scatter, rs_scatter], ['Full set', 'Reduced set'], loc='upper right')
plt.show()

