
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import metrics

from geopy.distance import great_circle
from shapely.geometry import MultiPoint
import matplotlib.pyplot as plt, time
df = pd.read_csv("E:/data/ExprimentField/manhatan/Poi_NYC_Manhatan.csv")
coords = df[['latitude', 'longitude']].values
for row in df:
    print(row)
rs = coords[(lambda row: df[(df['latitude']==row['latitude']) & (df['longitude']==row['longitude'])].iloc[0], axis=1)]
# print(rs)