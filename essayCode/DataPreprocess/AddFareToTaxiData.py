import csv
from datetime import time
# coding:utf-8
import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
import scipy.spatial
import math
import time
import datetime
from math import radians, cos, sin, asin, sqrt
import dask.dataframe as dd
import pandas as pd
from collections import namedtuple


def addFareToTaxiData(TaxiData,FareData,Target):
    start_time = time.time()
    mci = pd.read_csv(TaxiData)    # 文件1
    r = pd.read_csv(FareData)  # 文件2

    f = open(Target,'w',encoding='utf-8',newline='')  # 写入文件，注意newline=''，没有会造成有空行
    res = csv.writer(f)

    # res.writerow(["RID","label","PTID"])	# 写入表头

    for i in range(0,mci.shape[0]):		# 依次遍历文件2
        d = mci.iloc[i]['medallion']			# 获取当前行的 RID
        e = mci.iloc[i]['pickup_datetime']
        for j in range(0,r.shape[0]):	# 在文件 1 中依次查找 RID
            if r.iloc[j]['medallion'] == d and r.iloc[j][' pickup_datetime'] == e:
                res.writerow([d,mci.iloc[i]['hack_license'],mci.iloc[i]['pickup_datetime'],mci.iloc[i][' dropoff_datetime'],mci.iloc[i][' passenger_count'],mci.iloc[i][' trip_time_in_secs'],mci.iloc[i][' trip_distance'],mci.iloc[i][' pickup_longitude'],mci.iloc[i][' pickup_latitude'],mci.iloc[i][' dropoff_longitude'],mci.iloc[i][' dropoff_latitude'],r.iloc[j][' total_amount']])	# 查找到，写入新文件中
                break	# 直接break，进入下一行查找

    f.close()  # 关闭文件，千万别忘了

    end_time = time.time()
    cost_time = (end_time - start_time)
    print('Total time spent on loading car data {:.5f} second.'.format(cost_time))

TaxiData = "E:/data/ExprimentField/test/jan/jan1/timeSlot1.csv"
FareData = "E:/data/ExprimentField/TimeDivideForFare/jan/timeSlot1.csv"

target0 = "E:/data/ExprimentField/TaxiDataWithFare/jan/timeSlot0.csv"
target1 = "E:/data/ExprimentField/TaxiDataWithFare/jan/timeSlot1.csv"
target2 = "E:/data/ExprimentField/TaxiDataWithFare/jan/timeSlot2.csv"
target3 = "E:/data/ExprimentField/TaxiDataWithFare/jan/timeSlot3.csv"
target4 = "E:/data/ExprimentField/TaxiDataWithFare/jan/timeSlot4.csv"
target5 = "E:/data/ExprimentField/TaxiDataWithFare/jan/timeSlot5.csv"
target6 = "E:/data/ExprimentField/TaxiDataWithFare/jan/timeSlot6.csv"

addFareToTaxiData(TaxiData,FareData,target1)