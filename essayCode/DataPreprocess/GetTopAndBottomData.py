import csv

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
def findGoodAndBad(filePath):
    chunksize = 10 ** 6
    dict1 = dict()
    Top15Medallion = []
    Top15Income = []
    Bottom15Income = []
    Bottom15Medallion = []
    for chunk in pd.read_csv(filePath, chunksize=chunksize):
        chunkList = [[]]
        chunkList = chunk.values.tolist()

        for line in chunkList:
            if line[0] in dict1:
                dict1[line[0]]=dict1[line[0]]+line[10]
            else:
                dict1[line[0]]=line[10]
        items = dict1.items()
        m = sorted(items, key=(lambda x: x[1]))
        LengthTenPercent=len(m)/10#强转整数
        TenPercent=int(LengthTenPercent)


        for i in range(len(m)-1,int(TenPercent*8.5),-1):
            Top15Medallion.append(m[i][0])
            Top15Income.append(m[i][1])
        for i in range(int(TenPercent*1.5)):
            Bottom15Medallion.append(m[i][0])
            Bottom15Income.append(m[i][1])
    fileObject = open("E:/data/ExprimentField/TaxiDataWithFare/jan/jan1/Top15Medallion.csv", 'w')
    for ip in Top15Medallion:
        fileObject.write(str(ip))
        fileObject.write('\n')
    fileObject.close()
    fileObject = open("E:/data/ExprimentField/TaxiDataWithFare/jan/jan1/Top15Income.csv", 'w')
    for ip in Top15Income:
        fileObject.write(str(ip))
        fileObject.write('\n')
    fileObject.close()
    fileObject = open("E:/data/ExprimentField/TaxiDataWithFare/jan/jan1/Bottom15Medallion.csv", 'w')
    for ip in Bottom15Medallion:
        fileObject.write(str(ip))
        fileObject.write('\n')
    fileObject.close()
    fileObject = open("E:/data/ExprimentField/TaxiDataWithFare/jan/jan1/Bottom15Income.csv", 'w')
    for ip in Bottom15Income:
        fileObject.write(str(ip))
        fileObject.write('\n')
    fileObject.close()
    # with open("E:/data/ExprimentField/TaxiDataWithFare/jan/jan1/Bottom15Medallion.csv", "a", newline="") as csvfile:
    #     writer = csv.writer(csvfile)
    #     for row in Bottom15Medallion:
    #         writer.writerow(row)
    # with open("E:/data/ExprimentField/TaxiDataWithFare/jan/jan1/Bottom15Income.csv", "a", newline="") as csvfile:
    #     writer = csv.writer(csvfile)
    #     for row in Bottom15Income:
    #         writer.writerow(row)
    # with open("E:/data/ExprimentField/TaxiDataWithFare/jan/jan1/Bottom15Medallion.csv", "a", newline="") as csvfile:
    #     writer = csv.writer(csvfile)
    #     for row in Bottom15Medallion:
    #         writer.writerow(row)
    # with open("E:/data/ExprimentField/TaxiDataWithFare/jan/jan1/Bottom15Income.csv", "a", newline="") as csvfile:
    #     writer = csv.writer(csvfile)
    #     for row in Bottom15Income:
    #         writer.writerow(row)
    return Top15Medallion,Bottom15Medallion

def filterDataBasedOnGoodAndBad(Top15Medallion,Bottom15Medallion,taxidata):
    Top15Data = []
    Bottom15Data = []
    chunksize = 10 ** 6
    for chunk in pd.read_csv(taxidata, chunksize=chunksize):

        chunkList = chunk.values.tolist()

        for line in chunkList:
            if line[0] in Top15Medallion:
                Top15Data.append(line)
            if line[0] in Bottom15Medallion:
                Bottom15Data.append(line)
    with open("E:/data/ExprimentField/TaxiDataWithFare/jan/jan1/Top15Data.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
            # writer.writerow(
            #     [medallion, hack_license, vendor_id, pickup_datetime, payment_type, fare_amount, surcharge, mta_tax, tip_amount, tolls_amount, total_amount])
        for row in Top15Data:
            writer.writerow(row)
    with open("E:/data/ExprimentField/TaxiDataWithFare/jan/jan1/Bottom15Data.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for row in Bottom15Data:
            writer.writerow(row)


FareData = "E:/data/ExprimentField/TimeDivideForCalculateFare/jan/timeSlot1.csv"
OriginalTaxiData = "E:/data/ExprimentField/test/jan/jan1/timeSlot1.csv"
Top15Medallion,Bottom15Medallion=findGoodAndBad(FareData)
filterDataBasedOnGoodAndBad(Top15Medallion,Bottom15Medallion,OriginalTaxiData)