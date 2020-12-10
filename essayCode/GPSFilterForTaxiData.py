import csv
import time
import numpy as np
import pandas as pd
from shapely.geometry import mapping, shape, Polygon, MultiPoint


def getAttrFromCSVtoGraph(filePath):
    start_time = time.time()
    print("--------Add Attr--------")
    chunksize = 10 ** 6
    chunkNum = 0
    for chunk in pd.read_csv(filePath, chunksize=chunksize):
        chunkNum+=1
        validateNumber=0
        inValidateNumber=0
        chunkList = [[]]
        chunkList = chunk.values.tolist()
        EdgesCount = 0
        itemsCount = 0
        edgesList = []
        for line in chunkList:
            itemsCount = itemsCount + 1
            if itemsCount != 1:
                # lines = line.strip().split(",")
                if len(line) == 14:
                    # 判断经纬度是否为 00 0 0 如 果为0就说明这条记录无效
                    if (line[12]) == "":
                        continue
                    if abs(float(line[10])) < 1e-6 and abs(float(line[11])) <= 1e-6 and abs(
                            float(line[12])) <= 1e-6 and abs(float(line[13])) <= 1e-6:
                        1
                    else:
                        EdgesCount = EdgesCount + 1
                        edgesList.append(line)
        validateList=[]
        for line in edgesList:
            if judgerOfGps(line[10], line[11]) & judgerOfGps(line[12], line[13]):
                validateList.append(line)
                validateNumber=validateNumber+1
            else:
                inValidateNumber=inValidateNumber+1
        with open("E:/data/ExprimentField/manhatan/trip_data1_Manhatan.csv", "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # writer.writerow(["medallion", "hack_license", "vendor_id", "rate_code", "store_and_fwd_flag", "pickup_datetime", " dropoff_datetime", " passenger_count"," trip_time_in_secs"," trip_distance"," pickup_longitude"," pickup_latitude"," dropoff_longitude"," dropoff_latitude"])
            for row in validateList:
                writer.writerow(row)
        end_time = time.time()
        cost_time = (end_time - start_time)
        print("current chunk is chunk NO." + str(chunkNum))
        print("validate",validateNumber)
        print("validate list size",len(validateList))
        print("invalidate",inValidateNumber)
        print('Total time spent on loading car data {:.5f} second.'.format(cost_time))
        print("--------Done!--------")
def judgerOfGps(longitude,latitude):
    global b
    top_left = [-73.939961, 40.860598]
    bottom_left = [-74.035471, 40.709885]
    bottom_right = [-73.984962, 40.694339]
    top_right = [-73.926392, 40.801619]
    coordinates = [(longitude, latitude)]
    coordinates_shapely = MultiPoint(coordinates)
    polyNY_shapely = Polygon([(top_left), (bottom_left), (bottom_right), (top_right)])
    for pp in range(len(list(coordinates_shapely))):
        b=(polyNY_shapely.contains(coordinates_shapely[pp]))
    return b

filePath = "E:/data/ExprimentField/trip_data_1.csv"
getAttrFromCSVtoGraph(filePath)