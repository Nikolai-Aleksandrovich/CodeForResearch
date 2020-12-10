import csv
import time
import numpy as np
import pandas as pd
from shapely.geometry import mapping, shape, Polygon, MultiPoint


def splitCSVtoTimeSlot(filePath):
    start_time = time.time()
    print("--------Add Attr--------")
    chunksize = 10 ** 6
    chunkNum = 0
    for chunk in pd.read_csv(filePath, chunksize=chunksize):
        chunkNum = chunkNum + 1
        validateNumber = 0
        inValidateNumber = 0
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
        timeSlot0 = []  # 22-7
        timeSlot1 = []  # 7-9
        timeSlot2 = []  # 9-12
        timeSlot3 = []  # 12-14
        timeSlot4 = []  # 14-17
        timeSlot5 = []  # 17-19
        timeSlot6 = []  # 19-22
        for line in edgesList:
            EmptyTime = line[6]
            pickUpTime = line[5]

            get_time = str(pickUpTime).rpartition(' ')[-1]
            # print(get_time)
            get_hour1 = str(get_time).rpartition(':')[0]
            get_hour = str(get_hour1).rpartition(':')[0]
            get_hour_num = int(get_hour)
            if (get_hour_num >= 22 or get_hour_num < 7):
                timeSlot0.append(line)
            elif (get_hour_num >= 7 and get_hour_num < 9):
                timeSlot1.append(line)
            elif (get_hour_num >= 9 and get_hour_num < 12):
                timeSlot2.append(line)
            elif (get_hour_num >= 12 and get_hour_num < 14):
                timeSlot3.append(line)
            elif (get_hour_num >= 14 and get_hour_num < 17):
                timeSlot4.append(line)
            elif (get_hour_num >= 17 and get_hour_num < 19):
                timeSlot5.append(line)
            else:
                timeSlot6.append(line)

        with open("E:/data/ExprimentField/timedivide/trip_data1/timeSlot0.csv", "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # writer.writerow(
            #     ["medallion", "hack_license", "vendor_id", "rate_code", "store_and_fwd_flag", "pickup_datetime",
            #      " dropoff_datetime", " passenger_count", " trip_time_in_secs", " trip_distance", " pickup_longitude",
            #      " pickup_latitude", " dropoff_longitude", " dropoff_latitude"])
            for row in timeSlot0:
                writer.writerow(row)
        with open("E:/data/ExprimentField/timedivide/trip_data1/timeSlot1.csv", "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # writer.writerow(
            #     ["medallion", "hack_license", "vendor_id", "rate_code", "store_and_fwd_flag", "pickup_datetime",
            #      " dropoff_datetime", " passenger_count", " trip_time_in_secs", " trip_distance", " pickup_longitude",
            #      " pickup_latitude", " dropoff_longitude", " dropoff_latitude"])
            for row in timeSlot1:
                writer.writerow(row)
        with open("E:/data/ExprimentField/timedivide/trip_data1/timeSlot2.csv", "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # writer.writerow(
            #     ["medallion", "hack_license", "vendor_id", "rate_code", "store_and_fwd_flag", "pickup_datetime",
            #      " dropoff_datetime", " passenger_count", " trip_time_in_secs", " trip_distance", " pickup_longitude",
            #      " pickup_latitude", " dropoff_longitude", " dropoff_latitude"])
            for row in timeSlot2:
                writer.writerow(row)
        with open("E:/data/ExprimentField/timedivide/trip_data1/timeSlot3.csv", "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # writer.writerow(
            #     ["medallion", "hack_license", "vendor_id", "rate_code", "store_and_fwd_flag", "pickup_datetime",
            #      " dropoff_datetime", " passenger_count", " trip_time_in_secs", " trip_distance", " pickup_longitude",
            #      " pickup_latitude", " dropoff_longitude", " dropoff_latitude"])
            for row in timeSlot3:
                writer.writerow(row)
        with open("E:/data/ExprimentField/timedivide/trip_data1/timeSlot4.csv", "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # writer.writerow(
            #     ["medallion", "hack_license", "vendor_id", "rate_code", "store_and_fwd_flag", "pickup_datetime",
            #      " dropoff_datetime", " passenger_count", " trip_time_in_secs", " trip_distance", " pickup_longitude",
            #      " pickup_latitude", " dropoff_longitude", " dropoff_latitude"])
            for row in timeSlot4:
                writer.writerow(row)
        with open("E:/data/ExprimentField/timedivide/trip_data1/timeSlot5.csv", "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # writer.writerow(
            #     ["medallion", "hack_license", "vendor_id", "rate_code", "store_and_fwd_flag", "pickup_datetime",
            #      " dropoff_datetime", " passenger_count", " trip_time_in_secs", " trip_distance", " pickup_longitude",
            #      " pickup_latitude", " dropoff_longitude", " dropoff_latitude"])
            for row in timeSlot5:
                writer.writerow(row)
        with open("E:/data/ExprimentField/timedivide/trip_data1/timeSlot6.csv", "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # writer.writerow(
            #     ["medallion", "hack_license", "vendor_id", "rate_code", "store_and_fwd_flag", "pickup_datetime",
            #      " dropoff_datetime", " passenger_count", " trip_time_in_secs", " trip_distance", " pickup_longitude",
            #      " pickup_latitude", " dropoff_longitude", " dropoff_latitude"])
            for row in timeSlot6:
                writer.writerow(row)

        end_time = time.time()
        cost_time = (end_time - start_time)
        totalSize = len(timeSlot0)+len(timeSlot1)+len(timeSlot2)+len(timeSlot3)+len(timeSlot4)+len(timeSlot5)+len(timeSlot6)
        print("current chunk is chunk NO." + str(chunkNum))
        print("pick up number of timeSlot 0" + "   "+str(len(timeSlot0)))
        print("pick up number of timeSlot 1" + "   "+str(len(timeSlot1)))
        print("pick up number of timeSlot 2" + "   "+str(len(timeSlot2)))
        print("pick up number of timeSlot 3" + "   "+str(len(timeSlot3)))
        print("pick up number of timeSlot 4" + "   "+str(len(timeSlot4)))
        print("pick up number of timeSlot 5" + "   "+str(len(timeSlot5)))
        print("pick up number of timeSlot 6" + "   "+str(len(timeSlot6)))
        print("totalRecord in this chunk"+ "   "+str(totalSize))
        print('Total time spent on loading car data {:.5f} second.'.format(cost_time))
        print("--------Done!--------")


filePath = "E:/data/ExprimentField/manhatan/trip_data1_Manhatan.csv"
splitCSVtoTimeSlot(filePath)
