

import pandas as pd
import os
import csv
import numpy as np
#Trec3:将三个csv文件按列合并为一个csv:labelpu,labeldo,medallion,pickup_datetime, dropoff_datetime, trip_time_in_secs
origin_csv="E:/data/ExprimentField/test/timeSlot0_new.csv"
inputfile_csv_1="E:/data/ExprimentField/test/timeSlot0_pu_v2.csv"
inputfile_csv_2="E:/data/ExprimentField/test/timeSlot0_do_v2.csv"
outputfile="E:/data/ExprimentField/test/timeSlot0_all_v3.csv"
origin_data = pd.read_csv(origin_csv)
data_pu = origin_data.drop([" pickup_longitude"],axis = 1)
data_pu = data_pu.drop([" pickup_latitude"],axis = 1)
data_pu = data_pu.drop([" dropoff_longitude"],axis = 1)
data_pu = data_pu.drop([" dropoff_latitude"],axis = 1)
csv_1=pd.read_csv(inputfile_csv_1)
csv_2=pd.read_csv(inputfile_csv_2)
out_csv=pd.concat([csv_1,csv_2,data_pu],axis=1)
out_csv.to_csv(outputfile,index=False)