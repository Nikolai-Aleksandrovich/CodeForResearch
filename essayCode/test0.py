import pandas as pd
import csv
#TRec1：从原始数据去掉不需要的列
data = pd.read_csv("E:/data/ExprimentField/test/timeSlot0.csv")

# data_new = data.drop([128, 129, 130])  # 删除128，129，130行数据

data_new = data.drop(["hack_license"],axis = 1)  # 删除title这列数据
data_new = data_new.drop(["vendor_id"],axis = 1)
data_new = data_new.drop(["rate_code"],axis = 1)
data_new = data_new.drop(["store_and_fwd_flag"],axis = 1)
# data_new = data_new.drop(["pickup_datetime"],axis = 1)
# data_new = data_new.drop([" dropoff_datetime"],axis = 1)
data_new = data_new.drop([" passenger_count"],axis = 1)
data_new = data_new.drop([" trip_distance"],axis = 1)


data_new.to_csv("E:/data/ExprimentField/test/timeSlot0_new.csv", index=0)