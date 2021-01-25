import pandas as pd
import csv
#TRec3：从网格数据去掉不需要的列，需要手动加上第一列的列标签id
data_pu = pd.read_csv("E:/data/ExprimentField/test/timeSlot0_new_pickup.csv")
data_do = pd.read_csv("E:/data/ExprimentField/test/timeSlot0_new_dropoff.csv")

# data_new = data.drop([128, 129, 130])  # 删除128，129，130行数据

data_pu = data_pu.drop(["id"],axis = 1)  # 删除title这列数据
data_pu  = data_pu .drop(["0"],axis = 1)
data_pu  = data_pu .drop(["1"],axis = 1)
data_do  = data_do.drop(["id"],axis = 1)
data_do = data_do.drop(["0"],axis = 1)
data_do = data_do.drop(["1"],axis = 1)


data_pu.to_csv("E:/data/ExprimentField/test/timeSlot0_pu_v2.csv", index=0)
data_do.to_csv("E:/data/ExprimentField/test/timeSlot0_do_v2.csv", index=0)