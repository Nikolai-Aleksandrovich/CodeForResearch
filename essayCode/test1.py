import pandas as pd
import csv

filePathtest = "E:/data/ExprimentField/test/newtimeSlot0.csv"
data = pd.read_csv(filePathtest)

# data_new = data.drop(["hack_license"], axis=1)  # 删除title这列数据
# data_new = data_new.drop(["vendor_id"], axis=1)
# data_new = data_new.drop(["rate_code"], axis=1)
# data_new = data_new.drop(["store_and_fwd_flag"], axis=1)
data_new = data.drop([" passenger_count"], axis=1)





# 、、、、对于data进行多次操作，如果想要连续操作，记得都将.号之前的主语改成同一pandas对象，
# 比如前来两个操作，第二个主语需要改成data_new对象。如果想要保存新的csv文件，则为：

data_new.to_csv("E:/data/ExprimentField/test/newTimeSlot0.csv", index=0)
