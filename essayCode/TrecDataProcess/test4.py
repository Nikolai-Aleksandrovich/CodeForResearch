import numpy as np

import pandas as pd
#添加随机生成的方向和时间
expect_t=pd.DataFrame(np.random.normal(5, 1, [2722421,1]))
diection=pd.DataFrame(np.random.choice(['a', 'b', 'c','d','e','f','g','h','i'],[2722421,1]))
data1 = pd.read_csv("E:/data/ExprimentField/test/timeSlot0_do_v2.csv",encoding='utf-8',)
data1 = data1.drop([" dropoff_datetime"],axis = 1)
data1 = data1.drop(["pickup_datetime"],axis = 1)
# data[u'buy_place'] = data[u'buy_place'].astype(str)
# data[u'buy_place'] = data[u'buy_place'].apply(lambda x :x.split(' ')[-1])
out_csv=pd.concat([data1,expect_t,diection],axis=1)
out_csv.to_csv("E:/data/ExprimentField/test/timeSlot0_v4.csv",index=False, encoding='utf-8')