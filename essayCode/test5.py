import numpy as np

import pandas as pd
#添加随机生成的方向和时间
expect_t=pd.DataFrame(np.random.normal(300,150 , [2722421,1]))

data1 = pd.read_csv("E:/data/ExprimentField/test/timeSlot0_v5.csv",encoding='utf-8',)
data1 = data1.drop(["0"],axis = 1)

# data[u'buy_place'] = data[u'buy_place'].astype(str)
# data[u'buy_place'] = data[u'buy_place'].apply(lambda x :x.split(' ')[-1])
out_csv=pd.concat([data1,expect_t],axis=1)
out_csv.to_csv("E:/data/ExprimentField/test/timeSlot0_v5.csv",index=False, encoding='utf-8')