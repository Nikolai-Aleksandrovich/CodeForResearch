import pandas as pd
import random
LON1 = 40.694339
LON2 = 40.860598
LAT1 = -74.035471
LAT2 = -73.926392
df = pd.read_csv("E:/data/ExprimentField/test/neo10%of10%.csv", encoding='utf-8')
df.head()
data1 = df[[' pickup_latitude', ' pickup_longitude']].values
data2 = df[[' dropoff_latitude', ' dropoff_longitude']].values
data3=pd.DataFrame(data1)
data4=pd.DataFrame(data2)

def generalID(lon,lat,column_num,row_num):
    # 若在范围外的点，返回-1
    # if lon <= LON1 or lon >= LON2 or lat <= LAT1 or lat >= LAT2:
    #     return -1
    # 把经度范围根据列数等分切割
    column = (LON2 - LON1)/column_num
    # 把纬度范围根据行数数等分切割
    row = (LAT2 - LAT1)/row_num
    # 二维矩阵坐标索引转换为一维ID，即： （列坐标区域（向下取整）+ 1） + （行坐标区域 * 列数）
    return int((lon-LON1)/column)+ 1 + int((lat-LAT1)/row) * column_num


data3['label'] = data3.apply(lambda x: generalID(x[0], x[1],100,100), axis = 1)
data4['label'] = data4.apply(lambda x: generalID(x[0], x[1],100,100), axis = 1)
data3.to_csv('E:/data/ExprimentField/test/DO10%of10%.csv', encoding='utf-8')
data4.to_csv('E:/data/ExprimentField/test/DO10%of10%.csv', encoding='utf-8')
