import pandas as pd
import random
#这函数是为每个gps点分配网格id lable
LON1 = 40.694339
LON2 = 40.860598
LAT1 = -74.035471
LAT2 = -73.926392
df = pd.read_csv("E:/data/ExprimentField/manhatan/Poi_NYC_Manhatan.csv", encoding='utf-8')
df.head()
data = df[['latitude', 'longitude']].values
data1=pd.DataFrame(data)
# lon =[]
# lat =[]
# for i in range(100):
#     lon.append(round(random.uniform(LON1, LON2), 4))
#     lat.append(round(random.uniform(LAT1, LAT2), 4))
# c={"lon":lon,
#    'lat':lat}
# data= pd.DataFrame(c)
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


data1['label'] = data1.apply(lambda x: generalID(x[0], x[1],100,100), axis = 1)
data1.to_csv('E:/data/ExprimentField/manhatan/Poi_NYC_Manhatan_GridDivide.csv', encoding='utf-8')
