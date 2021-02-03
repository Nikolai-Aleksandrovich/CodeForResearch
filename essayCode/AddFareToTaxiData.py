import csv
import pandas as pd

r = pd.read_csv("ROSTER.csv")    # 文件1
mci = pd.read_csv("MCI_RID_Mlabel.csv")  # 文件2

f = open('res.csv','w',encoding='utf-8',newline='')  # 写入文件，注意newline=''，没有会造成有空行
res = csv.writer(f)

res.writerow(["RID","label","PTID"])	# 写入表头

for i in range(0,mci.shape[0]):		# 依次遍历文件2
    d = mci.iloc[i]['RID']			# 获取当前行的 RID
    for j in range(0,r.shape[0]):	# 在文件 1 中依次查找 RID
        if r.iloc[j]['RID'] == d:
            res.writerow([d,mci.iloc[i]['label'],r.iloc[j]['PTID']])	# 查找到，写入新文件中
            break	# 直接break，进入下一行查找

f.close()  # 关闭文件，千万别忘了