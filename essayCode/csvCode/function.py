
import csv
import os


def split_csv(path, total_len, per):

    # 如果train.csv和vali.csv存在就删除
    if os.path.exists('E:\\csv divide area\\smaller split\\MoreSmallerSets\\SmallerSets\\MoreSmallerSets\\40%.csv'):
        os.remove('E:\\csv divide area\\smaller split\\MoreSmallerSets\\SmallerSets\\MoreSmallerSets\\40%.csv')
    if os.path.exists('E:\\csv divide area\\smaller split\\MoreSmallerSets\\SmallerSets\\MoreSmallerSets\\60%.csv'):
        os.remove('E:\\csv divide area\\smaller split\\MoreSmallerSets\\SmallerSets\\MoreSmallerSets\\60%.csv')

    with open(path, 'r', newline='') as file:
        csvreader = csv.reader(file)
        i = 0
        for row in csvreader:

            if i < round(total_len * per/100):
                # train.csv存放路径
                csv_path = os.path.join("E:\\data\\ExprimentField\\test\\feb\\feb0", 'test.csv')
                # print(csv_path)
                # 不存在此文件的时候，就创建
                if not os.path.exists(csv_path):
                    with open(csv_path, 'w', newline='') as file:
                        csvwriter = csv.writer(file)
                        csvwriter.writerow(row)
                    i += 1
                # 存在的时候就往里面添加
                else:
                    with open(csv_path, 'a', newline='') as file:
                        csvwriter = csv.writer(file)
                        csvwriter.writerow(row)
                    i += 1
            elif (i >= round(total_len * per/100)) and (i < total_len):
            	# vali.csv存放路径
                csv_path = os.path.join("E:\\data\\ExprimentField\\test\\feb\\feb0", 'train.csv')
                print(csv_path)
                # 不存在此文件的时候，就创建
                if not os.path.exists(csv_path):
                    with open(csv_path, 'w', newline='') as file:
                        csvwriter = csv.writer(file)
                        csvwriter.writerow(row)
                    i += 1
                # 存在的时候就往里面添加
                else:
                    with open(csv_path, 'a', newline='') as file:
                        csvwriter = csv.writer(file)
                        csvwriter.writerow(row)
                    i += 1
            else:
                break

    print("训练集和验证集分离成功")
    return
if __name__ == '__main__':

    path = 'E:\\data\\ExprimentField\\test\\feb\\feb0\\timeSlot0.csv'
    total_len = len(open(path, 'r').readlines())# csv文件行数
    per = 20 # 分割比例%

    split_csv(path, total_len, per)


#
# jan0='E:\\data\\ExprimentField\\test\\jan\\jan0\\timeSlot0.csv'
# jan1='E:\\data\\ExprimentField\\test\\jan\\jan1\\timeSlot1.csv'
# jan2='E:\\data\\ExprimentField\\test\\jan\\jan2\\timeSlot2.csv'
# jan3='E:\\data\\ExprimentField\\test\\jan\\jan3\\timeSlot3.csv'
# jan4='E:\\data\\ExprimentField\\test\\jan\\jan4\\timeSlot4.csv'
# jan5='E:\\data\\ExprimentField\\test\\jan\\jan5\\timeSlot5.csv'
# jan6='E:\\data\\ExprimentField\\test\\jan\\jan6\\timeSlot6.csv'
# feb0='E:\\data\\ExprimentField\\test\\feb\\feb0\\timeSlot0.csv'
# feb1='E:\\data\\ExprimentField\\test\\feb\\feb1\\timeSlot1.csv'
# feb2='E:\\data\\ExprimentField\\test\\feb\\feb2\\timeSlot2.csv'
# feb3='E:\\data\\ExprimentField\\test\\feb\\feb3\\timeSlot3.csv'
# feb4='E:\\data\\ExprimentField\\test\\feb\\feb4\\timeSlot4.csv'
# feb5='E:\\data\\ExprimentField\\test\\feb\\feb5\\timeSlot5.csv'
# feb6='E:\\data\\ExprimentField\\test\\feb\\feb6\\timeSlot6.csv'
