import datetime
import os
import h5py
import numpy as np
import pandas as pd
# #
# # data = pd.HDFStore("E:/BaseLine/Trec-master/models/weigts.01-0.00.hdf5")
# with h5py.File("E:/BaseLine/Trec-master/models/weigts.01-0.00.hdf5") as f:
#     for key in f.keys():
#         print(key)
#     data = f['model_weights']
#     print(data[1])

f = h5py.File("E:/BaseLine/Trec-master/models/weigts.01-0.00.hdf5",'r')   #打开h5文件
print(list(f.keys()))
dset_m = f['model_weights']
dset_o=f['optimizer_weights']
print("dset_m",list(dset_m.keys()))
print("dset_o",list(dset_o.keys()))
dset1=dset_m['embedding_1']
dset2=dset_o['Adam']
dset3=dset2['iterations:0']
print(dset3.shape)
print(dset3)
print(list(dset2.keys()))


# print(dset.shape)
# print(dset.dtype)
# print(f.keys())                            #可以查看所有的主键
# # a = f['data'][:]                    #取出主键为data的所有的键值
# print(f['model_weights'][:] )
# f.close()
