
import sys
import pickle as pkl


def load_cora():
    names = ['x']
    with open("data/ind.cora.x", 'rb') as f:
        if sys.version_info > (3, 0):
            print(f)  # <_io.BufferedReader name='data/ind.cora.x'>
            data = pkl.load(f, encoding='latin1')
            print(type(data))  # <class 'scipy.sparse.csr.csr_matrix'>

            print(data.shape)  # (140, 1433)-ind.cora.x是140行，1433列的
            print(data.shape[0])  # row:140
            print(data.shape[1])  # column:1433
            print(data[1])
            # 变量data是个scipy.sparse.csr.csr_matrix，类似稀疏矩阵，输出得到的是矩阵中非0的行列坐标及值
            # (0, 19)	1.0
            # (0, 88)	1.0
            # (0, 149)	1.0
            # (0, 212)	1.0
            # (0, 233)	1.0
            # (0, 332)	1.0
            # (0, 336)	1.0
            # (0, 359)	1.0
            # (0, 472)	1.0
            # (0, 507)	1.0
            # (0, 548)	1.0
            # ...

            # print(data[100][1]) #IndexError: index (1) out of range
            nonzero = data.nonzero()
            print(nonzero)  # 输出非零元素对应的行坐标和列坐标
            # (array([  0,   0,   0, ..., 139, 139, 139], dtype=int32), array([  19,   81,  146, ..., 1263, 1274, 1393], dtype=int32))
            # nonzero是个tuple
            print(type(nonzero))  # <class 'tuple'>
            print(nonzero[0])  # 行：[  0   0   0 ... 139 139 139]
            print(nonzero[1])  # 列：[  19   81  146 ... 1263 1274 1393]
            print(nonzero[1][0])  # 19
            print(data.toarray())


# [[0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]
#  ...
#  [0. 0. 0. ... 0. 1. 0.]
#  [0. 0. 0. ... 0. 0. 0.]
#  [0. 1. 0. ... 0. 0. 0.]]




def load_cora():
    with open("data/ind.cora.y", 'rb') as f:
        if sys.version_info > (3, 0):
            print(f)  # <_io.BufferedReader name='data/ind.cora.y'>
            data = pkl.load(f, encoding='latin1')
            print(type(data))  # <class 'numpy.ndarray'>
            print(data.shape)  # (140, 7)
            print(data.shape[0])  # row:140
            print(data.shape[1])  # column:7
            print(data[1])  # [0 0 0 0 1 0 0]





def load_cora():
    with open("data/ind.cora.graph", 'rb') as f:
        if sys.version_info > (3, 0):
            data = pkl.load(f, encoding='latin1')
            print(type(data))  # <class 'collections.defaultdict'>
            print(data)
        # defaultdict(<class 'list'>, {0: [633, 1862, 2582], 1: [2, 652, 654], 2: [1986, 332, 1666, 1, 1454],


#   , ... ,
#   2706: [165, 2707, 1473, 169], 2707: [598, 165, 1473, 2706]})




test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
print("test index:", test_idx_reorder)
# test index: [2692, 2532, 2050, 1715, 2362, 2609, 2622, 1975, 2081, 1767, 2263,..]
print("min_index:", min(test_idx_reorder))
# min_index: 1708


# 处理citeseer中一些孤立的点
if dataset_str == 'citeseer':
    # Fix citeseer dataset (there are some isolated nodes in the graph)
    # Find isolated nodes, add them as zero-vecs into the right position

    test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
    # print("test_idx_range_full.length",len(test_idx_range_full))
    # test_idx_range_full.length 1015

    # 转化成LIL格式的稀疏矩阵,tx_extended.shape=(1015,1433)
    tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
    # print(tx_extended)
    # [2312 2313 2314 2315 2316 2317 2318 2319 2320 2321 2322 2323 2324 2325
    # ....
    # 3321 3322 3323 3324 3325 3326]

    # test_idx_range-min(test_idx_range):列表中每个元素都减去min(test_idx_range)，即将test_idx_range列表中的index值变为从0开始编号
    tx_extended[test_idx_range - min(test_idx_range), :] = tx
    # print(tx_extended.shape) #(1015, 3703)

    # print(tx_extended)
    # (0, 19) 1.0
    # (0, 21) 1.0
    # (0, 169) 1.0
    # (0, 170) 1.0
    # (0, 425) 1.0
    #  ...
    # (1014, 3243) 1.0
    # (1014, 3351) 1.0
    # (1014, 3472) 1.0

    tx = tx_extended
    # print(tx.shape)
    # (1015, 3703)
    # 997,994,993,980,938...等15行全为0

    ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
    ty_extended[test_idx_range - min(test_idx_range), :] = ty
    ty = ty_extended
    # for i in range(ty.shape[0]):
    #     print(i," ",ty[i])
    #     # 980 [0. 0. 0. 0. 0. 0.]
    #     # 994 [0. 0. 0. 0. 0. 0.]
    #     # 993 [0. 0. 0. 0. 0. 0.]