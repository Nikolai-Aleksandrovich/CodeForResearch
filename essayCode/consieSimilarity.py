
import time
import networkx as nx

import sklearn.metrics.pairwise as pw
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score


def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """

    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim


def bit_product_sum(x, y):
    return sum([item[0] * item[1] for item in zip(x, y)])


def precision_and_recall(ranked_list, ground_list):
    hits = 0
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_list:
            hits += 1
    pre = hits / (1.0 * len(ranked_list) if len(ground_list) != 0 else 1)
    rec = hits / (1.0 * len(ground_list) if len(ground_list) != 0 else 1)
    return pre, rec


def getDCG(scores):
    return np.sum(
        np.divide(np.power(2, scores) - 1, np.log(np.arange(scores.shape[0], dtype=np.float32) + 2)),
        dtype=np.float32)


def getNDCG(rank_list, pos_items):
    relevance = np.ones_like(pos_items)
    it2rel = {it: r for it, r in zip(pos_items, relevance)}
    rank_scores = np.asarray([it2rel.get(it, 0.0) for it in rank_list], dtype=np.float32)

    idcg = getDCG(np    .sort(relevance)[::-1])

    dcg = getDCG(rank_scores)

    if dcg == 0.0:
        return 0.0

    ndcg = dcg / idcg
    return ndcg


def AP(ranked_list, ground_truth):
    hits, sum_precs = 0, 0.0
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_truth:
            hits += 1
            sum_precs += hits / (i + 1.0)
    if hits > 0:
        return sum_precs / len(ground_truth)
    else:
        return 0.0


def cosine_similarity(x, y, norm=False):
    """ 计算两个向量x和y的余弦相似度 """
    assert len(x) == len(y), "len(x) != len(y)"
    zero_list = [0] * len(x)
    if x == zero_list or y == zero_list:
        return float(1) if x == y else float(0)

    # method 1
    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

    # method 2
    # cos = bit_product_sum(x, y) / (np.sqrt(bit_product_sum(x, x)) * np.sqrt(bit_product_sum(y, y)))

    # method 3
    # dot_product, square_sum_x, square_sum_y = 0, 0, 0
    # for i in range(len(x)):
    #     dot_product += x[i] * y[i]
    #     square_sum_x += x[i] * x[i]
    #     square_sum_y += y[i] * y[i]
    # cos = dot_product / (np.sqrt(square_sum_x) * np.sqrt(square_sum_y))

    return 0.5 * cos + 0.5 if norm else cos  # 归一化到[0, 1]区间内


def getCosineSimilarityFromOut(testEdgelist, trainResult, k):
    start_time = time.time()
    # 80%训练样本的Result.txt
    try:
        f = open(trainResult)
    except IOError:
        print("---**---打开文件失败！请检查是否存在该文件！并重新输入文件名!---**---")
    print("filePath:", trainResult)
    TrainResultCount = 0
    EdgeListNodeList = []
    for line in f.readlines():
        TrainResultCount = TrainResultCount + 1

        lines = (line.strip().split(" "))
        EdgeListNodeList.append(lines)

    TrainResult = np.array(EdgeListNodeList, dtype=float)
    # print(EdgeListNode)

    # 测试数据的edgelist文件
    try:
        f = open(testEdgelist)
    except IOError:
        print("---**---打开文件失败！请检查是否存在该文件！并重新输入文件名!---**---")
    print("filePath:", testEdgelist)
    TestitemsCount = 0
    TestVectorList = []
    for line in f.readlines():
        TestitemsCount = TestitemsCount + 1
        lines = line.strip().split(" ")
        TestVectorList.append(lines)
    TestList = np.array(TestVectorList, dtype=float)
    # print(TestVectorsArray)

    trainNodeID = []  # 较大训练数据集节点ID

    finalCSList = [[] for i in range(TrainResultCount)]
    a = 0

    SortedCarList = []
    ValidCount = 0

    for trainArray in TrainResult:
        CSListForEachRow = []
        trainNodeID.append(trainArray[0])
        trainVector = np.delete(trainArray, 0)
        for otherArray in TrainResult:
            if ((otherArray==trainArray).all()):
                continue
            otherVector=np.delete(otherArray, 0)
            CSListForEachRow.append(cos_sim(trainVector, otherVector))  # 每行的余弦相似性加入列表
        finalCSList[a] = CSListForEachRow
        a=a+1
    # print('start print finalCSList')
    # print(finalCSList)
    # print('end of finalCSList')
    TopKIndex = [[] for i in range(TrainResultCount)]
    for i in range(0, TrainResultCount):
        temp = finalCSList[i]
        for j in range(0, k):
            TopKIndex[i].append(temp.index(max(temp)))
            temp[temp.index(max(temp))] = 0.001

    # print('start print TopKIndex')
    # print(TopKIndex)
    # print('end of TopKIndex')
    hitratio = []
    precisionScore = 0
    recallScore = 0
    F1Score = 0
    ROC = 0
    HitRatioNumber = 0
    NDCG = 0
    AveragePrecision = 0

    for i in range(0,TrainResultCount):
        RealDestID = []
        RealWeight = []
        RecommendDestID = []
        topKRealDestID = []
        for line in TestList:
            if (trainNodeID[i] == line[0]):
                RealDestID.append(line[1])  # 真实对第i行值得去的K个地点
                RealWeight.append(line[2])
                for j in range(k):  # test[1，2，3，...，k-1,k]遍历每一个tuijian
                    RecommendDestID.append(trainNodeID[TopKIndex[i][j]])  # 推荐系统认为对第i行值得去的K个地点
        for g in range(k):
            if len(RealWeight)!=0:
                temp = max(RealWeight)
                index = RealWeight.index(temp)
                topKRealDestID.append(RealDestID[index])
                RealWeight[index] = 0.01

        tempprecisionScore, temprecallScore = precision_and_recall(RecommendDestID, topKRealDestID)
        precisionScore = precisionScore + tempprecisionScore
        recallScore = recallScore + temprecallScore
        NDCG = NDCG + getNDCG(RecommendDestID, topKRealDestID)
        AveragePrecision = AveragePrecision + AP(RecommendDestID, topKRealDestID)

        ifcomparefinish = False
        for recomID in RecommendDestID:
            if ifcomparefinish == True:
                continue
            for realID in topKRealDestID:
                if recomID == realID:
                    HitRatioNumber = HitRatioNumber + 1
                    ifcomparefinish = True
                    break
                else:
                    ifcomparefinish = False

    ans = HitRatioNumber / TestitemsCount
    precisionScore = precisionScore / TestitemsCount
    recallScore = recallScore / TestitemsCount
    F1Score = ((precisionScore * recallScore * 2) / (precisionScore + recallScore))
    NDCG = NDCG / TestitemsCount
    MAP = AveragePrecision / TestitemsCount
    end_time = time.time()
    cost_time = (end_time - start_time)
    print('Total time spent on loading car data {:.5f} second.'.format(cost_time))

    return ans, precisionScore, recallScore, F1Score, NDCG, MAP, k


testEdgelist = "E:/data/ExprimentField/test/jan/jan1/5测试图G/test.weighted.edgelist"
trainResult = "E:/data/ExprimentField/test/jan/jan1/4训练结果/result.txt"
fpath3 = "./data/test1.weighted.edgelist"


Hit, precisionScore, recallScore, F1Score, NDCG, MAP, t = getCosineSimilarityFromOut(testEdgelist, trainResult, 10)
# file1=open("E:/Anaconda/envs/Python35/src/code/essaycode/HYYcode/data/Percision.txt","w")
# file1.write(str(Perc))
# file1.close()
# file2=open("E:/Anaconda/envs/Python35/src/code/essaycode/HYYcode/data/recall.txt","w")
# file2.write(str(reca))
# file2.close()
print("Hit@", t, ":", Hit)
print("PrecisionScore@", t, ":", precisionScore)
print("RecallScore@", t, ":", recallScore)
print("F1Score@", t, ":", F1Score)
print("NDCG@", t, ":", NDCG)
print("MAP@", t, ":", MAP)

# file3 = open("E:/Anaconda/envs/Python35/src/code/essaycode/HYYcode/data/FHit.txt", "w")
# file3.write(str(Hit))
# file3.close()