
# -*- coding: utf-8 -*-
import sys
print(sys.path)
# import matplotlib.pyplot as plt
# import pylab
# import string
# import math
# import numpy
import getData as gd
import networkx as nx
import matplotlib as plt

# 创建空图
G = nx.DiGraph()

# 创建图中的一类节点 为POI bipartite=0
filePath = "./data/PoIs_NYC.txt"
G, lontitudelatitudeArray = gd.getPOIFromCSVtoGraph(filePath, G)
# print lontitudelatitudeArray
# 创建POI坐标的KDTree
mytree = gd.creatKdTree(lontitudelatitudeArray)

# 添加图中的另一类节点 为时间节点 bipartite=1
# G=gd.getTimeNodestoGraph(filePath,G)

# nx.draw_networkx(G)

filePath = "./data/40%Of30.csv"
IfPrint = 0
G = gd.getEdgesFromCSVtoGraph(filePath, G, lontitudelatitudeArray, mytree, IfPrint)
print(nx.number_of_edges(G))

# nx.write_edgelist(G, "./data/test.edgelist", comments='#', delimiter=' ', data=True, encoding='utf-8')
#nx.draw(G)                          #绘制网络G
# plt.savefig("ba.png")         #将图像存为一个png格式的图片文件
# plt.show()
# nx.write_gexf(G,'EmbbeddingGraph.gexf')