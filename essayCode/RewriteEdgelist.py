

import numpy as np
try:
    f = open("test1.weighted.edgelist")
except IOError:
    print("---**---打开文件失败！请检查是否存在该文件！并重新输入文件名!---**---")
print("filePath:", "test1.weighted.edgelist")
EdgeListNodeList = []
for line in f.readlines():
    lines = (line.strip().split(" "))
    EdgeListNodeList.append(lines)
EdgeListNode = np.array(EdgeListNodeList, dtype=float)
print(EdgeListNodeList)
newList = []

for l in EdgeListNodeList:
    if (float(l[2]) > 1.0):
        newList.append(l)
NewList = np.array(newList)

with open('a.another.edgelist','w') as w:
    for l in EdgeListNodeList:
        np.savetxt('a.another.edgelist',str(l))



