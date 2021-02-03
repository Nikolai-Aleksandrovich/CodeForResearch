import networkx as nx
G = nx.DiGraph()
G.add_node(1, time="5pm")
G.add_node(2, time="6pm")
G.add_node(3, time="7pm")
G.add_node(4, time="8pm")
G.add_node(5, time="9m")

print(G.nodes[1]['time'])
# for node, attr in G.nodes(data=True):
#     frequency = attr['time']
#     print(frequency)

