# import dgl
# import torch
#
# # 创建异构图
# num_nodes = {'item': 800, 'user': 24586}
# edges = (torch.tensor([0]), torch.tensor([0]))  # 示例边
# G = dgl.heterograph({
#     ('user', 'buy', 'item'): edges
# }, num_nodes)
#
# # 添加反向边以创建无向图
# u, v = G.edges(etype='buy')
# # 添加反向边 (item, buyed, user)
# G.add_edges(v, u, etype='buyed')
#
# # 更新元图
# G.add_etypes(('item', 'buyed', 'user'))
#
# # 检查结果
# print(G)
# print("Num edges:", G.num_edges())
# print("Num nodes:", G.num_nodes())
# print("All edge types:", G.etypes)