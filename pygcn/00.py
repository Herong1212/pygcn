# 读取 content 文件的所有节点 ID
content_ids = set()
with open("../data/cora/cora.content") as f:
    for line in f:
        content_ids.add(line.split()[0])

# 读取 cites 文件的所有节点 ID
cites_ids = set()
with open("../data/cora/cora.cites") as f:
    for line in f:
        source, target = line.split()
        cites_ids.add(source)
        cites_ids.add(target)

# 检查 cites 文件中的节点是否都在 content 文件中
if cites_ids.issubset(content_ids):
    print("所有 cites 文件中的节点都能在 content 文件中找到。")
else:
    print("有节点在 cites 文件中，但不在 content 文件中。")
