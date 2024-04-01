import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
data = pd.read_csv("C:/Users/10937/Documents/数据集/archive/JokeText.csv")
ratings1 = pd.read_csv("C:/Users/10937/Documents/数据集/archive/UserRatings1.csv")
ratings2 = pd.read_csv("C:/Users/10937/Documents/数据集/archive/UserRatings2.csv")
merged_ratings = pd.concat([ratings1, ratings2],axis=1)
#分割数据集
# 提取第一列
first_column = merged_ratings.iloc[:, 0]

# 删除第一列
merged_ratings = merged_ratings.drop(merged_ratings.columns[0], axis=1)

# 分割剩余的列
num_columns = merged_ratings.shape[1]
train_size = int(0.8 * num_columns)
train_data = merged_ratings.iloc[:, :train_size]
test_data = merged_ratings.iloc[:, train_size:]

# 将第一列添加回去
train_data.insert(0, 'JokeId', first_column)
test_data.insert(0, 'JokeId', first_column)
# 将数据集转换为稀疏矩阵
train_data_sparse = csr_matrix(train_data.values)
print(train_data.shape)
print(train_data_sparse.shape)

# 计算笑话之间的余弦相似度
# similarity_matrix = cosine_similarity(train_data_sparse.T)

# # 遍历训练集中的每个样本
# for i in range(train_data.shape[0]):
#     # 找到当前样本中缺失值的索引
#     missing_indexes = train_data.iloc[i].isnull()

#     # 如果当前样本有缺失值
#     if missing_indexes.any():
#         # 获取当前样本的非缺失值
#         known_ratings = train_data.iloc[i][~missing_indexes]

#         # 获取当前样本的笑话id
#         joke_id = train_data.iloc[i]['JokeId']

#         # 获取与当前笑话最相似的笑话的索引
#         most_similar_joke_index = np.argmax(similarity_matrix[joke_id])

#         # 使用最相似笑话的评分填充缺失值
#         train_data.iloc[i, missing_indexes] = train_data.iloc[most_similar_joke_index, missing_indexes]

