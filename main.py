import pandas as pd

import numpy as np  # Added missing import statement

from sklearn.metrics.pairwise import cosine_similarity

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

def matrixModify(path):
    rating=pd.read_csv(path)
    rating=rating.fillna(0)
    rating_similarity = cosine_similarity(rating)
    rating_similarity_df = pd.DataFrame(rating_similarity, index=rating.index, columns=rating.index)
    return rating_similarity_df
    

# 创建一个函数，输入一个物品，返回最相似的物品
def recommend_items(item, similarity_matrix, n_recommendations):
    similarity_scores = list(enumerate(similarity_matrix.loc[item]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    rec_items = [similarity_matrix.index[i] for i, _ in similarity_scores[1:n_recommendations+1]]
    return rec_items
