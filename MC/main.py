import os
import numpy as np  
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from surprise import SVD, Dataset, Reader, accuracy,NMF 
from surprise.model_selection import cross_validate, train_test_split,GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt

dataset_path = os.path.join(os.getcwd(),'MC\\archive')

# 读取并处理评分数据
ratings_file_path = os.path.join(dataset_path, 'BX-Book-Ratings.csv')
ratings_df = pd.read_csv(ratings_file_path, sep=';', header=0,encoding='iso-8859-1')
ratings_df.columns = ['user_id', 'book_id', 'rating']
ratings_df['rating'] = ratings_df['rating'].apply(lambda x: float(x))



# 读取并处理书籍数据
books_file_path = os.path.join(dataset_path, 'BX-Books.csv')
books_df = pd.read_csv(books_file_path, sep=';', header=0, on_bad_lines='skip',encoding='iso-8859-1',dtype={3:str})
books_df.columns = ['book_id', 'title', 'author', 'year_of_publication', 'publisher', 'image_url_s', 'image_url_m', 'image_url_l']

mask = ratings_df['book_id'].isin(books_df['book_id'])
ratings_df = ratings_df[mask]

# 使用 LabelEncoder 对书籍 ID 进行编码
le = LabelEncoder()
books_df['book_id'] = le.fit_transform(books_df['book_id'])
ratings_df['book_id'] = le.transform(ratings_df['book_id'])


reader = Reader(rating_scale=(0,10))
data = Dataset.load_from_df(ratings_df[['user_id', 'book_id', 'rating']], reader)
#12 0.005 0.2
# param_grid = {'n_epochs': [16,14,12, 10], 'lr_all': [0.008,0.007,0.006, 0.005],'reg_all': [0.5,0.4, 0.3,0.2]}
# gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
# gs.fit(data)
# 定义参数网格
# param_grid = {'n_epochs': [10,25, 50], 'n_factors': [10, 25, 50]}
# # 进行网格搜索
# gs = GridSearchCV(NMF, param_grid, measures=['rmse'], cv=3)
# gs.fit(data)
# # 输出最佳参数
# print(gs.best_params['rmse'])


# 分割数据集
trainset, testset = train_test_split(data, test_size=.25)

def SVD_get_predictions():
    best_params = {'n_epochs': 12, 'lr_all': 0.005, 'reg_all': 0.2}
    algo = SVD(**best_params)
    algo.fit(trainset)
    predictions = algo.test(testset)
    return predictions

def NMF_get_predictions():
    best_params = {'n_epochs': 25, 'n_factors': 80}  # 假设的最佳参数
    algo = NMF(**best_params)
    algo.fit(trainset)
    predictions = algo.test(testset)
    return predictions


















# 展示预测结果
# for uid, iid, true_r, est, _ in predictions[:10]:
#     print(f'user {uid} for item {iid}: actual rating {true_r}, estimated rating {est}')


# 计算误差
# errors = [true_r - est for uid, iid, true_r, est, _ in predictions]
# 绘制误差分布图
# plt.hist(errors, bins='auto')
# plt.title('Distribution of prediction errors')
# plt.xlabel('Error')
# plt.ylabel('Frequency')
# plt.show()