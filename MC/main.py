import os
import base64
import csv
import io
import numpy as np  
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from surprise import SVD, Dataset, Reader, accuracy,NMF 
from surprise.model_selection import cross_validate, train_test_split,GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt


# 读取并处理书籍数据
dir=os.path.dirname(os.path.abspath(__file__))
path=os.path.join(dir,'BX-Book-Ratings.csv')
ratings_df=pd.read_csv(path, sep=';', header=0, on_bad_lines='skip',encoding='iso-8859-1',dtype={3:str})
ratings_df.columns = ['user_id', 'book_id', 'rating']


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

def SVD_get_predictions(n_epochs=12, lr_all=0.005, reg_all=0.2):
    best_params = {'n_epochs': n_epochs, 'lr_all': lr_all, 'reg_all': reg_all}
    algo = SVD(**best_params)
    algo.fit(trainset)
    predictions = algo.test(testset)
    return predictions

def NMF_get_predictions(n_epochs=25, n_factors=80):
    best_params = {'n_epochs': n_epochs, 'n_factors': n_factors}
    algo = NMF(**best_params)
    algo.fit(trainset)
    predictions = algo.test(testset)
    return predictions


















