import os
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import cross_validate, train_test_split,GridSearchCV
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

param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005],'reg_all': [0.4, 0.6]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
gs.fit(data)

print(gs.best_score['rmse'])
print(gs.best_params['rmse'])

# 使用最佳参数创建模型
algo = SVD(**gs.best_params['rmse'])

# 分割数据集
trainset, testset = train_test_split(data, test_size=.25)

# 训练模型
algo.fit(trainset)

# 预测
predictions = algo.test(testset)

# 计算 RMSE
accuracy.rmse(predictions)

actual_ratings = [pred.r_ui for pred in predictions]
pred_ratings = [pred.est for pred in predictions]
plt.scatter(actual_ratings, pred_ratings,s=1)
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Actual vs Predicted Ratings')
plt.show()
