import os
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import cross_validate, train_test_split


dataset_path = os.path.join(os.getcwd(),'MC\\archive')

# 读取并处理评分数据
ratings_file_path = os.path.join(dataset_path, 'BX-Book-Ratings.csv')
ratings_df = pd.read_csv(ratings_file_path, sep=';', header=0,encoding='iso-8859-1')
ratings_df.columns = ['user_id', 'book_id', 'rating']
ratings_df['user_id'] = ratings_df['user_id'].apply(lambda x: int(x))
ratings_df['rating'] = ratings_df['rating'].apply(lambda x: float(x[1:-1]) if isinstance(x, str) else float(x))

# 读取并处理书籍数据
books_file_path = os.path.join(dataset_path, 'BX-Books.csv')
books_df = pd.read_csv(books_file_path, sep=';', header=0, on_bad_lines='warn')
books_df.columns = ['book_id', 'title', 'author', 'year_of_publication', 'publisher', 'image_url_s', 'image_url_m', 'image_url_l']
books_df['title'] = books_df['title'].apply(lambda x: x[1:-1])
books_df['author'] = books_df['author'].apply(lambda x: x[1:-1])
books_df['year_of_publication'] = books_df['year_of_publication'].apply(lambda x: x[1:-1])
books_df['publisher'] = books_df['publisher'].apply(lambda x: x[1:-1])
books_df['image_url_s'] = books_df['image_url_s'].apply(lambda x: x[1:-1])
books_df['image_url_m'] = books_df['image_url_m'].apply(lambda x: x[1:-1])
books_df['image_url_l'] = books_df['image_url_l'].apply(lambda x: x[1:-1])

# 使用 LabelEncoder 对书籍 ID 进行编码
le = LabelEncoder()
books_df['book_id'] = le.fit_transform(books_df['book_id'])
ratings_df['book_id'] = le.transform(ratings_df['book_id'])


# 数据需要以三列的形式存在，分别是 user_id, item_id 和 rating
reader = Reader(rating_scale=(0,10))
data = Dataset.load_from_df(ratings_df[['user_id', 'book_id', 'rating']], reader)

# 使用 SVD 算法，这是一种 ALS 的变体
algo = SVD()

# 划分训练集和验证集
trainset, testset = train_test_split(data, test_size=.25)

# 训练模型
algo.fit(trainset)

# 预测
predictions = algo.test(testset)

# 计算 RMSE
accuracy.rmse(predictions)

pred_ratings = [pred.est for pred in predictions]
plt.hist(pred_ratings, bins=10, edgecolor='black')
plt.xlabel('Predicted Ratings')
plt.ylabel('Count')
plt.title('Distribution of Predicted Ratings')
plt.show()