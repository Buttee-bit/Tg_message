import joblib
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import  TruncatedSVD
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from preprocess import *
from model import *
from tqdm import tqdm
import pandas as pd

FEATURE_DIM=8000
N_CLUSTERS = 3
TRAIN_SIZE = 10000

vec_file = 'vectorizer.pkl'
dec_file = 'decompositor.pkl'
model_file = 'model.pkl'

# запусти этот скрипт чтобы обучить все модели
def fit_all(df): # вход dataframe выход - dataframe с предсказаниями
    data = df.text
    data = data.dropna()
    vectorizer = TfidfVectorizer() # настраиваем тфидфы и сокращение
    decompositor = TruncatedSVD(n_components=FEATURE_DIM)

    centers = [data[456], data[219], data[63]] # настраиваем центры
    centers = list(map(clean, centers)) # чистим
    data = shuffle(data)  # mix
    data = data[:TRAIN_SIZE]
    cleaned_data = data.apply(clean) # чистим
    cleaned_data.to_csv('cleaned_data.csv')
    print('cleaned')

    vectorizer = vectorizer.fit(cleaned_data) # обучаем векторы
    joblib.dump(vectorizer, vec_file)
    print('vectorizer fitted')

    train_tfidf_vectors = vectorizer.transform(cleaned_data) # для обучения сокращения размерности и получения прореженных векторов
    decompositor = decompositor.fit(train_tfidf_vectors) # обучаем сокращение размерности
    joblib.dump(decompositor,dec_file)
    print('decompositor fitted')
    print(centers)
    tfidf_centers = vectorizer.transform(centers) #векторы центров
    rare_vector_centers = decompositor.transform(tfidf_centers)

      # нужно для определения нейтрального класса
    rare_train_vectors = decompositor.transform(train_tfidf_vectors)  # для обучения

    Cluster_model = KMeans(n_clusters=N_CLUSTERS, init=rare_vector_centers, n_init=1) # обучаем модель
    Cluster_model.fit(rare_train_vectors)
    save_model(Cluster_model, model_file)
    print('fitting done!')

if __name__ == '__main__':
   df = pd.read_csv('data_tg.csv')
   fit_all(df)
