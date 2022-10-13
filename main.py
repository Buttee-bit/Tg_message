import pandas as pd
from tqdm import tqdm

from model import *
from preprocess import *

base_words = ['сдаться', 'плен', 'пленный', 'военнопленный', "сдача", 'раненый', 'поставка',
              'помощь', 'пво', 'сбить', 'ракета', 'авиация', "уничтожить", "наркотик"]

FEATURE_DIM = 8000  # размерность слов
N_CLUSTERS = 3  # количество кластеров

# загружаем все модели
vec_file = 'weights/vectorizer_45k_re.pkl'
dec_file = 'weights/svd_45_8k_re.pkl'
model_file = 'weights/KMeans_3cls.pkl'


def filter_neutral_n_drugs(vectorizer,
                           df):  # принимаем исходный датафрейм, на выходе получаем  датафрейм с наркотиками и прочим
    trash_data = []  # кроме того исходныф датафрейм рчищается от наркотков и прочего
    trash_classes = []
    index_base_words = [vectorizer.vocabulary_[w] for w in base_words]
    df.reset_index(drop=True, inplace=True)  # erase the index
    for row in tqdm(df):
        vec = vectorizer.transform([row]).toarray()[0]
        if sum([vec[i] for i in index_base_words]) == 0:  # если компоненты индикатора равны 0- значит нейтральный класс
            df = df[df != row]  # убираем из основных данных, чтобы они модель не путали
            trash_data.append(row)  # записываем в список
            trash_classes.append(4)  # и класс тоже

        elif vec[index_base_words[
            -1]] != 0:  # ecли координата отвечающая за слово наркотики ненулевая, значит текст про наркотики
            df = df[df != row]  # убираем из основных данных, чтобы они модель не путали
            trash_data.append(row)
            trash_classes.append(3)

        else:
            continue  # если не то ни другое, то переходим к следующей

    trash_dict = {'news': trash_data, 'class': trash_classes}
    trash_df = pd.DataFrame(trash_dict, columns=trash_dict.keys())
    trash_df.reset_index(drop=True, inplace=True)
    return trash_df


def predict_on_DataFrame(df):  # вход dataframe выход - dataframe с предсказаниями
    data = df.text
    data = data.dropna()

    cleaned_data = data.apply(clean)  # чистим

    vectorizer = load_vectorizer(vec_file)  # подгружаем веторы
    decompositor = load_decompositor(dec_file)  # подгружаем сокращение размерности
    Cluster_model = load_model(model_file)  # подгружаем модель
    print("in-cluster distance:", Cluster_model.inertia_)
    trash_df = filter_neutral_n_drugs(vectorizer, cleaned_data)  # десь датафрейм где наркотики и прочие

    predicts = []
    tfidf_vectors = vectorizer.transform(cleaned_data)  # получаем векторы для оставшейся части датасета
    rare_vectors = decompositor.transform(tfidf_vectors)
    for rare_vec in tqdm(rare_vectors):  # получаем предсвазания
        pr = Cluster_model.predict(rare_vec.reshape(1, -1))
        predicts.append(pr[0])

    pd_data = {'news': cleaned_data, 'class': predicts}  # оформляем в dataframe
    df_preds = pd.DataFrame(pd_data, columns=pd_data.keys())
    df_preds.reset_index(drop=True, inplace=True)
    result = pd.concat([df_preds, trash_df])
    result = result.reset_index(drop=True)
    return result


if __name__ == '__main__':
    df = pd.read_csv('data_tg.csv')
    df = df[:1000]
    df_preds = predict_on_DataFrame(df)
    df_preds.to_csv('predictions.csv')
