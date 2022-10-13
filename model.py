from sklearn.cluster import KMeans
import joblib

def fit_model(model, rare_vectors):
    model = model.fit(rare_vectors)
    print('model fitted!')
    return model

def save_model(model, file):
    joblib.dump(model,file)
    print("model fitted!")

def load_model(file): # звгрузить из pkl файла
    model = joblib.load(file)
    print("model loaded!")
    return model

def predict(model, rare_vector):  # принимает вектор возвращает метку класса
    return model.predict(rare_vector.reshape(1, -1))[0]