"""Derin Öğrenme Deneyleri

YSA, LSTM, CNN, Transformer ve Hiyerarşik Dikkat Ağı deneylerini içerir.

Bu dosya, çalışmada kullanılan bağımsız deney bloklarını model grubuna göre bir araya getirir.
Bölümler Google Colab üzerinde ayrı deneyler olarak çalıştırılmak üzere hazırlanmıştır;
bu nedenle veri yükleme ve bazı import adımları bölümler arasında tekrar eder.
"""


########################################################################################
# Yapay Sinir Ağı (YSA) — PCA Karşılaştırması
########################################################################################

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import re
import string

from google.colab import files
print("Lütfen veri setini yükleyin (ör. all-data.csv):")
uploaded = files.upload()

data_file_name = list(uploaded.keys())[0]
print("Yüklenen veri setinin adı:", data_file_name)

df = pd.read_csv(data_file_name, encoding='ISO-8859-1')
print("\nVeri setinin ilk 5 satırı:")
print(df.head())

def temizle_metin(metin):
    metin = metin.lower()
    metin = re.sub(f"[{re.escape(string.punctuation)}]", "", metin)
    metin = re.sub(r"\d+", "", metin)
    return metin

haber_basliklari = df.iloc[:, 1].apply(temizle_metin)
print("\nHaber Başlıklarının İlk 5 Satırı (Temizlenmiş):")
print(haber_basliklari.head())

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(haber_basliklari)

print("\nTF-IDF ile öznitelik matrisi oluşturuldu:")
print(f"Öznitelik matrisi boyutu: {X.shape}")

print("\nTF-IDF Matrisindeki Örnek Kelimeler:")
print(tfidf_vectorizer.get_feature_names_out()[:10])

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['neutral'])
y_categorical = to_categorical(y)

pca = PCA(n_components=100, random_state=42)
X_pca = pca.fit_transform(X.toarray())

explained_variance_ratio = pca.explained_variance_ratio_.sum()
print(f"\nPCA ile açıklanan toplam varyans oranı: {explained_variance_ratio:.4f}")

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results_pca = []
results_no_pca = []

def create_ann(input_dim, num_classes):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y), start=1):
    X_train_no_pca, X_test_no_pca = X[train_idx], X[test_idx]
    y_train_no_pca, y_test_no_pca = y_categorical[train_idx], y[test_idx]
    
    X_train_pca, X_test_pca = X_pca[train_idx], X_pca[test_idx]
    y_train_pca, y_test_pca = y_categorical[train_idx], y[test_idx]
    
    model_no_pca = create_ann(input_dim=X_train_no_pca.shape[1], num_classes=y_categorical.shape[1])
    model_no_pca.fit(X_train_no_pca.toarray(), y_train_no_pca, epochs=10, batch_size=32, verbose=0)
    y_pred_no_pca = model_no_pca.predict(X_test_no_pca.toarray())
    y_pred_classes_no_pca = np.argmax(y_pred_no_pca, axis=1)
    cm_no_pca = confusion_matrix(y_test_no_pca, y_pred_classes_no_pca)
    specificity_no_pca = cm_no_pca[0, 0] / (cm_no_pca[0, 0] + cm_no_pca[0, 1]) if cm_no_pca.shape[0] > 1 else 0
    results_no_pca.append([
        fold,
        accuracy_score(y_test_no_pca, y_pred_classes_no_pca),
        precision_score(y_test_no_pca, y_pred_classes_no_pca, average='weighted', zero_division=0),
        recall_score(y_test_no_pca, y_pred_classes_no_pca, average='weighted', zero_division=0),
        f1_score(y_test_no_pca, y_pred_classes_no_pca, average='weighted', zero_division=0),
        specificity_no_pca,
        roc_auc_score(y_test_no_pca, y_pred_no_pca, multi_class='ovr')
    ])
    
    model_pca = create_ann(input_dim=X_train_pca.shape[1], num_classes=y_categorical.shape[1])
    model_pca.fit(X_train_pca, y_train_pca, epochs=10, batch_size=32, verbose=0)
    y_pred_pca = model_pca.predict(X_test_pca)
    y_pred_classes_pca = np.argmax(y_pred_pca, axis=1)
    cm_pca = confusion_matrix(y_test_pca, y_pred_classes_pca)
    specificity_pca = cm_pca[0, 0] / (cm_pca[0, 0] + cm_pca[0, 1]) if cm_pca.shape[0] > 1 else 0
    results_pca.append([
        fold,
        accuracy_score(y_test_pca, y_pred_classes_pca),
        precision_score(y_test_pca, y_pred_classes_pca, average='weighted', zero_division=0),
        recall_score(y_test_pca, y_pred_classes_pca, average='weighted', zero_division=0),
        f1_score(y_test_pca, y_pred_classes_pca, average='weighted', zero_division=0),
        specificity_pca,
        roc_auc_score(y_test_pca, y_pred_pca, multi_class='ovr')
    ])

columns = ['Fold', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'AUC']
results_no_pca_df = pd.DataFrame(results_no_pca, columns=columns)
results_pca_df = pd.DataFrame(results_pca, columns=columns)

results_no_pca_df.loc['Average'] = results_no_pca_df.mean(numeric_only=True)
results_pca_df.loc['Average'] = results_pca_df.mean(numeric_only=True)

print("\nPCA Öncesi Sonuçlar:")
print(results_no_pca_df)

print("\nPCA Sonrası Sonuçlar:")
print(results_pca_df)


########################################################################################
# LSTM — PCA Karşılaştırması
########################################################################################

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
import re
import string

from google.colab import files
print("Lütfen veri setini yükleyin (ör. all-data.csv):")
uploaded = files.upload()

data_file_name = list(uploaded.keys())[0]
print("Yüklenen veri setinin adı:", data_file_name)

df = pd.read_csv(data_file_name, encoding='ISO-8859-1')
print("\nVeri setinin ilk 5 satırı:")
print(df.head())

def temizle_metin(metin):
    metin = metin.lower()
    metin = re.sub(f"[{re.escape(string.punctuation)}]", "", metin)
    metin = re.sub(r"\d+", "", metin)
    return metin

haber_basliklari = df.iloc[:, 1].apply(temizle_metin)
print("\nHaber Başlıklarının İlk 5 Satırı (Temizlenmiş):")
print(haber_basliklari.head())

max_words = 5000  # Maksimum kelime sayısı
max_len = 100  # Maksimum dizi uzunluğu
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(haber_basliklari)
X = tokenizer.texts_to_sequences(haber_basliklari)
X = pad_sequences(X, maxlen=max_len)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['neutral'])
y_categorical = to_categorical(y)

pca = PCA(n_components=50, random_state=42)  # Daha düşük bileşen sayısı
X_pca = pca.fit_transform(X)

explained_variance_ratio = pca.explained_variance_ratio_.sum()
print(f"\nPCA ile açıklanan toplam varyans oranı: {explained_variance_ratio:.4f}")

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results_pca = []
results_no_pca = []

def create_lstm_no_pca(input_dim, output_dim, num_classes):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=output_dim))
    model.add(Bidirectional(LSTM(32, return_sequences=True)))  # Daha küçük LSTM boyutu
    model.add(Dropout(0.3))
    model.add(LSTM(32))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_lstm_with_pca(input_dim, num_classes):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y), start=1):
    X_train_no_pca, X_test_no_pca = X[train_idx], X[test_idx]
    y_train_no_pca, y_test_no_pca = y_categorical[train_idx], y[test_idx]
    
    X_train_pca, X_test_pca = X_pca[train_idx], X_pca[test_idx]
    y_train_pca, y_test_pca = y_categorical[train_idx], y[test_idx]
    
    model_no_pca = create_lstm_no_pca(input_dim=max_words, output_dim=64, num_classes=y_categorical.shape[1])
    model_no_pca.fit(X_train_no_pca, y_train_no_pca, epochs=5, batch_size=64, verbose=0)
    y_pred_no_pca = model_no_pca.predict(X_test_no_pca)
    y_pred_classes_no_pca = np.argmax(y_pred_no_pca, axis=1)
    cm_no_pca = confusion_matrix(y_test_no_pca, y_pred_classes_no_pca)
    specificity_no_pca = cm_no_pca[0, 0] / (cm_no_pca[0, 0] + cm_no_pca[0, 1]) if cm_no_pca.shape[0] > 1 else 0
    results_no_pca.append([
        fold,
        accuracy_score(y_test_no_pca, y_pred_classes_no_pca),
        precision_score(y_test_no_pca, y_pred_classes_no_pca, average='weighted', zero_division=0),
        recall_score(y_test_no_pca, y_pred_classes_no_pca, average='weighted', zero_division=0),
        f1_score(y_test_no_pca, y_pred_classes_no_pca, average='weighted', zero_division=0),
        specificity_no_pca,
        roc_auc_score(y_test_no_pca, y_pred_no_pca, multi_class='ovr')
    ])
    
    model_pca = create_lstm_with_pca(input_dim=X_train_pca.shape[1], num_classes=y_categorical.shape[1])
    model_pca.fit(X_train_pca, y_train_pca, epochs=5, batch_size=64, verbose=0)
    y_pred_pca = model_pca.predict(X_test_pca)
    y_pred_classes_pca = np.argmax(y_pred_pca, axis=1)
    cm_pca = confusion_matrix(y_test_pca, y_pred_classes_pca)
    specificity_pca = cm_pca[0, 0] / (cm_pca[0, 0] + cm_pca[0, 1]) if cm_pca.shape[0] > 1 else 0
    results_pca.append([
        fold,
        accuracy_score(y_test_pca, y_pred_classes_pca),
        precision_score(y_test_pca, y_pred_classes_pca, average='weighted', zero_division=0),
        recall_score(y_test_pca, y_pred_classes_pca, average='weighted', zero_division=0),
        f1_score(y_test_pca, y_pred_classes_pca, average='weighted', zero_division=0),
        specificity_pca,
        roc_auc_score(y_test_pca, y_pred_pca, multi_class='ovr')
    ])

columns = ['Fold', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'AUC']
results_no_pca_df = pd.DataFrame(results_no_pca, columns=columns)
results_pca_df = pd.DataFrame(results_pca, columns=columns)

results_no_pca_df.loc['Average'] = results_no_pca_df.mean(numeric_only=True)
results_pca_df.loc['Average'] = results_pca_df.mean(numeric_only=True)

print("\nPCA Öncesi Sonuçlar:")
print(results_no_pca_df)

print("\nPCA Sonrası Sonuçlar:")
print(results_pca_df)


########################################################################################
# Evrişimsel Sinir Ağı (CNN) — PCA Karşılaştırması
########################################################################################

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout

from google.colab import files
print("Lütfen veri setini yükleyin (ör. all-data.csv):")
uploaded = files.upload()

data_file_name = list(uploaded.keys())[0]
print("Yüklenen veri setinin adı:", data_file_name)

df = pd.read_csv(data_file_name, encoding='ISO-8859-1')
print("\nVeri setinin ilk 5 satırı:")
print(df.head())

def temizle_metin(metin):
    return str(metin).lower()

haber_basliklari = df.iloc[:, 1].apply(temizle_metin).tolist()
print("\nHaber Başlıklarının İlk 5 Satırı (Temizlenmiş):")
print(haber_basliklari[:5])

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['neutral'])

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(haber_basliklari).toarray()

pca = PCA(n_components=100, random_state=42)
X_pca = pca.fit_transform(X)
explained_variance_ratio = pca.explained_variance_ratio_.sum()
print(f"\nPCA ile açıklanan toplam varyans oranı: {explained_variance_ratio:.4f}")

def create_cnn(input_dim, num_classes):
    model = Sequential([
        tf.keras.layers.Reshape((input_dim, 1), input_shape=(input_dim,)),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def calculate_specificity_multiclass(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    specificity_per_class = []
    for i in range(len(cm)):
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        fp = np.sum(cm[:, i]) - cm[i, i]
        if (tn + fp) > 0:
            specificity = tn / (tn + fp)
        else:
            specificity = 0
        specificity_per_class.append(specificity)
    return np.mean(specificity_per_class)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results_pca = []
results_no_pca = []

for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y), start=1):
    print(f"Fold {fold} başlıyor...")
    
    X_train_no_pca, X_test_no_pca = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    X_train_pca, X_test_pca = X_pca[train_idx], X_pca[test_idx]
    
    model_no_pca = create_cnn(input_dim=X_train_no_pca.shape[1], num_classes=len(np.unique(y)))
    model_no_pca.fit(X_train_no_pca, y_train, epochs=3, batch_size=32, verbose=0)
    y_pred_no_pca = model_no_pca.predict(X_test_no_pca)
    y_pred_classes_no_pca = np.argmax(y_pred_no_pca, axis=1)
    specificity_no_pca = calculate_specificity_multiclass(y_test, y_pred_classes_no_pca)
    results_no_pca.append([
        fold,
        accuracy_score(y_test, y_pred_classes_no_pca),
        precision_score(y_test, y_pred_classes_no_pca, average='weighted', zero_division=0),
        recall_score(y_test, y_pred_classes_no_pca, average='weighted', zero_division=0),
        f1_score(y_test, y_pred_classes_no_pca, average='weighted', zero_division=0),
        roc_auc_score(y_test, y_pred_no_pca, multi_class='ovr'),
        specificity_no_pca
    ])
    
    model_pca = create_cnn(input_dim=X_train_pca.shape[1], num_classes=len(np.unique(y)))
    model_pca.fit(X_train_pca, y_train, epochs=3, batch_size=32, verbose=0)
    y_pred_pca = model_pca.predict(X_test_pca)
    y_pred_classes_pca = np.argmax(y_pred_pca, axis=1)
    specificity_pca = calculate_specificity_multiclass(y_test, y_pred_classes_pca)
    results_pca.append([
        fold,
        accuracy_score(y_test, y_pred_classes_pca),
        precision_score(y_test, y_pred_classes_pca, average='weighted', zero_division=0),
        recall_score(y_test, y_pred_classes_pca, average='weighted', zero_division=0),
        f1_score(y_test, y_pred_classes_pca, average='weighted', zero_division=0),
        roc_auc_score(y_test, y_pred_pca, multi_class='ovr'),
        specificity_pca
    ])

columns = ['Fold', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC', 'Specificity']
results_no_pca_df = pd.DataFrame(results_no_pca, columns=columns)
results_pca_df = pd.DataFrame(results_pca, columns=columns)

results_no_pca_df.loc['Average'] = results_no_pca_df.mean(numeric_only=True)
results_pca_df.loc['Average'] = results_pca_df.mean(numeric_only=True)

print("\nPCA Öncesi Sonuçlar:")
print(results_no_pca_df)

print("\nPCA Sonrası Sonuçlar:")
print(results_pca_df)


########################################################################################
# Transformer — PCA Karşılaştırması
########################################################################################

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import re
import string

from google.colab import files
print("Lütfen veri setini yükleyin (ör. all-data.csv):")
uploaded = files.upload()

data_file_name = list(uploaded.keys())[0]
print("Yüklenen veri setinin adı:", data_file_name)

df = pd.read_csv(data_file_name, encoding='ISO-8859-1')
print("\nVeri setinin ilk 5 satırı:")
print(df.head())

def temizle_metin(metin):
    metin = metin.lower()
    metin = re.sub(f"[{re.escape(string.punctuation)}]", "", metin)
    metin = re.sub(r"\d+", "", metin)
    return metin

haber_basliklari = df.iloc[:, 1].apply(temizle_metin)
print("\nHaber Başlıklarının İlk 5 Satırı (Temizlenmiş):")
print(haber_basliklari.head())

tfidf_vectorizer = TfidfVectorizer(max_features=2000)
X = tfidf_vectorizer.fit_transform(haber_basliklari).toarray()

print("\nTF-IDF ile öznitelik matrisi oluşturuldu:")
print(f"Öznitelik matrisi boyutu: {X.shape}")

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['neutral'])

pca = PCA(n_components=30, random_state=42)
X_pca = pca.fit_transform(X)
explained_variance_ratio = pca.explained_variance_ratio_.sum()
print(f"\nPCA ile açıklanan toplam varyans oranı: {explained_variance_ratio:.4f}")

def create_transformer_classifier(input_dim, num_classes, num_heads=1, ff_dim=32):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Embedding(input_dim=2000, output_dim=32)(inputs)
    x = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=32)(x, x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def calculate_specificity_multiclass(cm):
    specificity_per_class = []
    for i in range(len(cm)):
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        fp = np.sum(cm[:, i]) - cm[i, i]
        if (tn + fp) > 0:
            specificity = tn / (tn + fp)
        else:
            specificity = 0
        specificity_per_class.append(specificity)
    return np.mean(specificity_per_class)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results_no_pca = []
results_pca = []

for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y), start=1):
    print(f"Fold {fold} başlıyor...")
    
    X_train_no_pca, X_test_no_pca = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    X_train_pca, X_test_pca = X_pca[train_idx], X_pca[test_idx]
    
    model_no_pca = create_transformer_classifier(input_dim=X_train_no_pca.shape[1], num_classes=len(np.unique(y)))
    model_no_pca.fit(X_train_no_pca, y_train, epochs=1, batch_size=8, verbose=1)
    y_pred_no_pca = np.argmax(model_no_pca.predict(X_test_no_pca), axis=1)
    cm_no_pca = confusion_matrix(y_test, y_pred_no_pca)
    specificity_no_pca = calculate_specificity_multiclass(cm_no_pca)
    results_no_pca.append([
        fold,
        accuracy_score(y_test, y_pred_no_pca),
        precision_score(y_test, y_pred_no_pca, average='weighted', zero_division=0),
        recall_score(y_test, y_pred_no_pca, average='weighted', zero_division=0),
        f1_score(y_test, y_pred_no_pca, average='weighted', zero_division=0),
        roc_auc_score(y_test, model_no_pca.predict(X_test_no_pca), multi_class='ovr'),
        specificity_no_pca
    ])

    model_pca = create_transformer_classifier(input_dim=X_train_pca.shape[1], num_classes=len(np.unique(y)))
    model_pca.fit(X_train_pca, y_train, epochs=1, batch_size=8, verbose=1)
    y_pred_pca = np.argmax(model_pca.predict(X_test_pca), axis=1)
    cm_pca = confusion_matrix(y_test, y_pred_pca)
    specificity_pca = calculate_specificity_multiclass(cm_pca)
    results_pca.append([
        fold,
        accuracy_score(y_test, y_pred_pca),
        precision_score(y_test, y_pred_pca, average='weighted', zero_division=0),
        recall_score(y_test, y_pred_pca, average='weighted', zero_division=0),
        f1_score(y_test, y_pred_pca, average='weighted', zero_division=0),
        roc_auc_score(y_test, model_pca.predict(X_test_pca), multi_class='ovr'),
        specificity_pca
    ])

columns = ['Fold', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)', 'AUC (%)', 'Specificity (%)']
results_no_pca_df = pd.DataFrame(results_no_pca, columns=columns)
results_pca_df = pd.DataFrame(results_pca, columns=columns)

results_no_pca_df.loc['Average'] = results_no_pca_df.mean(numeric_only=True)
results_pca_df.loc['Average'] = results_pca_df.mean(numeric_only=True)

print("\nPCA Öncesi Sonuçlar:")
print(results_no_pca_df)

print("\nPCA Sonrası Sonuçlar:")
print(results_pca_df)


########################################################################################
# Hiyerarşik Dikkat Ağı (HAN) — PCA Karşılaştırması
########################################################################################

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, GRU, Dense, GlobalAveragePooling1D
import re
import string
import numpy as np

from google.colab import files
print("Lütfen veri setini yükleyin (ör. all-data.csv):")
uploaded = files.upload()

data_file_name = list(uploaded.keys())[0]
print("Yüklenen veri setinin adı:", data_file_name)

df = pd.read_csv(data_file_name, encoding='ISO-8859-1')
print("\nVeri setinin ilk 5 satırı:")
print(df.head())

neutral_sayisi = df[df['neutral'] == 'neutral'].shape[0]
positive_sayisi = df[df['neutral'] == 'positive'].shape[0]
negative_sayisi = df[df['neutral'] == 'negative'].shape[0]

print("\nDuygu etiketlerinin sayısı:")
print(f"Nötr: {neutral_sayisi}")
print(f"Pozitif: {positive_sayisi}")
print(f"Negatif: {negative_sayisi}")

def temizle_metin(metin):
    metin = metin.lower()
    metin = re.sub(f"[{re.escape(string.punctuation)}]", "", metin)
    metin = re.sub(r"\d+", "", metin)
    return metin

haber_basliklari = df.iloc[:, 1].apply(temizle_metin)
print("\nHaber Başlıklarının İlk 5 Satırı (Temizlenmiş):")
print(haber_basliklari.head())

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(haber_basliklari).toarray()

print("\nTF-IDF ile öznitelik matrisi oluşturuldu:")
print(f"Öznitelik matrisi boyutu: {X.shape}")

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['neutral'])

pca = PCA(n_components=100, random_state=42)
X_pca = pca.fit_transform(X)

explained_variance_ratio = pca.explained_variance_ratio_.sum()
print(f"\nPCA ile açıklanan toplam varyans oranı: {explained_variance_ratio:.4f}")

def create_optimized_han_model(input_dim, num_classes):
    inputs = Input(shape=(input_dim,))
    x = Embedding(input_dim=5000, output_dim=16)(inputs)
    x = GRU(8, return_sequences=True)(x)
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results_no_pca = []
results_pca = []

for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y), start=1):
    print(f"Fold {fold} başlıyor...")

    X_train_no_pca, X_test_no_pca = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    X_train_pca, X_test_pca = X_pca[train_idx], X_pca[test_idx]

    model_no_pca = create_optimized_han_model(input_dim=X_train_no_pca.shape[1], num_classes=len(np.unique(y)))
    model_no_pca.fit(X_train_no_pca, y_train, epochs=1, batch_size=64, verbose=1)
    y_pred_no_pca = np.argmax(model_no_pca.predict(X_test_no_pca), axis=1)
    auc_no_pca = roc_auc_score(y_test, model_no_pca.predict(X_test_no_pca), multi_class='ovr')
    cm_no_pca = confusion_matrix(y_test, y_pred_no_pca)
    specificity_no_pca = cm_no_pca[0, 0] / (cm_no_pca[0, 0] + cm_no_pca[0, 1])
    results_no_pca.append([
        fold,
        accuracy_score(y_test, y_pred_no_pca),
        precision_score(y_test, y_pred_no_pca, average='weighted', zero_division=0),
        recall_score(y_test, y_pred_no_pca, average='weighted', zero_division=0),
        f1_score(y_test, y_pred_no_pca, average='weighted', zero_division=0),
        specificity_no_pca,
        auc_no_pca
    ])

    model_pca = create_optimized_han_model(input_dim=X_train_pca.shape[1], num_classes=len(np.unique(y)))
    model_pca.fit(X_train_pca, y_train, epochs=1, batch_size=64, verbose=1)
    y_pred_pca = np.argmax(model_pca.predict(X_test_pca), axis=1)
    auc_pca = roc_auc_score(y_test, model_pca.predict(X_test_pca), multi_class='ovr')
    cm_pca = confusion_matrix(y_test, y_pred_pca)
    specificity_pca = cm_pca[0, 0] / (cm_pca[0, 0] + cm_pca[0, 1])
    results_pca.append([
        fold,
        accuracy_score(y_test, y_pred_pca),
        precision_score(y_test, y_pred_pca, average='weighted', zero_division=0),
        recall_score(y_test, y_pred_pca, average='weighted', zero_division=0),
        f1_score(y_test, y_pred_pca, average='weighted', zero_division=0),
        specificity_pca,
        auc_pca
    ])

columns = ['Fold', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)', 'Specificity (%)', 'AUC (%)']
results_no_pca_df = pd.DataFrame(results_no_pca, columns=columns)
results_pca_df = pd.DataFrame(results_pca, columns=columns)

results_no_pca_df.loc['Average'] = results_no_pca_df.mean(numeric_only=True)
results_pca_df.loc['Average'] = results_pca_df.mean(numeric_only=True)

print("\nPCA Öncesi Sonuçlar:")
print(results_no_pca_df)

print("\nPCA Sonrası Sonuçlar:")
print(results_pca_df)


########################################################################################
# Yapay Sinir Ağı (YSA) — Varyans Eşiği ve PCA Karşılaştırması
########################################################################################

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import re
import string

from google.colab import files
print("Lütfen veri setini yükleyin (ör. all-data.csv):")
uploaded = files.upload()

data_file_name = list(uploaded.keys())[0]
print("Yüklenen veri setinin adı:", data_file_name)

df = pd.read_csv(data_file_name, encoding='ISO-8859-1')
print("\nVeri setinin ilk 5 satırı:")
print(df.head())

def temizle_metin(metin):
    metin = metin.lower()
    metin = re.sub(f"[{re.escape(string.punctuation)}]", "", metin)
    metin = re.sub(r"\d+", "", metin)
    return metin

haber_basliklari = df.iloc[:, 1].apply(temizle_metin)
print("\nHaber Başlıklarının İlk 5 Satırı (Temizlenmiş):")
print(haber_basliklari.head())

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(haber_basliklari)

print("\nTF-IDF ile öznitelik matrisi oluşturuldu:")
print(f"Öznitelik matrisi boyutu: {X.shape}")

threshold = 0.0005
var_thresh = VarianceThreshold(threshold=threshold)
X_var = var_thresh.fit_transform(X.toarray())
print(f"\nVaryans Eşiği Sonrası Öznitelik Matrisi Boyutu: {X_var.shape}")

pca = PCA(n_components=100, random_state=42)
X_pca = pca.fit_transform(X_var)

explained_variance_ratio = pca.explained_variance_ratio_.sum()
print(f"\nPCA ile açıklanan toplam varyans oranı: {explained_variance_ratio:.4f}")

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['neutral'])
y_categorical = to_categorical(y)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results_pca = []
results_no_pca = []

def create_ann(input_dim, num_classes):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

for fold, (train_idx, test_idx) in enumerate(kfold.split(X_var, y), start=1):
    # PCA öncesi veriler
    X_train_no_pca, X_test_no_pca = X_var[train_idx], X_var[test_idx]
    y_train_no_pca, y_test_no_pca = y_categorical[train_idx], y[test_idx]
    
    # PCA sonrası veriler
    X_train_pca, X_test_pca = X_pca[train_idx], X_pca[test_idx]
    y_train_pca, y_test_pca = y_categorical[train_idx], y[test_idx]
    
    model_no_pca = create_ann(input_dim=X_train_no_pca.shape[1], num_classes=y_categorical.shape[1])
    model_no_pca.fit(X_train_no_pca, y_train_no_pca, epochs=10, batch_size=32, verbose=0)
    y_pred_no_pca = model_no_pca.predict(X_test_no_pca)
    y_pred_classes_no_pca = np.argmax(y_pred_no_pca, axis=1)
    cm_no_pca = confusion_matrix(y_test_no_pca, y_pred_classes_no_pca)
    specificity_no_pca = cm_no_pca[0, 0] / (cm_no_pca[0, 0] + cm_no_pca[0, 1]) if cm_no_pca.shape[0] > 1 else 0
    results_no_pca.append([
        fold,
        accuracy_score(y_test_no_pca, y_pred_classes_no_pca),
        precision_score(y_test_no_pca, y_pred_classes_no_pca, average='weighted', zero_division=0),
        recall_score(y_test_no_pca, y_pred_classes_no_pca, average='weighted', zero_division=0),
        f1_score(y_test_no_pca, y_pred_classes_no_pca, average='weighted', zero_division=0),
        specificity_no_pca,
        roc_auc_score(y_test_no_pca, y_pred_no_pca, multi_class='ovr')
    ])
    
    model_pca = create_ann(input_dim=X_train_pca.shape[1], num_classes=y_categorical.shape[1])
    model_pca.fit(X_train_pca, y_train_pca, epochs=10, batch_size=32, verbose=0)
    y_pred_pca = model_pca.predict(X_test_pca)
    y_pred_classes_pca = np.argmax(y_pred_pca, axis=1)
    cm_pca = confusion_matrix(y_test_pca, y_pred_classes_pca)
    specificity_pca = cm_pca[0, 0] / (cm_pca[0, 0] + cm_pca[0, 1]) if cm_pca.shape[0] > 1 else 0
    results_pca.append([
        fold,
        accuracy_score(y_test_pca, y_pred_classes_pca),
        precision_score(y_test_pca, y_pred_classes_pca, average='weighted', zero_division=0),
        recall_score(y_test_pca, y_pred_classes_pca, average='weighted', zero_division=0),
        f1_score(y_test_pca, y_pred_classes_pca, average='weighted', zero_division=0),
        specificity_pca,
        roc_auc_score(y_test_pca, y_pred_pca, multi_class='ovr')
    ])

columns = ['Fold', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'AUC']
results_no_pca_df = pd.DataFrame(results_no_pca, columns=columns)
results_pca_df = pd.DataFrame(results_pca, columns=columns)

results_no_pca_df.loc['Average'] = results_no_pca_df.mean(numeric_only=True)
results_pca_df.loc['Average'] = results_pca_df.mean(numeric_only=True)

print("\nPCA Öncesi Sonuçlar:")
print(results_no_pca_df)

print("\nPCA Sonrası Sonuçlar:")
print(results_pca_df)


########################################################################################
# LSTM — Varyans Eşiği ve PCA Karşılaştırması
########################################################################################

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
import re
import string

from google.colab import files
print("Lütfen veri setini yükleyin (ör. all-data.csv):")
uploaded = files.upload()

data_file_name = list(uploaded.keys())[0]
print("Yüklenen veri setinin adı:", data_file_name)

df = pd.read_csv(data_file_name, encoding='ISO-8859-1')
print("\nVeri setinin ilk 5 satırı:")
print(df.head())

def temizle_metin(metin):
    metin = metin.lower()
    metin = re.sub(f"[{re.escape(string.punctuation)}]", "", metin)
    metin = re.sub(r"\d+", "", metin)
    return metin

haber_basliklari = df.iloc[:, 1].apply(temizle_metin)
print("\nHaber Başlıklarının İlk 5 Satırı (Temizlenmiş):")
print(haber_basliklari.head())

max_words = 5000  # Maksimum kelime sayısı
max_len = 100  # Maksimum dizi uzunluğu
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(haber_basliklari)
X = tokenizer.texts_to_sequences(haber_basliklari)
X = pad_sequences(X, maxlen=max_len)

threshold = 0.0005
var_thresh = VarianceThreshold(threshold=threshold)
X_var_thresh = var_thresh.fit_transform(X)
print(f"\nVaryans Eşiği Sonrası Öznitelik Matrisi Boyutu: {X_var_thresh.shape}")

pca_n_components = min(50, X_var_thresh.shape[1])  # Bileşen sayısı özellik sayısını aşamaz
pca = PCA(n_components=pca_n_components, random_state=42)
X_pca = pca.fit_transform(X_var_thresh)

explained_variance_ratio = pca.explained_variance_ratio_.sum()
print(f"\nPCA ile açıklanan toplam varyans oranı: {explained_variance_ratio:.4f}")

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['neutral'])
y_categorical = to_categorical(y)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results_pca = []
results_no_pca = []

def create_lstm_no_pca(input_dim, output_dim, num_classes):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=output_dim))
    model.add(Bidirectional(LSTM(32, return_sequences=True)))  # Daha küçük LSTM boyutu
    model.add(Dropout(0.3))
    model.add(LSTM(32))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_lstm_with_pca(input_dim, num_classes):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

for fold, (train_idx, test_idx) in enumerate(kfold.split(X_var_thresh, y), start=1):
    X_train_no_pca, X_test_no_pca = X_var_thresh[train_idx], X_var_thresh[test_idx]
    y_train_no_pca, y_test_no_pca = y_categorical[train_idx], y[test_idx]
    
    X_train_pca, X_test_pca = X_pca[train_idx], X_pca[test_idx]
    y_train_pca, y_test_pca = y_categorical[train_idx], y[test_idx]
    
    model_no_pca = create_lstm_no_pca(input_dim=max_words, output_dim=64, num_classes=y_categorical.shape[1])
    model_no_pca.fit(X_train_no_pca, y_train_no_pca, epochs=5, batch_size=64, verbose=0)
    y_pred_no_pca = model_no_pca.predict(X_test_no_pca)
    y_pred_classes_no_pca = np.argmax(y_pred_no_pca, axis=1)
    cm_no_pca = confusion_matrix(y_test_no_pca, y_pred_classes_no_pca)
    specificity_no_pca = cm_no_pca[0, 0] / (cm_no_pca[0, 0] + cm_no_pca[0, 1]) if cm_no_pca.shape[0] > 1 else 0
    results_no_pca.append([
        fold,
        accuracy_score(y_test_no_pca, y_pred_classes_no_pca),
        precision_score(y_test_no_pca, y_pred_classes_no_pca, average='weighted', zero_division=0),
        recall_score(y_test_no_pca, y_pred_classes_no_pca, average='weighted', zero_division=0),
        f1_score(y_test_no_pca, y_pred_classes_no_pca, average='weighted', zero_division=0),
        specificity_no_pca,
        roc_auc_score(y_test_no_pca, y_pred_no_pca, multi_class='ovr')
    ])
    
    model_pca = create_lstm_with_pca(input_dim=X_train_pca.shape[1], num_classes=y_categorical.shape[1])
    model_pca.fit(X_train_pca, y_train_pca, epochs=5, batch_size=64, verbose=0)
    y_pred_pca = model_pca.predict(X_test_pca)
    y_pred_classes_pca = np.argmax(y_pred_pca, axis=1)
    cm_pca = confusion_matrix(y_test_pca, y_pred_classes_pca)
    specificity_pca = cm_pca[0, 0] / (cm_pca[0, 0] + cm_pca[0, 1]) if cm_pca.shape[0] > 1 else 0
    results_pca.append([
        fold,
        accuracy_score(y_test_pca, y_pred_classes_pca),
        precision_score(y_test_pca, y_pred_classes_pca, average='weighted', zero_division=0),
        recall_score(y_test_pca, y_pred_classes_pca, average='weighted', zero_division=0),
        f1_score(y_test_pca, y_pred_classes_pca, average='weighted', zero_division=0),
        specificity_pca,
        roc_auc_score(y_test_pca, y_pred_pca, multi_class='ovr')
    ])

columns = ['Fold', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'AUC']
results_no_pca_df = pd.DataFrame(results_no_pca, columns=columns)
results_pca_df = pd.DataFrame(results_pca, columns=columns)

results_no_pca_df.loc['Average'] = results_no_pca_df.mean(numeric_only=True)
results_pca_df.loc['Average'] = results_pca_df.mean(numeric_only=True)

print("\nPCA Öncesi Sonuçlar:")
print(results_no_pca_df)

print("\nPCA Sonrası Sonuçlar:")
print(results_pca_df)


########################################################################################
# Evrişimsel Sinir Ağı (CNN) — Varyans Eşiği ve PCA Karşılaştırması
########################################################################################

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout

from google.colab import files
print("Lütfen veri setini yükleyin (ör. all-data.csv):")
uploaded = files.upload()

data_file_name = list(uploaded.keys())[0]
print("Yüklenen veri setinin adı:", data_file_name)

df = pd.read_csv(data_file_name, encoding='ISO-8859-1')
print("\nVeri setinin ilk 5 satırı:")
print(df.head())

def temizle_metin(metin):
    return str(metin).lower()

haber_basliklari = df.iloc[:, 1].apply(temizle_metin).tolist()
print("\nHaber Başlıklarının İlk 5 Satırı (Temizlenmiş):")
print(haber_basliklari[:5])

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['neutral'])

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(haber_basliklari).toarray()

threshold = 0.0005
var_thresh = VarianceThreshold(threshold=threshold)
X_var_thresh = var_thresh.fit_transform(X)
print(f"\nVaryans Eşiği Sonrası Öznitelik Matrisi Boyutu: {X_var_thresh.shape}")

pca = PCA(n_components=min(100, X_var_thresh.shape[1]), random_state=42)
X_pca = pca.fit_transform(X_var_thresh)
explained_variance_ratio = pca.explained_variance_ratio_.sum()
print(f"\nPCA ile açıklanan toplam varyans oranı: {explained_variance_ratio:.4f}")

def create_cnn(input_dim, num_classes):
    model = Sequential([
        tf.keras.layers.Reshape((input_dim, 1), input_shape=(input_dim,)),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def calculate_specificity_multiclass(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    specificity_per_class = []
    for i in range(len(cm)):
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        fp = np.sum(cm[:, i]) - cm[i, i]
        if (tn + fp) > 0:
            specificity = tn / (tn + fp)
        else:
            specificity = 0
        specificity_per_class.append(specificity)
    return np.mean(specificity_per_class)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results_pca = []
results_no_pca = []

for fold, (train_idx, test_idx) in enumerate(kfold.split(X_var_thresh, y), start=1):
    print(f"Fold {fold} başlıyor...")

    X_train_no_pca, X_test_no_pca = X_var_thresh[train_idx], X_var_thresh[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    X_train_pca, X_test_pca = X_pca[train_idx], X_pca[test_idx]
    
    model_no_pca = create_cnn(input_dim=X_train_no_pca.shape[1], num_classes=len(np.unique(y)))
    model_no_pca.fit(X_train_no_pca, y_train, epochs=3, batch_size=32, verbose=0)
    y_pred_no_pca = model_no_pca.predict(X_test_no_pca)
    y_pred_classes_no_pca = np.argmax(y_pred_no_pca, axis=1)
    specificity_no_pca = calculate_specificity_multiclass(y_test, y_pred_classes_no_pca)
    results_no_pca.append([
        fold,
        accuracy_score(y_test, y_pred_classes_no_pca),
        precision_score(y_test, y_pred_classes_no_pca, average='weighted', zero_division=0),
        recall_score(y_test, y_pred_classes_no_pca, average='weighted', zero_division=0),
        f1_score(y_test, y_pred_classes_no_pca, average='weighted', zero_division=0),
        roc_auc_score(y_test, y_pred_no_pca, multi_class='ovr'),
        specificity_no_pca
    ])
    
    model_pca = create_cnn(input_dim=X_train_pca.shape[1], num_classes=len(np.unique(y)))
    model_pca.fit(X_train_pca, y_train, epochs=3, batch_size=32, verbose=0)
    y_pred_pca = model_pca.predict(X_test_pca)
    y_pred_classes_pca = np.argmax(y_pred_pca, axis=1)
    specificity_pca = calculate_specificity_multiclass(y_test, y_pred_classes_pca)
    results_pca.append([
        fold,
        accuracy_score(y_test, y_pred_classes_pca),
        precision_score(y_test, y_pred_classes_pca, average='weighted', zero_division=0),
        recall_score(y_test, y_pred_classes_pca, average='weighted', zero_division=0),
        f1_score(y_test, y_pred_classes_pca, average='weighted', zero_division=0),
        roc_auc_score(y_test, y_pred_pca, multi_class='ovr'),
        specificity_pca
    ])

columns = ['Fold', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC', 'Specificity']
results_no_pca_df = pd.DataFrame(results_no_pca, columns=columns)
results_pca_df = pd.DataFrame(results_pca, columns=columns)

results_no_pca_df.loc['Average'] = results_no_pca_df.mean(numeric_only=True)
results_pca_df.loc['Average'] = results_pca_df.mean(numeric_only=True)

print("\nPCA Öncesi Sonuçlar:")
print(results_no_pca_df)

print("\nPCA Sonrası Sonuçlar:")
print(results_pca_df)


########################################################################################
# Transformer — Varyans Eşiği ve PCA Karşılaştırması
########################################################################################

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import re
import string

from google.colab import files
print("Lütfen veri setini yükleyin (ör. all-data.csv):")
uploaded = files.upload()

data_file_name = list(uploaded.keys())[0]
print("Yüklenen veri setinin adı:", data_file_name)

df = pd.read_csv(data_file_name, encoding='ISO-8859-1')
print("\nVeri setinin ilk 5 satırı:")
print(df.head())

def temizle_metin(metin):
    metin = metin.lower()
    metin = re.sub(f"[{re.escape(string.punctuation)}]", "", metin)
    metin = re.sub(r"\d+", "", metin)
    return metin

haber_basliklari = df.iloc[:, 1].apply(temizle_metin)
print("\nHaber Başlıklarının İlk 5 Satırı (Temizlenmiş):")
print(haber_basliklari.head())

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(haber_basliklari).toarray()

print("\nTF-IDF ile öznitelik matrisi oluşturuldu:")
print(f"Öznitelik matrisi boyutu: {X.shape}")

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['neutral'])

threshold = 0.0005
selector = VarianceThreshold(threshold=threshold)
X_high_variance = selector.fit_transform(X)

print("\nVaryans Eşiği Sonrası Öznitelik Matrisi:")
print(f"Öznitelik matrisi boyutu: {X_high_variance.shape}")

pca = PCA(n_components=30, random_state=42)
X_pca = pca.fit_transform(X_high_variance)
explained_variance_ratio = pca.explained_variance_ratio_.sum()
print(f"\nPCA ile açıklanan toplam varyans oranı: {explained_variance_ratio:.4f}")

def create_transformer_classifier(input_dim, num_classes, num_heads=1, ff_dim=32):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Embedding(input_dim=5000, output_dim=32)(inputs)
    x = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=32)(x, x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def calculate_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    specificity_per_class = []
    for i in range(len(cm)):
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        fp = np.sum(cm[:, i]) - cm[i, i]
        if tn + fp > 0:
            specificity = tn / (tn + fp)
        else:
            specificity = 0
        specificity_per_class.append(specificity)
    return np.mean(specificity_per_class)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results_no_pca = []
results_pca = []

for fold, (train_idx, test_idx) in enumerate(kfold.split(X_high_variance, y), start=1):
    print(f"Fold {fold} başlıyor...")
    
    X_train_no_pca, X_test_no_pca = X_high_variance[train_idx], X_high_variance[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    X_train_pca, X_test_pca = X_pca[train_idx], X_pca[test_idx]
    
    model_no_pca = create_transformer_classifier(input_dim=X_train_no_pca.shape[1], num_classes=len(np.unique(y)))
    model_no_pca.fit(X_train_no_pca, y_train, epochs=1, batch_size=16, verbose=1)
    y_pred_no_pca = np.argmax(model_no_pca.predict(X_test_no_pca), axis=1)
    specificity_no_pca = calculate_specificity(y_test, y_pred_no_pca)
    auc_no_pca = roc_auc_score(y_test, model_no_pca.predict(X_test_no_pca), multi_class='ovr')
    results_no_pca.append([
        fold,
        accuracy_score(y_test, y_pred_no_pca),
        precision_score(y_test, y_pred_no_pca, average='weighted', zero_division=0),
        specificity_no_pca,
        f1_score(y_test, y_pred_no_pca, average='weighted', zero_division=0),
        recall_score(y_test, y_pred_no_pca, average='weighted', zero_division=0),
        auc_no_pca
    ])

    model_pca = create_transformer_classifier(input_dim=X_train_pca.shape[1], num_classes=len(np.unique(y)))
    model_pca.fit(X_train_pca, y_train, epochs=1, batch_size=16, verbose=1)
    y_pred_pca = np.argmax(model_pca.predict(X_test_pca), axis=1)
    specificity_pca = calculate_specificity(y_test, y_pred_pca)
    auc_pca = roc_auc_score(y_test, model_pca.predict(X_test_pca), multi_class='ovr')
    results_pca.append([
        fold,
        accuracy_score(y_test, y_pred_pca),
        precision_score(y_test, y_pred_pca, average='weighted', zero_division=0),
        specificity_pca,
        f1_score(y_test, y_pred_pca, average='weighted', zero_division=0),
        recall_score(y_test, y_pred_pca, average='weighted', zero_division=0),
        auc_pca
    ])

columns = ['Fold', 'Accuracy (%)', 'Precision (%)', 'Specificity (%)', 'F1-Score (%)', 'Sensitivity (%)', 'AUC (%)']
results_no_pca_df = pd.DataFrame(results_no_pca, columns=columns)
results_pca_df = pd.DataFrame(results_pca, columns=columns)

results_no_pca_df.loc['Average'] = results_no_pca_df.mean(numeric_only=True)
results_pca_df.loc['Average'] = results_pca_df.mean(numeric_only=True)

print("\nPCA Öncesi Sonuçlar:")
print(results_no_pca_df)

print("\nPCA Sonrası Sonuçlar:")
print(results_pca_df)


########################################################################################
# Hiyerarşik Dikkat Ağı (HAN) — Varyans Eşiği ve PCA Karşılaştırması
########################################################################################

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import re
import string

from google.colab import files
print("Lütfen veri setini yükleyin (ör. all-data.csv):")
uploaded = files.upload()

data_file_name = list(uploaded.keys())[0]
print("Yüklenen veri setinin adı:", data_file_name)

df = pd.read_csv(data_file_name, encoding='ISO-8859-1')
print("\nVeri setinin ilk 5 satırı:")
print(df.head())

neutral_sayisi = df[df['neutral'] == 'neutral'].shape[0]
positive_sayisi = df[df['neutral'] == 'positive'].shape[0]
negative_sayisi = df[df['neutral'] == 'negative'].shape[0]

print("\nDuygu etiketlerinin sayısı:")
print(f"Nötr: {neutral_sayisi}")
print(f"Pozitif: {positive_sayisi}")
print(f"Negatif: {negative_sayisi}")

def temizle_metin(metin):
    metin = metin.lower()
    metin = re.sub(f"[{re.escape(string.punctuation)}]", "", metin)
    metin = re.sub(r"\d+", "", metin)
    return metin

haber_basliklari = df.iloc[:, 1].apply(temizle_metin)
print("\nHaber Başlıklarının İlk 5 Satırı (Temizlenmiş):")
print(haber_basliklari.head())

tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # En sık kullanılan 5000 kelimeyi seç
X = tfidf_vectorizer.fit_transform(haber_basliklari)

print("\nTF-IDF ile öznitelik matrisi oluşturuldu:")
print(f"Öznitelik matrisi boyutu: {X.shape}")

print("\nTF-IDF Matrisindeki Örnek Kelimeler:")
print(tfidf_vectorizer.get_feature_names_out()[:10])

threshold = 0.0005  # Varyans için eşik değeri
selector = VarianceThreshold(threshold=threshold)
X_high_variance = selector.fit_transform(X.toarray())  # Düşük varyanslı özellikleri çıkar

print("\nVaryans Eşiği Sonrası Öznitelik Matrisi:")
print(f"Öznitelik matrisi boyutu: {X_high_variance.shape}")

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['neutral'])  # Etiketleri sayısal hale çevir

pca = PCA(n_components=100, random_state=42)  # 100 bileşene indirgeniyor
X_pca = pca.fit_transform(X_high_variance)  # PCA için düşük varyanslı matris kullanılır

explained_variance_ratio = pca.explained_variance_ratio_.sum()
print(f"\nPCA ile açıklanan toplam varyans oranı: {explained_variance_ratio:.4f}")

def create_han_model(input_dim, num_classes):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Embedding(input_dim=5000, output_dim=128)(inputs)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
    x = tf.keras.layers.Attention()([x, x])
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results_no_pca = []
results_pca = []

for fold, (train_idx, test_idx) in enumerate(kfold.split(X_high_variance, y), start=1):
    print(f"Fold {fold} başlıyor...")
    
    X_train_no_pca, X_test_no_pca = X_high_variance[train_idx], X_high_variance[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    X_train_pca, X_test_pca = X_pca[train_idx], X_pca[test_idx]
    
    model_no_pca = create_han_model(X_train_no_pca.shape[1], len(np.unique(y)))
    model_no_pca.fit(X_train_no_pca, y_train, epochs=1, batch_size=16, verbose=1)
    y_pred_no_pca = np.argmax(model_no_pca.predict(X_test_no_pca), axis=1)
    auc_no_pca = roc_auc_score(y_test, model_no_pca.predict(X_test_no_pca), multi_class='ovr')
    results_no_pca.append([
        fold,
        accuracy_score(y_test, y_pred_no_pca),
        precision_score(y_test, y_pred_no_pca, average='weighted'),
        recall_score(y_test, y_pred_no_pca, average='weighted'),
        f1_score(y_test, y_pred_no_pca, average='weighted'),
        auc_no_pca
    ])

    model_pca = create_han_model(X_train_pca.shape[1], len(np.unique(y)))
    model_pca.fit(X_train_pca, y_train, epochs=1, batch_size=16, verbose=1)
    y_pred_pca = np.argmax(model_pca.predict(X_test_pca), axis=1)
    auc_pca = roc_auc_score(y_test, model_pca.predict(X_test_pca), multi_class='ovr')
    results_pca.append([
        fold,
        accuracy_score(y_test, y_pred_pca),
        precision_score(y_test, y_pred_pca, average='weighted'),
        recall_score(y_test, y_pred_pca, average='weighted'),
        f1_score(y_test, y_pred_pca, average='weighted'),
        auc_pca
    ])

columns = ['Fold', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)', 'AUC (%)']
results_no_pca_df = pd.DataFrame(results_no_pca, columns=columns)
results_pca_df = pd.DataFrame(results_pca, columns=columns)

results_no_pca_df.loc['Average'] = results_no_pca_df.mean(numeric_only=True)
results_pca_df.loc['Average'] = results_pca_df.mean(numeric_only=True)

print("\nPCA Öncesi Sonuçlar:")
print(results_no_pca_df)

print("\nPCA Sonrası Sonuçlar:")
print(results_pca_df)
