"""Makine Öğrenmesi Deneyleri

Naive Bayes, SVM, Lojistik Regresyon, Rastgele Orman ve KNN modellerinin karşılaştırmalı deneylerini içerir.

Bu dosya, çalışmada kullanılan bağımsız deney bloklarını model grubuna göre bir araya getirir.
Bölümler Google Colab üzerinde ayrı deneyler olarak çalıştırılmak üzere hazırlanmıştır;
bu nedenle veri yükleme ve bazı import adımları bölümler arasında tekrar eder.
"""


########################################################################################
# Naive Bayes — PCA Karşılaştırması
########################################################################################

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
import re
import string

# Çok sınıflı özgüllük hesaplama fonksiyonu
def calculate_specificity_multiclass(confusion_matrix):
    """
    Çok sınıflı özgüllük hesaplar.
    Her sınıf için özgüllüğü bulur ve bunların ortalamasını döner.
    """
    num_classes = confusion_matrix.shape[0]
    specificity_list = []
    
    for class_idx in range(num_classes):
        true_negatives = np.sum(np.delete(np.delete(confusion_matrix, class_idx, axis=0), class_idx, axis=1))
        false_positives = np.sum(np.delete(confusion_matrix[:, class_idx], class_idx))
        
        if (true_negatives + false_positives) > 0:
            specificity = true_negatives / (true_negatives + false_positives)
        else:
            specificity = 0  # Bölme hatasını önlemek için
        specificity_list.append(specificity)
    
    return np.mean(specificity_list)

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

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(haber_basliklari)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['neutral'])

pca = PCA(n_components=100, random_state=42)
X_pca = pca.fit_transform(X.toarray())

scaler = MinMaxScaler()
X_pca = scaler.fit_transform(X_pca)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results_pca = []
results_no_pca = []

for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y), start=1):
    # PCA öncesi veriler
    X_train_no_pca, X_test_no_pca = X[train_idx], X[test_idx]
    X_train_pca, X_test_pca = X_pca[train_idx], X_pca[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    nb_model = MultinomialNB()
    
    nb_model.fit(X_train_no_pca, y_train)
    y_pred_no_pca = nb_model.predict(X_test_no_pca)
    y_prob_no_pca = nb_model.predict_proba(X_test_no_pca)
    
    nb_model.fit(X_train_pca, y_train)
    y_pred_pca = nb_model.predict(X_test_pca)
    y_prob_pca = nb_model.predict_proba(X_test_pca)
    
    cm_no_pca = confusion_matrix(y_test, y_pred_no_pca)
    accuracy_no_pca = accuracy_score(y_test, y_pred_no_pca)
    precision_no_pca = precision_score(y_test, y_pred_no_pca, average='weighted', zero_division=0)
    recall_no_pca = recall_score(y_test, y_pred_no_pca, average='weighted', zero_division=0)
    f1_no_pca = f1_score(y_test, y_pred_no_pca, average='weighted', zero_division=0)
    auc_no_pca = roc_auc_score(y_test, y_prob_no_pca, multi_class='ovr')
    specificity_no_pca = calculate_specificity_multiclass(cm_no_pca)
    
    results_no_pca.append([fold, accuracy_no_pca, precision_no_pca, recall_no_pca, f1_no_pca, specificity_no_pca, auc_no_pca])
    
    cm_pca = confusion_matrix(y_test, y_pred_pca)
    accuracy_pca = accuracy_score(y_test, y_pred_pca)
    precision_pca = precision_score(y_test, y_pred_pca, average='weighted', zero_division=0)
    recall_pca = recall_score(y_test, y_pred_pca, average='weighted', zero_division=0)
    f1_pca = f1_score(y_test, y_pred_pca, average='weighted', zero_division=0)
    auc_pca = roc_auc_score(y_test, y_prob_pca, multi_class='ovr')
    specificity_pca = calculate_specificity_multiclass(cm_pca)
    
    results_pca.append([fold, accuracy_pca, precision_pca, recall_pca, f1_pca, specificity_pca, auc_pca])

columns = ['Fold', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)', 'Specificity (%)', 'AUC']
results_no_pca_df = pd.DataFrame(results_no_pca, columns=columns)
results_pca_df = pd.DataFrame(results_pca, columns=columns)

results_no_pca_df.loc['Average'] = results_no_pca_df.mean(numeric_only=True)
results_pca_df.loc['Average'] = results_pca_df.mean(numeric_only=True)

print("\nPCA Öncesi Sonuçlar:")
print(results_no_pca_df)

print("\nPCA Sonrası Sonuçlar:")
print(results_pca_df)


########################################################################################
# Destek Vektör Makinesi (SVM) — PCA Karşılaştırması
########################################################################################

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
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

pca = PCA(n_components=100, random_state=42)
X_pca = pca.fit_transform(X.toarray())

explained_variance_ratio = pca.explained_variance_ratio_.sum()
print(f"\nPCA ile açıklanan toplam varyans oranı: {explained_variance_ratio:.4f}")

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results_pca = []
results_no_pca = []

for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y), start=1):
    # PCA öncesi veriler
    X_train_no_pca, X_test_no_pca = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    X_train_pca, X_test_pca = X_pca[train_idx], X_pca[test_idx]
    
    svm_model = SVC(kernel='linear', probability=True, random_state=42)
    
    svm_model.fit(X_train_no_pca, y_train)
    y_pred_no_pca = svm_model.predict(X_test_no_pca)
    y_prob_no_pca = svm_model.predict_proba(X_test_no_pca)
    cm_no_pca = confusion_matrix(y_test, y_pred_no_pca)
    specificity_no_pca = cm_no_pca[0, 0] / (cm_no_pca[0, 0] + cm_no_pca[0, 1]) if cm_no_pca.shape[0] > 1 else 0
    results_no_pca.append([
        fold,
        accuracy_score(y_test, y_pred_no_pca),
        precision_score(y_test, y_pred_no_pca, average='weighted', zero_division=0),
        recall_score(y_test, y_pred_no_pca, average='weighted', zero_division=0),
        f1_score(y_test, y_pred_no_pca, average='weighted', zero_division=0),
        specificity_no_pca,
        roc_auc_score(y_test, y_prob_no_pca, multi_class='ovr'),
    ])
    
    svm_model.fit(X_train_pca, y_train)
    y_pred_pca = svm_model.predict(X_test_pca)
    y_prob_pca = svm_model.predict_proba(X_test_pca)
    cm_pca = confusion_matrix(y_test, y_pred_pca)
    specificity_pca = cm_pca[0, 0] / (cm_pca[0, 0] + cm_pca[0, 1]) if cm_pca.shape[0] > 1 else 0
    results_pca.append([
        fold,
        accuracy_score(y_test, y_pred_pca),
        precision_score(y_test, y_pred_pca, average='weighted', zero_division=0),
        recall_score(y_test, y_pred_pca, average='weighted', zero_division=0),
        f1_score(y_test, y_pred_pca, average='weighted', zero_division=0),
        specificity_pca,
        roc_auc_score(y_test, y_prob_pca, multi_class='ovr'),
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
# Lojistik Regresyon — PCA Karşılaştırması
########################################################################################

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
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

pca = PCA(n_components=100, random_state=42)
X_pca = pca.fit_transform(X.toarray())

explained_variance_ratio = pca.explained_variance_ratio_.sum()
print(f"\nPCA ile açıklanan toplam varyans oranı: {explained_variance_ratio:.4f}")

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results_pca = []
results_no_pca = []

for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y), start=1):
    X_train_no_pca, X_test_no_pca = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    X_train_pca, X_test_pca = X_pca[train_idx], X_pca[test_idx]
    
    logreg_model = LogisticRegression(max_iter=1000, random_state=42)
    
    logreg_model.fit(X_train_no_pca, y_train)
    y_pred_no_pca = logreg_model.predict(X_test_no_pca)
    y_prob_no_pca = logreg_model.predict_proba(X_test_no_pca)
    cm_no_pca = confusion_matrix(y_test, y_pred_no_pca)
    specificity_no_pca = cm_no_pca[0, 0] / (cm_no_pca[0, 0] + cm_no_pca[0, 1]) if cm_no_pca.shape[0] > 1 else 0
    results_no_pca.append([
        fold,
        accuracy_score(y_test, y_pred_no_pca),
        precision_score(y_test, y_pred_no_pca, average='weighted', zero_division=0),
        recall_score(y_test, y_pred_no_pca, average='weighted', zero_division=0),
        f1_score(y_test, y_pred_no_pca, average='weighted', zero_division=0),
        specificity_no_pca,
        roc_auc_score(y_test, y_prob_no_pca, multi_class='ovr'),
    ])
    
    logreg_model.fit(X_train_pca, y_train)
    y_pred_pca = logreg_model.predict(X_test_pca)
    y_prob_pca = logreg_model.predict_proba(X_test_pca)
    cm_pca = confusion_matrix(y_test, y_pred_pca)
    specificity_pca = cm_pca[0, 0] / (cm_pca[0, 0] + cm_pca[0, 1]) if cm_pca.shape[0] > 1 else 0
    results_pca.append([
        fold,
        accuracy_score(y_test, y_pred_pca),
        precision_score(y_test, y_pred_pca, average='weighted', zero_division=0),
        recall_score(y_test, y_pred_pca, average='weighted', zero_division=0),
        f1_score(y_test, y_pred_pca, average='weighted', zero_division=0),
        specificity_pca,
        roc_auc_score(y_test, y_prob_pca, multi_class='ovr'),
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
# Rastgele Orman — PCA Karşılaştırması
########################################################################################

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
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

pca = PCA(n_components=100, random_state=42)
X_pca = pca.fit_transform(X.toarray())

explained_variance_ratio = pca.explained_variance_ratio_.sum()
print(f"\nPCA ile açıklanan toplam varyans oranı: {explained_variance_ratio:.4f}")

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results_pca = []
results_no_pca = []

for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y), start=1):
    X_train_no_pca, X_test_no_pca = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    X_train_pca, X_test_pca = X_pca[train_idx], X_pca[test_idx]
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    rf_model.fit(X_train_no_pca, y_train)
    y_pred_no_pca = rf_model.predict(X_test_no_pca)
    y_prob_no_pca = rf_model.predict_proba(X_test_no_pca)
    cm_no_pca = confusion_matrix(y_test, y_pred_no_pca)
    specificity_no_pca = cm_no_pca[0, 0] / (cm_no_pca[0, 0] + cm_no_pca[0, 1]) if cm_no_pca.shape[0] > 1 else 0
    results_no_pca.append([
        fold,
        accuracy_score(y_test, y_pred_no_pca),
        precision_score(y_test, y_pred_no_pca, average='weighted', zero_division=0),
        recall_score(y_test, y_pred_no_pca, average='weighted', zero_division=0),
        f1_score(y_test, y_pred_no_pca, average='weighted', zero_division=0),
        specificity_no_pca,
        roc_auc_score(y_test, y_prob_no_pca, multi_class='ovr'),
    ])
    
    rf_model.fit(X_train_pca, y_train)
    y_pred_pca = rf_model.predict(X_test_pca)
    y_prob_pca = rf_model.predict_proba(X_test_pca)
    cm_pca = confusion_matrix(y_test, y_pred_pca)
    specificity_pca = cm_pca[0, 0] / (cm_pca[0, 0] + cm_pca[0, 1]) if cm_pca.shape[0] > 1 else 0
    results_pca.append([
        fold,
        accuracy_score(y_test, y_pred_pca),
        precision_score(y_test, y_pred_pca, average='weighted', zero_division=0),
        recall_score(y_test, y_pred_pca, average='weighted', zero_division=0),
        f1_score(y_test, y_pred_pca, average='weighted', zero_division=0),
        specificity_pca,
        roc_auc_score(y_test, y_prob_pca, multi_class='ovr'),
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
# K-En Yakın Komşu (KNN) — PCA Karşılaştırması
########################################################################################

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
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

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['neutral'])

pca = PCA(n_components=100, random_state=42)
X_pca = pca.fit_transform(X.toarray())

explained_variance_ratio = pca.explained_variance_ratio_.sum()
print(f"\nPCA ile açıklanan toplam varyans oranı: {explained_variance_ratio:.4f}")

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results_pca = []
results_no_pca = []

for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y), start=1):
    X_train_no_pca, X_test_no_pca = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    X_train_pca, X_test_pca = X_pca[train_idx], X_pca[test_idx]
    
    knn_model = KNeighborsClassifier(n_neighbors=5)
    
    knn_model.fit(X_train_no_pca, y_train)
    y_pred_no_pca = knn_model.predict(X_test_no_pca)
    y_prob_no_pca = knn_model.predict_proba(X_test_no_pca)
    auc_no_pca = roc_auc_score(y_test, y_prob_no_pca, multi_class='ovr')  # AUC hesaplama
    
    knn_model.fit(X_train_pca, y_train)
    y_pred_pca = knn_model.predict(X_test_pca)
    y_prob_pca = knn_model.predict_proba(X_test_pca)
    auc_pca = roc_auc_score(y_test, y_prob_pca, multi_class='ovr')  # AUC hesaplama
    
    cm_no_pca = confusion_matrix(y_test, y_pred_no_pca)
    specificity_no_pca = cm_no_pca[0, 0] / (cm_no_pca[0, 0] + cm_no_pca[0, 1]) if cm_no_pca.shape[0] > 1 else 0
    results_no_pca.append([
        fold,
        accuracy_score(y_test, y_pred_no_pca),
        precision_score(y_test, y_pred_no_pca, average='weighted', zero_division=0),
        recall_score(y_test, y_pred_no_pca, average='weighted', zero_division=0),
        f1_score(y_test, y_pred_no_pca, average='weighted', zero_division=0),
        specificity_no_pca,
        auc_no_pca
    ])
    
    cm_pca = confusion_matrix(y_test, y_pred_pca)
    specificity_pca = cm_pca[0, 0] / (cm_pca[0, 0] + cm_pca[0, 1]) if cm_pca.shape[0] > 1 else 0
    results_pca.append([
        fold,
        accuracy_score(y_test, y_pred_pca),
        precision_score(y_test, y_pred_pca, average='weighted', zero_division=0),
        recall_score(y_test, y_pred_pca, average='weighted', zero_division=0),
        f1_score(y_test, y_pred_pca, average='weighted', zero_division=0),
        specificity_pca,
        auc_pca
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
# Naive Bayes — Varyans Eşiği ve PCA Karşılaştırması
########################################################################################

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
import re
import string

def calculate_specificity_multiclass(cm):
    specificity_list = []
    for i in range(len(cm)):
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity_list.append(specificity)
    return np.mean(specificity_list)

from google.colab import files
print("Lütfen veri setini yükleyin (ör. all-data.csv):")
uploaded = files.upload()

data_file_name = list(uploaded.keys())[0]
print("Yüklenen veri setinin adı:", data_file_name)

# Veri setini pandas ile oku
df = pd.read_csv(data_file_name, encoding='ISO-8859-1')
print("\nVeri setinin ilk 5 satırı:")
print(df.head())

def temizle_metin(metin):
    metin = metin.lower()
    metin = re.sub(f"[{re.escape(string.punctuation)}]", "", metin)
    metin = re.sub(r"\d+", "", metin)
    return metin

haber_basliklari = df.iloc[:, 1].apply(temizle_metin)

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(haber_basliklari)

threshold = 0.0005
var_thresh = VarianceThreshold(threshold=threshold)
X_var_thresh = var_thresh.fit_transform(X.toarray())
print(f"\nVaryans Eşiği Sonrası Öznitelik Matrisi Boyutu: {X_var_thresh.shape}")

pca = PCA(n_components=100, random_state=42)
X_pca = pca.fit_transform(X_var_thresh)

scaler = MinMaxScaler()
X_pca = scaler.fit_transform(X_pca)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['neutral'])

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results_pca = []
results_no_pca = []

for fold, (train_idx, test_idx) in enumerate(kfold.split(X_var_thresh, y), start=1):
    # PCA öncesi veriler
    X_train_no_pca, X_test_no_pca = X_var_thresh[train_idx], X_var_thresh[test_idx]
    X_train_pca, X_test_pca = X_pca[train_idx], X_pca[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    nb_model = MultinomialNB()
    
    nb_model.fit(X_train_no_pca, y_train)
    y_pred_no_pca = nb_model.predict(X_test_no_pca)
    y_prob_no_pca = nb_model.predict_proba(X_test_no_pca)
    
    nb_model.fit(X_train_pca, y_train)
    y_pred_pca = nb_model.predict(X_test_pca)
    y_prob_pca = nb_model.predict_proba(X_test_pca)
    
    cm_no_pca = confusion_matrix(y_test, y_pred_no_pca)
    accuracy_no_pca = accuracy_score(y_test, y_pred_no_pca)
    precision_no_pca = precision_score(y_test, y_pred_no_pca, average='weighted', zero_division=0)
    recall_no_pca = recall_score(y_test, y_pred_no_pca, average='weighted', zero_division=0)
    f1_no_pca = f1_score(y_test, y_pred_no_pca, average='weighted', zero_division=0)
    auc_no_pca = roc_auc_score(y_test, y_prob_no_pca, multi_class='ovr')
    specificity_no_pca = calculate_specificity_multiclass(cm_no_pca)
    
    results_no_pca.append([fold, accuracy_no_pca, precision_no_pca, recall_no_pca, f1_no_pca, specificity_no_pca, auc_no_pca])
    
    cm_pca = confusion_matrix(y_test, y_pred_pca)
    accuracy_pca = accuracy_score(y_test, y_pred_pca)
    precision_pca = precision_score(y_test, y_pred_pca, average='weighted', zero_division=0)
    recall_pca = recall_score(y_test, y_pred_pca, average='weighted', zero_division=0)
    f1_pca = f1_score(y_test, y_pred_pca, average='weighted', zero_division=0)
    auc_pca = roc_auc_score(y_test, y_prob_pca, multi_class='ovr')
    specificity_pca = calculate_specificity_multiclass(cm_pca)
    
    results_pca.append([fold, accuracy_pca, precision_pca, recall_pca, f1_pca, specificity_pca, auc_pca])

columns = ['Fold', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)', 'Specificity (%)', 'AUC']
results_no_pca_df = pd.DataFrame(results_no_pca, columns=columns)
results_pca_df = pd.DataFrame(results_pca, columns=columns)

results_no_pca_df.loc['Average'] = results_no_pca_df.mean(numeric_only=True)
results_pca_df.loc['Average'] = results_pca_df.mean(numeric_only=True)

print("\nPCA Öncesi Sonuçlar:")
print(results_no_pca_df)

print("\nPCA Sonrası Sonuçlar:")
print(results_pca_df)


########################################################################################
# Destek Vektör Makinesi (SVM) — Varyans Eşiği ve PCA Karşılaştırması
########################################################################################

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
import re
import string

from google.colab import files
print("Lütfen veri setini yükleyin (ör. all-data.csv):")
uploaded = files.upload()

data_file_name = list(uploaded.keys())[0]
print("Yüklenen veri setinin adı:", data_file_name)

# Veri setini pandas ile oku
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

threshold = 0.0005
var_thresh = VarianceThreshold(threshold=threshold)
X_var_thresh = var_thresh.fit_transform(X.toarray())
print(f"\nVaryans Eşiği Sonrası Öznitelik Matrisi Boyutu: {X_var_thresh.shape}")

pca = PCA(n_components=min(100, X_var_thresh.shape[1]), random_state=42)
X_pca = pca.fit_transform(X_var_thresh)

explained_variance_ratio = pca.explained_variance_ratio_.sum()
print(f"\nPCA ile açıklanan toplam varyans oranı: {explained_variance_ratio:.4f}")

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['neutral'])

# K-Fold Çapraz Doğrulama
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results_pca = []
results_no_pca = []

for fold, (train_idx, test_idx) in enumerate(kfold.split(X_var_thresh, y), start=1):
    X_train_no_pca, X_test_no_pca = X_var_thresh[train_idx], X_var_thresh[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    X_train_pca, X_test_pca = X_pca[train_idx], X_pca[test_idx]
    
    svm_model = SVC(kernel='linear', probability=True, random_state=42)
    
    svm_model.fit(X_train_no_pca, y_train)
    y_pred_no_pca = svm_model.predict(X_test_no_pca)
    y_prob_no_pca = svm_model.predict_proba(X_test_no_pca)
    cm_no_pca = confusion_matrix(y_test, y_pred_no_pca)
    specificity_no_pca = cm_no_pca[0, 0] / (cm_no_pca[0, 0] + cm_no_pca[0, 1]) if cm_no_pca.shape[0] > 1 else 0
    results_no_pca.append([
        fold,
        accuracy_score(y_test, y_pred_no_pca),
        precision_score(y_test, y_pred_no_pca, average='weighted', zero_division=0),
        recall_score(y_test, y_pred_no_pca, average='weighted', zero_division=0),
        f1_score(y_test, y_pred_no_pca, average='weighted', zero_division=0),
        specificity_no_pca,
        roc_auc_score(y_test, y_prob_no_pca, multi_class='ovr'),
    ])
    
    svm_model.fit(X_train_pca, y_train)
    y_pred_pca = svm_model.predict(X_test_pca)
    y_prob_pca = svm_model.predict_proba(X_test_pca)
    cm_pca = confusion_matrix(y_test, y_pred_pca)
    specificity_pca = cm_pca[0, 0] / (cm_pca[0, 0] + cm_pca[0, 1]) if cm_pca.shape[0] > 1 else 0
    results_pca.append([
        fold,
        accuracy_score(y_test, y_pred_pca),
        precision_score(y_test, y_pred_pca, average='weighted', zero_division=0),
        recall_score(y_test, y_pred_pca, average='weighted', zero_division=0),
        f1_score(y_test, y_pred_pca, average='weighted', zero_division=0),
        specificity_pca,
        roc_auc_score(y_test, y_prob_pca, multi_class='ovr'),
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
# Lojistik Regresyon — Varyans Eşiği ve PCA Karşılaştırması
########################################################################################

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
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

threshold = 0.0005
var_thresh = VarianceThreshold(threshold=threshold)
X_var_thresh = var_thresh.fit_transform(X.toarray())
print(f"\nVaryans Eşiği Sonrası Öznitelik Matrisi Boyutu: {X_var_thresh.shape}")

pca = PCA(n_components=100, random_state=42)
X_pca = pca.fit_transform(X_var_thresh)

explained_variance_ratio = pca.explained_variance_ratio_.sum()
print(f"\nPCA ile açıklanan toplam varyans oranı: {explained_variance_ratio:.4f}")

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['neutral'])

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results_pca = []
results_no_pca = []

for fold, (train_idx, test_idx) in enumerate(kfold.split(X_var_thresh, y), start=1):
    X_train_no_pca, X_test_no_pca = X_var_thresh[train_idx], X_var_thresh[test_idx]
    X_train_pca, X_test_pca = X_pca[train_idx], X_pca[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    logreg_model = LogisticRegression(max_iter=1000, random_state=42)
    
    logreg_model.fit(X_train_no_pca, y_train)
    y_pred_no_pca = logreg_model.predict(X_test_no_pca)
    y_prob_no_pca = logreg_model.predict_proba(X_test_no_pca)
    cm_no_pca = confusion_matrix(y_test, y_pred_no_pca)
    specificity_no_pca = cm_no_pca[0, 0] / (cm_no_pca[0, 0] + cm_no_pca[0, 1]) if cm_no_pca.shape[0] > 1 else 0
    results_no_pca.append([
        fold,
        accuracy_score(y_test, y_pred_no_pca),
        precision_score(y_test, y_pred_no_pca, average='weighted', zero_division=0),
        recall_score(y_test, y_pred_no_pca, average='weighted', zero_division=0),
        f1_score(y_test, y_pred_no_pca, average='weighted', zero_division=0),
        specificity_no_pca,
        roc_auc_score(y_test, y_prob_no_pca, multi_class='ovr'),
    ])
    
    logreg_model.fit(X_train_pca, y_train)
    y_pred_pca = logreg_model.predict(X_test_pca)
    y_prob_pca = logreg_model.predict_proba(X_test_pca)
    cm_pca = confusion_matrix(y_test, y_pred_pca)
    specificity_pca = cm_pca[0, 0] / (cm_pca[0, 0] + cm_pca[0, 1]) if cm_pca.shape[0] > 1 else 0
    results_pca.append([
        fold,
        accuracy_score(y_test, y_pred_pca),
        precision_score(y_test, y_pred_pca, average='weighted', zero_division=0),
        recall_score(y_test, y_pred_pca, average='weighted', zero_division=0),
        f1_score(y_test, y_pred_pca, average='weighted', zero_division=0),
        specificity_pca,
        roc_auc_score(y_test, y_prob_pca, multi_class='ovr'),
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
# Rastgele Orman — Varyans Eşiği ve PCA Karşılaştırması
########################################################################################

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
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
X_var_thresh = var_thresh.fit_transform(X.toarray())
print(f"\nVaryans Eşiği Sonrası Öznitelik Matrisi Boyutu: {X_var_thresh.shape}")

pca = PCA(n_components=100, random_state=42)
X_pca = pca.fit_transform(X_var_thresh)

explained_variance_ratio = pca.explained_variance_ratio_.sum()
print(f"\nPCA ile açıklanan toplam varyans oranı: {explained_variance_ratio:.4f}")

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['neutral'])

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results_pca = []
results_no_pca = []

for fold, (train_idx, test_idx) in enumerate(kfold.split(X_var_thresh, y), start=1):
    X_train_no_pca, X_test_no_pca = X_var_thresh[train_idx], X_var_thresh[test_idx]
    X_train_pca, X_test_pca = X_pca[train_idx], X_pca[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    rf_model.fit(X_train_no_pca, y_train)
    y_pred_no_pca = rf_model.predict(X_test_no_pca)
    y_prob_no_pca = rf_model.predict_proba(X_test_no_pca)
    cm_no_pca = confusion_matrix(y_test, y_pred_no_pca)
    specificity_no_pca = cm_no_pca[0, 0] / (cm_no_pca[0, 0] + cm_no_pca[0, 1]) if cm_no_pca.shape[0] > 1 else 0
    results_no_pca.append([
        fold,
        accuracy_score(y_test, y_pred_no_pca),
        precision_score(y_test, y_pred_no_pca, average='weighted', zero_division=0),
        recall_score(y_test, y_pred_no_pca, average='weighted', zero_division=0),
        f1_score(y_test, y_pred_no_pca, average='weighted', zero_division=0),
        specificity_no_pca,
        roc_auc_score(y_test, y_prob_no_pca, multi_class='ovr'),
    ])
    
    rf_model.fit(X_train_pca, y_train)
    y_pred_pca = rf_model.predict(X_test_pca)
    y_prob_pca = rf_model.predict_proba(X_test_pca)
    cm_pca = confusion_matrix(y_test, y_pred_pca)
    specificity_pca = cm_pca[0, 0] / (cm_pca[0, 0] + cm_pca[0, 1]) if cm_pca.shape[0] > 1 else 0
    results_pca.append([
        fold,
        accuracy_score(y_test, y_pred_pca),
        precision_score(y_test, y_pred_pca, average='weighted', zero_division=0),
        recall_score(y_test, y_pred_pca, average='weighted', zero_division=0),
        f1_score(y_test, y_pred_pca, average='weighted', zero_division=0),
        specificity_pca,
        roc_auc_score(y_test, y_prob_pca, multi_class='ovr'),
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
# K-En Yakın Komşu (KNN) — Varyans Eşiği ve PCA Karşılaştırması
########################################################################################

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
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
X_var_thresh = var_thresh.fit_transform(X.toarray())
print(f"\nVaryans Eşiği Sonrası Öznitelik Matrisi Boyutu: {X_var_thresh.shape}")

pca = PCA(n_components=100, random_state=42)
X_pca = pca.fit_transform(X_var_thresh)

explained_variance_ratio = pca.explained_variance_ratio_.sum()
print(f"\nPCA ile açıklanan toplam varyans oranı: {explained_variance_ratio:.4f}")

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['neutral'])

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results_pca = []
results_no_pca = []

for fold, (train_idx, test_idx) in enumerate(kfold.split(X_var_thresh, y), start=1):
    X_train_no_pca, X_test_no_pca = X_var_thresh[train_idx], X_var_thresh[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    X_train_pca, X_test_pca = X_pca[train_idx], X_pca[test_idx]
    
    knn_model = KNeighborsClassifier(n_neighbors=5)
    
    knn_model.fit(X_train_no_pca, y_train)
    y_pred_no_pca = knn_model.predict(X_test_no_pca)
    y_prob_no_pca = knn_model.predict_proba(X_test_no_pca)
    auc_no_pca = roc_auc_score(y_test, y_prob_no_pca, multi_class="ovr")
    cm_no_pca = confusion_matrix(y_test, y_pred_no_pca)
    specificity_no_pca = cm_no_pca[0, 0] / (cm_no_pca[0, 0] + cm_no_pca[0, 1]) if cm_no_pca.shape[0] > 1 else 0
    results_no_pca.append([
        fold,
        accuracy_score(y_test, y_pred_no_pca),
        precision_score(y_test, y_pred_no_pca, average='weighted', zero_division=0),
        recall_score(y_test, y_pred_no_pca, average='weighted', zero_division=0),
        f1_score(y_test, y_pred_no_pca, average='weighted', zero_division=0),
        specificity_no_pca,
        auc_no_pca
    ])
    
    knn_model.fit(X_train_pca, y_train)
    y_pred_pca = knn_model.predict(X_test_pca)
    y_prob_pca = knn_model.predict_proba(X_test_pca)
    auc_pca = roc_auc_score(y_test, y_prob_pca, multi_class="ovr")
    cm_pca = confusion_matrix(y_test, y_pred_pca)
    specificity_pca = cm_pca[0, 0] / (cm_pca[0, 0] + cm_pca[0, 1]) if cm_pca.shape[0] > 1 else 0
    results_pca.append([
        fold,
        accuracy_score(y_test, y_pred_pca),
        precision_score(y_test, y_pred_pca, average='weighted', zero_division=0),
        recall_score(y_test, y_pred_pca, average='weighted', zero_division=0),
        f1_score(y_test, y_pred_pca, average='weighted', zero_division=0),
        specificity_pca,
        auc_pca
    ])

columns = ['Katlama (Fold)', 'Doğruluk (Accuracy)', 'Kesinlik (Precision)', 'Duyarlılık (Recall)', 'F1-Skoru (F1-Score)', 'Özgüllük (Specificity)', 'AUC']
results_no_pca_df = pd.DataFrame(results_no_pca, columns=columns)
results_pca_df = pd.DataFrame(results_pca, columns=columns)

results_no_pca_df.loc['Ortalama'] = results_no_pca_df.mean(numeric_only=True)
results_pca_df.loc['Ortalama'] = results_pca_df.mean(numeric_only=True)

print("\nPCA Öncesi Sonuçlar:")
print(results_no_pca_df)

print("\nPCA Sonrası Sonuçlar:")
print(results_pca_df)
