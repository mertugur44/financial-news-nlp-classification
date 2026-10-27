"""Veri Hazırlama ve Ön İşleme Deneyleri

Metin ön işleme, TF-IDF, PCA, Varyans Eşiği ve 5-Fold Stratified Cross-Validation hazırlıklarını içerir.

Bu dosya, çalışmada kullanılan bağımsız deney bloklarını model grubuna göre bir araya getirir.
Bölümler Google Colab üzerinde ayrı deneyler olarak çalıştırılmak üzere hazırlanmıştır;
bu nedenle veri yükleme ve bazı import adımları bölümler arasında tekrar eder.
"""


########################################################################################
# 5-Fold Stratified Cross-Validation ve PCA Hazırlığı
########################################################################################

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
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

neutral_sayisi = df[df['neutral'] == 'neutral'].shape[0]
positive_sayisi = df[df['neutral'] == 'positive'].shape[0]
negative_sayisi = df[df['neutral'] == 'negative'].shape[0]

print("\nDuygu etiketlerinin sayısı:")
print(f"Nötr: {neutral_sayisi}")
print(f"Pozitif: {positive_sayisi}")
print(f"Negatif: {negative_sayisi}")

def temizle_metin(metin):
    # Küçük harfe dönüştür
    metin = metin.lower()
    # Noktalama işaretlerini ve özel karakterleri kaldır
    metin = re.sub(f"[{re.escape(string.punctuation)}]", "", metin)
    # Sayıları kaldır
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

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['neutral'])  # Etiketleri sayısal hale çevir

pca = PCA(n_components=100, random_state=42)  # 100 bileşene indirgeniyor
X_pca = pca.fit_transform(X.toarray())  # PCA için matris yoğun forma dönüştürülür

explained_variance_ratio = pca.explained_variance_ratio_.sum()
print(f"\nPCA ile açıklanan toplam varyans oranı: {explained_variance_ratio:.4f}")

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

folds = []

for fold, (train_idx, test_idx) in enumerate(kfold.split(X_pca, y), start=1):
    fold_data = {
        "Fold": fold,
        "Train Indexes": train_idx.tolist(),
        "Test Indexes": test_idx.tolist()
    }
    folds.append(fold_data)

folds_df = pd.DataFrame(folds)

print("\nK-Fold Bölünme Sonuçları:")
print(folds_df)


########################################################################################
# Varyans Eşiği ile Öznitelik Seçimi
########################################################################################

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
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

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

folds = []

for fold, (train_idx, test_idx) in enumerate(kfold.split(X_pca, y), start=1):
    fold_data = {
        "Fold": fold,
        "Train Indexes": train_idx.tolist(),
        "Test Indexes": test_idx.tolist()
    }
    folds.append(fold_data)

folds_df = pd.DataFrame(folds)
print("\nK-Fold Bölünme Sonuçları:")
print(folds_df)
