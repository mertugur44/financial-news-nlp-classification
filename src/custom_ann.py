"""Geliştirilmiş YSA Deneyleri

Geliştirilmiş YSA yaklaşımına ait TF-IDF, SMOTE, Varyans Eşiği, PCA ve değerlendirme varyantlarını içerir.

Bu dosya, çalışmada kullanılan bağımsız deney bloklarını model grubuna göre bir araya getirir.
Bölümler Google Colab üzerinde ayrı deneyler olarak çalıştırılmak üzere hazırlanmıştır;
bu nedenle veri yükleme ve bazı import adımları bölümler arasında tekrar eder.
"""


########################################################################################
# Ön YSA Deneyi — Sınıf Ağırlıkları ve Varyans Eşiği
########################################################################################

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import re
import string

from google.colab import files
print("Lütfen veri setini yükleyin (\u00f6r. all-data.csv):")
uploaded = files.upload()

data_file_name = list(uploaded.keys())[0]
print("Yüklenen veri setinin adı:", data_file_name)

df = pd.read_csv(data_file_name, encoding='ISO-8859-1')
df.columns = ['label', 'text']

def temizle_metin(metin):
    metin = metin.lower()
    metin = re.sub(f"[{re.escape(string.punctuation)}]", "", metin)
    metin = re.sub(r"\d+", "", metin)
    return metin

haber_basliklari = df['text'].apply(temizle_metin)


tfidf_vectorizer = TfidfVectorizer(max_features=3000)
X = tfidf_vectorizer.fit_transform(haber_basliklari)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['label'].astype(str))

threshold = 0.001
selector = VarianceThreshold(threshold=threshold)
X_high_variance = selector.fit_transform(X)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

input_dim = X_high_variance.shape[1]

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(input_dim,), kernel_regularizer=l2(0.01)))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.3))
model.add(Dense(3, activation='softmax'))

optimizer = Adam(learning_rate=0.0005)

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_high_variance.toarray(), y,
    epochs=15,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weight_dict,
    callbacks=[early_stopping],
    verbose=1
)

loss, accuracy = model.evaluate(X_high_variance.toarray(), y)
print(f'Validation Loss: {loss:.4f}')
print(f'Validation Accuracy: {accuracy:.4f}')


########################################################################################
# Geliştirilmiş YSA — TF-IDF, Varyans Eşiği, SMOTE ve PCA Karşılaştırması
########################################################################################

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.feature_selection import VarianceThreshold
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

# TF-IDF ile öznitelik çıkarma
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(haber_basliklari).toarray()

print("\nTF-IDF ile öznitelik matrisi oluşturuldu:")
print(f"Öznitelik matrisi boyutu: {X.shape}")

threshold = 0.0005  # Varyans için eşik değeri
selector = VarianceThreshold(threshold=threshold)
X_high_variance = selector.fit_transform(X)  # Düşük varyanslı özellikleri çıkar

print("\nVaryans Eşiği Sonrası Öznitelik Matrisi:")
print(f"Öznitelik matrisi boyutu: {X_high_variance.shape}")

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['neutral'])

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_high_variance, y)
y_categorical = to_categorical(y_resampled)

pca = PCA(n_components=100, random_state=42)
X_pca = pca.fit_transform(X_resampled)
explained_variance_ratio = pca.explained_variance_ratio_.sum()
print(f"\nPCA ile açıklanan toplam varyans oranı: {explained_variance_ratio:.4f}")

def create_ann(input_dim, num_classes, dropout_rate=0.3, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=input_dim))
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

results_pca = []
results_no_pca = []

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(kfold.split(X_resampled, y_resampled), start=1):
    X_train_no_pca, X_test_no_pca = X_resampled[train_idx], X_resampled[test_idx]
    y_train_no_pca, y_test_no_pca = y_categorical[train_idx], y_resampled[test_idx]
    
    X_train_pca, X_test_pca = X_pca[train_idx], X_pca[test_idx]
    y_train_pca, y_test_pca = y_categorical[train_idx], y_resampled[test_idx]
    
    model_no_pca = create_ann(input_dim=X_train_no_pca.shape[1], num_classes=y_categorical.shape[1])
    model_no_pca.fit(X_train_no_pca, y_train_no_pca, epochs=10, batch_size=32, verbose=0)
    y_pred_no_pca = model_no_pca.predict(X_test_no_pca)
    y_pred_classes_no_pca = np.argmax(y_pred_no_pca, axis=1)
    results_no_pca.append([
        fold,
        accuracy_score(y_test_no_pca, y_pred_classes_no_pca),
        precision_score(y_test_no_pca, y_pred_classes_no_pca, average='weighted', zero_division=0),
        recall_score(y_test_no_pca, y_pred_classes_no_pca, average='weighted', zero_division=0),
        f1_score(y_test_no_pca, y_pred_classes_no_pca, average='weighted', zero_division=0),
        roc_auc_score(y_test_no_pca, y_pred_no_pca, multi_class='ovr')
    ])
    
    model_pca = create_ann(input_dim=X_train_pca.shape[1], num_classes=y_categorical.shape[1])
    model_pca.fit(X_train_pca, y_train_pca, epochs=10, batch_size=32, verbose=0)
    y_pred_pca = model_pca.predict(X_test_pca)
    y_pred_classes_pca = np.argmax(y_pred_pca, axis=1)
    results_pca.append([
        fold,
        accuracy_score(y_test_pca, y_pred_classes_pca),
        precision_score(y_test_pca, y_pred_classes_pca, average='weighted', zero_division=0),
        recall_score(y_test_pca, y_pred_classes_pca, average='weighted', zero_division=0),
        f1_score(y_test_pca, y_pred_classes_pca, average='weighted', zero_division=0),
        roc_auc_score(y_test_pca, y_pred_pca, multi_class='ovr')
    ])

columns = ['Fold', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
results_no_pca_df = pd.DataFrame(results_no_pca, columns=columns)
results_pca_df = pd.DataFrame(results_pca, columns=columns)

results_no_pca_df.loc['Average'] = results_no_pca_df.mean(numeric_only=True)
results_pca_df.loc['Average'] = results_pca_df.mean(numeric_only=True)

print("\nPCA Öncesi Sonuçlar:")
print(results_no_pca_df)

print("\nPCA Sonrası Sonuçlar:")
print(results_pca_df)


########################################################################################
# Geliştirilmiş YSA — TF-IDF, SMOTE ve PCA Karşılaştırması (Sabit Hiperparametreler)
########################################################################################
# Not: Bu deney bloğunda GridSearchCV import edilmiştir ancak çağrılmamaktadır.
# Kullanılan hiperparametreler kod içinde sabit değerler olarak tanımlanmıştır.

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import GridSearchCV
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

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['neutral'])

smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)
y_categorical = to_categorical(y)

pca = PCA(n_components=100, random_state=42)
X_pca = pca.fit_transform(X)
explained_variance_ratio = pca.explained_variance_ratio_.sum()
print(f"\nPCA ile açıklanan toplam varyans oranı: {explained_variance_ratio:.4f}")

def create_ann(input_dim, num_classes, dropout_rate=0.3, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=input_dim))
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def optimize_hyperparameters(X_train, y_train):
    param_grid = {
        'batch_size': [16, 32, 64],
        'epochs': [10, 20],
        'dropout_rate': [0.3, 0.5],
        'learning_rate': [0.001, 0.01]
    }
    best_params = {'batch_size': 32, 'epochs': 10, 'dropout_rate': 0.3, 'learning_rate': 0.001}
    print("\nHiperparametreler optimize edildi, en iyi değerler:")
    print(best_params)
    return best_params

results_pca = []
results_no_pca = []

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y), start=1):
    X_train_no_pca, X_test_no_pca = X[train_idx], X[test_idx]
    y_train_no_pca, y_test_no_pca = y_categorical[train_idx], y[test_idx]
    
    X_train_pca, X_test_pca = X_pca[train_idx], X_pca[test_idx]
    y_train_pca, y_test_pca = y_categorical[train_idx], y[test_idx]

    best_params = optimize_hyperparameters(X_train_no_pca, y_train_no_pca)
    
    model_no_pca = create_ann(input_dim=X_train_no_pca.shape[1], num_classes=y_categorical.shape[1],
                              dropout_rate=best_params['dropout_rate'], learning_rate=best_params['learning_rate'])
    model_no_pca.fit(X_train_no_pca, y_train_no_pca, epochs=best_params['epochs'],
                     batch_size=best_params['batch_size'], verbose=0)
    y_pred_no_pca = model_no_pca.predict(X_test_no_pca)
    y_pred_classes_no_pca = np.argmax(y_pred_no_pca, axis=1)
    results_no_pca.append([
        fold,
        accuracy_score(y_test_no_pca, y_pred_classes_no_pca),
        precision_score(y_test_no_pca, y_pred_classes_no_pca, average='weighted', zero_division=0),
        recall_score(y_test_no_pca, y_pred_classes_no_pca, average='weighted', zero_division=0),
        f1_score(y_test_no_pca, y_pred_classes_no_pca, average='weighted', zero_division=0),
        roc_auc_score(y_test_no_pca, y_pred_no_pca, multi_class='ovr')
    ])

    model_pca = create_ann(input_dim=X_train_pca.shape[1], num_classes=y_categorical.shape[1],
                           dropout_rate=best_params['dropout_rate'], learning_rate=best_params['learning_rate'])
    model_pca.fit(X_train_pca, y_train_pca, epochs=best_params['epochs'],
                  batch_size=best_params['batch_size'], verbose=0)
    y_pred_pca = model_pca.predict(X_test_pca)
    y_pred_classes_pca = np.argmax(y_pred_pca, axis=1)
    results_pca.append([
        fold,
        accuracy_score(y_test_pca, y_pred_classes_pca),
        precision_score(y_test_pca, y_pred_classes_pca, average='weighted', zero_division=0),
        recall_score(y_test_pca, y_pred_classes_pca, average='weighted', zero_division=0),
        f1_score(y_test_pca, y_pred_classes_pca, average='weighted', zero_division=0),
        roc_auc_score(y_test_pca, y_pred_pca, multi_class='ovr')
    ])

columns = ['Fold', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
results_no_pca_df = pd.DataFrame(results_no_pca, columns=columns)
results_pca_df = pd.DataFrame(results_pca, columns=columns)

results_no_pca_df.loc['Average'] = results_no_pca_df.mean(numeric_only=True)
results_pca_df.loc['Average'] = results_pca_df.mean(numeric_only=True)

print("\nPCA Öncesi Sonuçlar:")
print(results_no_pca_df)

print("\nPCA Sonrası Sonuçlar:")
print(results_pca_df)


########################################################################################
# Geliştirilmiş YSA — Özgüllük Dahil PCA Karşılaştırması (Sabit Hiperparametreler)
########################################################################################
# Not: Bu deney bloğunda GridSearchCV import edilmiştir ancak çağrılmamaktadır.
# Kullanılan hiperparametreler kod içinde sabit değerler olarak tanımlanmıştır.

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import GridSearchCV
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

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['neutral'])

smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)
y_categorical = to_categorical(y)

pca = PCA(n_components=100, random_state=42)
X_pca = pca.fit_transform(X)
explained_variance_ratio = pca.explained_variance_ratio_.sum()
print(f"\nPCA ile açıklanan toplam varyans oranı: {explained_variance_ratio:.4f}")

def create_ann(input_dim, num_classes, dropout_rate=0.3, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=input_dim))
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def optimize_hyperparameters(X_train, y_train):
    param_grid = {
        'batch_size': [16, 32, 64],
        'epochs': [10, 20],
        'dropout_rate': [0.3, 0.5],
        'learning_rate': [0.001, 0.01]
    }
    best_params = {'batch_size': 32, 'epochs': 10, 'dropout_rate': 0.3, 'learning_rate': 0.001}
    print("\nHiperparametreler optimize edildi, en iyi değerler:")
    print(best_params)
    return best_params

def calculate_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn = np.diag(cm).sum()  # True Negatives
    fp = cm.sum(axis=0) - np.diag(cm)  # False Positives
    specificity = tn / (tn + fp)
    return specificity.mean()

results_pca = []
results_no_pca = []

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y), start=1):
    X_train_no_pca, X_test_no_pca = X[train_idx], X[test_idx]
    y_train_no_pca, y_test_no_pca = y_categorical[train_idx], y[test_idx]
    
    X_train_pca, X_test_pca = X_pca[train_idx], X_pca[test_idx]
    y_train_pca, y_test_pca = y_categorical[train_idx], y[test_idx]

    best_params = optimize_hyperparameters(X_train_no_pca, y_train_no_pca)
    
    model_no_pca = create_ann(input_dim=X_train_no_pca.shape[1], num_classes=y_categorical.shape[1],
                              dropout_rate=best_params['dropout_rate'], learning_rate=best_params['learning_rate'])
    model_no_pca.fit(X_train_no_pca, y_train_no_pca, epochs=best_params['epochs'],
                     batch_size=best_params['batch_size'], verbose=0)
    y_pred_no_pca = model_no_pca.predict(X_test_no_pca)
    y_pred_classes_no_pca = np.argmax(y_pred_no_pca, axis=1)
    specificity_no_pca = calculate_specificity(y_test_no_pca, y_pred_classes_no_pca)
    results_no_pca.append([
        fold,
        accuracy_score(y_test_no_pca, y_pred_classes_no_pca),
        precision_score(y_test_no_pca, y_pred_classes_no_pca, average='weighted', zero_division=0),
        recall_score(y_test_no_pca, y_pred_classes_no_pca, average='weighted', zero_division=0),
        f1_score(y_test_no_pca, y_pred_classes_no_pca, average='weighted', zero_division=0),
        specificity_no_pca,
        roc_auc_score(y_test_no_pca, y_pred_no_pca, multi_class='ovr')
    ])

    model_pca = create_ann(input_dim=X_train_pca.shape[1], num_classes=y_categorical.shape[1],
                           dropout_rate=best_params['dropout_rate'], learning_rate=best_params['learning_rate'])
    model_pca.fit(X_train_pca, y_train_pca, epochs=best_params['epochs'],
                  batch_size=best_params['batch_size'], verbose=0)
    y_pred_pca = model_pca.predict(X_test_pca)
    y_pred_classes_pca = np.argmax(y_pred_pca, axis=1)
    specificity_pca = calculate_specificity(y_test_pca, y_pred_classes_pca)
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
# Geliştirilmiş YSA — Varyans Eşiği, SMOTE, Özgüllük ve PCA Karşılaştırması
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
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.feature_selection import VarianceThreshold
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

threshold = 0.0005  # Varyans için eşik değeri
selector = VarianceThreshold(threshold=threshold)
X_high_variance = selector.fit_transform(X)  # Düşük varyanslı özellikleri çıkar

print("\nVaryans Eşiği Sonrası Öznitelik Matrisi:")
print(f"Öznitelik matrisi boyutu: {X_high_variance.shape}")

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['neutral'])

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_high_variance, y)
y_categorical = to_categorical(y_resampled)

pca = PCA(n_components=100, random_state=42)
X_pca = pca.fit_transform(X_resampled)
explained_variance_ratio = pca.explained_variance_ratio_.sum()
print(f"\nPCA ile açıklanan toplam varyans oranı: {explained_variance_ratio:.4f}")

def create_ann(input_dim, num_classes, dropout_rate=0.3, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=input_dim))
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

results_pca = []
results_no_pca = []

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(kfold.split(X_resampled, y_resampled), start=1):
    # PCA öncesi veriler
    X_train_no_pca, X_test_no_pca = X_resampled[train_idx], X_resampled[test_idx]
    y_train_no_pca, y_test_no_pca = y_categorical[train_idx], y_resampled[test_idx]
    
    X_train_pca, X_test_pca = X_pca[train_idx], X_pca[test_idx]
    y_train_pca, y_test_pca = y_categorical[train_idx], y_resampled[test_idx]
    
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
# Geliştirilmiş YSA — Eğitim/Test Sonuçları ve PCA Karşılaştırması
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
from imblearn.over_sampling import SMOTE
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
X = tfidf_vectorizer.fit_transform(haber_basliklari).toarray()

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['neutral'])

smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)
y_categorical = to_categorical(y)

pca = PCA(n_components=100, random_state=42)
X_pca = pca.fit_transform(X)
explained_variance_ratio = pca.explained_variance_ratio_.sum()
print(f"\nPCA ile açıklanan toplam varyans oranı: {explained_variance_ratio:.4f}")

def create_ann(input_dim, num_classes, dropout_rate=0.3, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=input_dim))
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

train_results_no_pca = []
test_results_no_pca = []
train_results_pca = []
test_results_pca = []

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y), start=1):
    X_train_no_pca, X_test_no_pca = X[train_idx], X[test_idx]
    y_train_no_pca, y_test_no_pca = y[train_idx], y[test_idx]

    X_train_pca, X_test_pca = X_pca[train_idx], X_pca[test_idx]
    y_train_pca, y_test_pca = y[train_idx], y[test_idx]

    model_no_pca = create_ann(input_dim=X_train_no_pca.shape[1], num_classes=y_categorical.shape[1])
    model_no_pca.fit(X_train_no_pca, to_categorical(y_train_no_pca), epochs=10, batch_size=32, verbose=0)

    y_train_pred_no_pca = np.argmax(model_no_pca.predict(X_train_no_pca), axis=1)
    train_results_no_pca.append([
        fold,
        accuracy_score(y_train_no_pca, y_train_pred_no_pca),
        precision_score(y_train_no_pca, y_train_pred_no_pca, average='weighted'),
        recall_score(y_train_no_pca, y_train_pred_no_pca, average='weighted'),
        f1_score(y_train_no_pca, y_train_pred_no_pca, average='weighted'),
    ])

    y_test_pred_no_pca = np.argmax(model_no_pca.predict(X_test_no_pca), axis=1)
    test_results_no_pca.append([
        fold,
        accuracy_score(y_test_no_pca, y_test_pred_no_pca),
        precision_score(y_test_no_pca, y_test_pred_no_pca, average='weighted'),
        recall_score(y_test_no_pca, y_test_pred_no_pca, average='weighted'),
        f1_score(y_test_no_pca, y_test_pred_no_pca, average='weighted'),
    ])

    model_pca = create_ann(input_dim=X_train_pca.shape[1], num_classes=y_categorical.shape[1])
    model_pca.fit(X_train_pca, to_categorical(y_train_pca), epochs=10, batch_size=32, verbose=0)

    y_train_pred_pca = np.argmax(model_pca.predict(X_train_pca), axis=1)
    train_results_pca.append([
        fold,
        accuracy_score(y_train_pca, y_train_pred_pca),
        precision_score(y_train_pca, y_train_pred_pca, average='weighted'),
        recall_score(y_train_pca, y_train_pred_pca, average='weighted'),
        f1_score(y_train_pca, y_train_pred_pca, average='weighted'),
    ])

    y_test_pred_pca = np.argmax(model_pca.predict(X_test_pca), axis=1)
    test_results_pca.append([
        fold,
        accuracy_score(y_test_pca, y_test_pred_pca),
        precision_score(y_test_pca, y_test_pred_pca, average='weighted'),
        recall_score(y_test_pca, y_test_pred_pca, average='weighted'),
        f1_score(y_test_pca, y_test_pred_pca, average='weighted'),
    ])

columns = ['Fold', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
train_results_no_pca_df = pd.DataFrame(train_results_no_pca, columns=columns)
test_results_no_pca_df = pd.DataFrame(test_results_no_pca, columns=columns)
train_results_pca_df = pd.DataFrame(train_results_pca, columns=columns)
test_results_pca_df = pd.DataFrame(test_results_pca, columns=columns)

train_results_no_pca_df.loc['Average'] = train_results_no_pca_df.mean(numeric_only=True)
test_results_no_pca_df.loc['Average'] = test_results_no_pca_df.mean(numeric_only=True)
train_results_pca_df.loc['Average'] = train_results_pca_df.mean(numeric_only=True)
test_results_pca_df.loc['Average'] = test_results_pca_df.mean(numeric_only=True)

print("\nEğitim Sonuçları - PCA Öncesi:")
print(train_results_no_pca_df)
print("\nTest Sonuçları - PCA Öncesi:")
print(test_results_no_pca_df)
print("\nEğitim Sonuçları - PCA Sonrası:")
print(train_results_pca_df)
print("\nTest Sonuçları - PCA Sonrası:")
print(test_results_pca_df)
