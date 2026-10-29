# Kaynak Kodlar

Bu klasör, finansal haberlerin **pozitif, negatif ve nötr** olarak sınıflandırılması için gerçekleştirilen veri ön işleme, makine öğrenmesi, derin öğrenme ve geliştirilmiş YSA çalışmalarının Python kodlarını içermektedir.

Kodlar proje kapsamında Google Colab ortamında geliştirilen deneysel çalışma akışından düzenlenerek GitHub üzerinde daha okunabilir bir yapıda sunulmuştur.

## Dosyalar

### `preprocessing.py`

Veri hazırlama ve özellik çıkarımı işlemlerini içerir.

Başlıca işlemler:

- Metin verilerinin temizlenmesi ve hazırlanması
- TF-IDF ile metinlerin sayısal özelliklere dönüştürülmesi
- Label Encoding
- Variance Threshold ile özellik seçimi
- PCA ile boyut indirgeme
- Stratified K-Fold Cross Validation yapısının hazırlanması

---

### `machine_learning_models.py`

Geleneksel makine öğrenmesi algoritmalarıyla gerçekleştirilen sınıflandırma deneylerini içerir.

Kullanılan modeller:

- Naive Bayes
- Support Vector Machine (SVM)
- Logistic Regression
- Random Forest
- K-Nearest Neighbors (KNN)

Modeller farklı ön işleme ve özellik indirgeme yaklaşımlarıyla değerlendirilmiştir. PCA ve Variance Threshold uygulamalarının model performansına etkileri karşılaştırılmıştır.

---

### `deep_learning_models.py`

Finansal haber sınıflandırması için kullanılan derin öğrenme modellerini içerir.

Çalışmada değerlendirilen modeller:

- Artificial Neural Network (ANN / YSA)
- Long Short-Term Memory (LSTM)
- Convolutional Neural Network (CNN)
- Hierarchical Attention Network (HAN)
- Transformer

Modellerin farklı veri hazırlama ve boyut indirgeme yöntemleri altındaki performansları karşılaştırılmıştır.

---

### `custom_ann.py`

Çalışma kapsamında geliştirilen özel Yapay Sinir Ağı modelinin eğitim ve değerlendirme sürecini içerir.

Bu bölümde:

- TF-IDF
- SMOTE
- Variance Threshold
- PCA
- Stratified 5-Fold Cross Validation

yaklaşımları kullanılarak farklı deneyler gerçekleştirilmiştir.

Geliştirilen YSA modeli PCA uygulanmadan gerçekleştirilen deneylerde çalışmanın en yüksek sonuçlarını elde etmiştir.

Başlıca sonuçlar:

| Metrik | Sonuç |
| --- | ---: |
| Accuracy | **%91.46** |
| Precision | **%91.54** |
| Recall | **%91.46** |
| F1-Score | **%91.37** |
| Specificity | **%96.95** |
| AUC | **%97.81** |

## Kod Yapısı Hakkında

Projenin orijinal deneyleri Google Colab üzerinde ayrı çalışma blokları halinde yürütülmüştür.

Bu nedenle bazı veri hazırlama, model oluşturma veya değerlendirme adımları farklı deney dosyalarında tekrar edebilir. Bu tekrarlar farklı model ve ön işleme kombinasyonlarının bağımsız olarak değerlendirilebilmesi amacıyla korunmuştur.

Kodlarda `GridSearchCV` import edilmiş olmakla birlikte otomatik hiperparametre optimizasyonu gerçekleştirilmemiştir.

## Bağımlılıklar

Projenin temel Python bağımlılıkları repository kök dizinindeki [`requirements.txt`](../requirements.txt) dosyasında bulunmaktadır.

Bağımlılıkları yüklemek için:

```bash
pip install -r requirements.txt
```

Ana proje açıklaması, veri seti, yöntemler ve sonuçlar için repository kök dizinindeki [`README.md`](../README.md) dosyasına bakabilirsiniz.
