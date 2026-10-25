# Doğal Dil İşleme ile Finansal Haberlerin Sınıflandırılması

Bu projede finansal haber başlıklarının **pozitif, negatif ve nötr** olarak sınıflandırılması için Doğal Dil İşleme (NLP), Makine Öğrenmesi ve Derin Öğrenme yöntemleri karşılaştırılmıştır.

Çalışmada farklı veri ön işleme ve öznitelik çıkarım yöntemlerinin model performansına etkisi incelenmiş; geleneksel makine öğrenmesi modelleri ile derin öğrenme modellerinin sonuçları karşılaştırılmış ve YSA tabanlı geliştirilmiş bir model değerlendirilmiştir.

## Veri Seti

Projede Kaggle üzerinde yayımlanan **Sentiment Analysis for Financial News** veri seti kullanılmıştır.

Veri seti finansal haber başlıkları ve bunlara ait üç duygu sınıfından oluşmaktadır:

- `positive`
- `negative`
- `neutral`

Çalışmanın modelleme sürecinde **4.845 finansal haber kaydı** değerlendirilmiştir.

📊 [`all-data.csv`](./all-data.csv) — Projede kullanılan finansal haber veri seti

## Veri Hazırlama ve NLP Süreci

Modelleme öncesinde haber başlıklarına çeşitli veri hazırlama işlemleri uygulanmıştır:

- Metin temizleme
- Küçük harfe dönüştürme
- Noktalama işaretleri ve sayıların temizlenmesi
- TF-IDF ile metinlerin sayısal özniteliklere dönüştürülmesi
- Varyans Eşiği ile düşük varyanslı özniteliklerin incelenmesi
- PCA ile boyut indirgeme
- SMOTE ile sınıf dengesizliğinin ele alınması
- 5-Fold Stratified Cross-Validation ile model değerlendirme

TF-IDF aşamasında en fazla **5.000 öznitelik** kullanılmış ve PCA'nın model performansına etkisi ayrıca karşılaştırılmıştır.

## Karşılaştırılan Modeller

### Makine Öğrenmesi

- Naive Bayes
- Support Vector Machine (SVM)
- Logistic Regression
- Random Forest
- K-Nearest Neighbors (KNN)

### Derin Öğrenme

- Artificial Neural Network (ANN / YSA)
- Long Short-Term Memory (LSTM)
- Convolutional Neural Network (CNN)
- Hierarchical Attention Network (HAN)
- Transformer

Bunlara ek olarak YSA tabanlı geliştirilmiş model üzerinde veri hazırlama, sınıf dengeleme ve öznitelik işleme yöntemlerinin etkisi incelenmiştir.

## En İyi Sonuç

En yüksek performans, **PCA uygulanmadan değerlendirilen geliştirilmiş YSA modelinde** elde edilmiştir.

| Metrik | Sonuç |
|---|---:|
| Accuracy | **%91,46** |
| Precision | **%91,54** |
| Recall | **%91,46** |
| F1-Score | **%91,37** |
| Specificity | **%96,95** |
| AUC | **%97,81** |

PCA uygulandığında doğruluk **%83,38** seviyesine gerilemiştir. Bu sonuç, bu veri setinde boyut indirgeme işleminin hesaplama maliyetini azaltabilmesine rağmen sınıflandırma performansında bilgi kaybına yol açabildiğini göstermiştir.

## Kullanılan Teknolojiler ve Yöntemler

`Python` `NLP` `Machine Learning` `Deep Learning` `TensorFlow` `Keras` `Scikit-learn` `Pandas` `NumPy` `TF-IDF` `SMOTE` `PCA` `ANN` `LSTM` `CNN` `HAN` `Transformer` `Stratified K-Fold`

## Makale

📄 [`FINANCIAL_NEWS_NLP_ARTICLE.pdf`](./FINANCIAL_NEWS_NLP_ARTICLE.pdf) — Çalışmanın yöntemlerini, literatür incelemesini, deneylerini ve ayrıntılı model sonuçlarını içeren akademik makale

## Çalışmanın Yazarları

- Sudenaz Özel
- Mert Uğur
- Gülay Çiçek
