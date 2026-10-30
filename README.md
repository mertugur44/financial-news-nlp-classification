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
| --- | ---: |
| Accuracy | **%91,46** |
| Precision | **%91,54** |
| Recall | **%91,46** |
| F1-Score | **%91,37** |
| Specificity | **%96,95** |
| AUC | **%97,81** |

PCA uygulandığında doğruluk **%83,38** seviyesine gerilemiştir. Bu sonuç, bu veri setinde boyut indirgeme işleminin hesaplama maliyetini azaltabilmesine rağmen sınıflandırma performansında bilgi kaybına yol açabildiğini göstermiştir.

## Kaynak Kodlar

Projenin Python kaynak kodları [`src/`](./src) klasörü altında yer almaktadır.

- [`preprocessing.py`](./src/preprocessing.py) — Veri ön işleme, TF-IDF, özellik seçimi, PCA ve çapraz doğrulama hazırlıkları
- [`machine_learning_models.py`](./src/machine_learning_models.py) — Naive Bayes, SVM, Logistic Regression, Random Forest ve KNN modelleri
- [`deep_learning_models.py`](./src/deep_learning_models.py) — ANN, LSTM, CNN, HAN ve Transformer deneyleri
- [`custom_ann.py`](./src/custom_ann.py) — Geliştirilmiş YSA modeli, SMOTE, PCA karşılaştırmaları ve değerlendirme metrikleri

Kaynak kodların deney yapısı ve dosya açıklamaları için:

📁 [`src/README.md`](./src/README.md)

## Proje Dosyaları

- 📊 [`all-data.csv`](./all-data.csv) — Finansal haber veri seti
- 💻 [`src/`](./src) — Python kaynak kodları
- 📦 [`requirements.txt`](./requirements.txt) — Python bağımlılıkları
- 📄 [`FINANCIAL_NEWS_NLP_ARTICLE.pdf`](./FINANCIAL_NEWS_NLP_ARTICLE.pdf) — Türkçe akademik makale
- 📄 [`FINANCIAL_NEWS_NLP_ARTICLE_EN.pdf`](./FINANCIAL_NEWS_NLP_ARTICLE_EN.pdf) — İngilizce akademik makale
- 📈 [`FINANCIAL_NEWS_NLP_RESULTS.pdf`](./FINANCIAL_NEWS_NLP_RESULTS.pdf) — Model karşılaştırmaları, confusion matrix'ler, PCA öncesi/sonrası sonuçlar ve diğer görseller

## Kullanılan Teknolojiler ve Yöntemler

`Python` `NLP` `Machine Learning` `Deep Learning` `TensorFlow` `Keras` `Scikit-learn` `Pandas` `NumPy` `TF-IDF` `SMOTE` `PCA` `ANN` `LSTM` `CNN` `HAN` `Transformer` `Stratified K-Fold`

## Kurulum

Gerekli Python bağımlılıkları repository kök dizinindeki `requirements.txt` dosyasında bulunmaktadır.

```bash
pip install -r requirements.txt
```

Kodlar proje kapsamında gerçekleştirilen farklı deneyleri temsil edecek şekilde `src/` klasörü altında düzenlenmiştir.

## Akademik Makaleler

Çalışmanın yöntemleri, literatür incelemesi, deney süreci ve ayrıntılı model sonuçları akademik makalelerde yer almaktadır.

🇹🇷 [Türkçe Makaleyi Görüntüle](./FINANCIAL_NEWS_NLP_ARTICLE.pdf)

🇬🇧 [English Article](./FINANCIAL_NEWS_NLP_ARTICLE_EN.pdf)

## Sonuçlar ve Görseller

Model karşılaştırmaları, karmaşıklık matrisleri, PCA öncesi ve sonrası deneyler ile diğer sonuç görselleri ayrı bir PDF dosyasında sunulmuştur.

📈 [Model Sonuçlarını ve Görsellerini Görüntüle](./FINANCIAL_NEWS_NLP_RESULTS.pdf)

## Çalışmanın Yazarları

- Mert Uğur
- Sudenaz Özel
- Gülay Çiçek
