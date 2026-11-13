# Proje 4: Kredi Kartı Dolandırıcılık Analizi (Credit Card Fraud Detection)
### Büyük Veri Mühendisliği, Sınıf Dengesizliği ve Makine Öğrenimi ile Uçtan Uca Dolandırıcılık Tespiti

---

## 📌 1. Proje Özeti ve Problem Tanımı
Finansal işlemlerde kredi kartı dolandırıcılığı (fraud), milyonlarca normal işlem arasında son derece nadir (%0.1 - %0.2 oranında) meydana gelen kritik bir güvenlik riskidir. Bu projede, Kaggle standardındaki işlem verileri ve büyük ölçekli sentetik simülasyonlar kullanılarak, sınıf dengesizliğinin uç boyutlarda olduğu finansal veriler üzerinde yüksek performanslı, üretime hazır bir makine öğrenimi ve PySpark boru hattı (pipeline) inşa edilmiştir.

### 🎯 Araştırma Sorusu
> **"Çok veri olması, nadir fraud olaylarını bulmayı gerçekten kolaylaştırıyor mu?"**

Bu soru projenin temel eksenini oluşturmakta olup, farklı veri boyutları (%10, %25, %50, %100) ve dengesizlik oranları altında yürütülen kontrollü deneylerle ampirik olarak test edilmiş ve yanıtlanmıştır.

---

## 👥 2. Görev Dağılımı ve Modüler Mimari (5 Öğrenci Raporu)

Proje, 5 kişilik uzman bir veri bilimi ve büyük veri mühendisliği ekibinin görev dağılımına uygun olarak modüler mimaride tasarlanmıştır:

| Öğrenci | Modül Dosyası | Temel Sorumluluk ve Çıktılar |
| :--- | :--- | :--- |
| **ÖĞRENCİ 1** | `src/data_prep.py` | **Veri Hazırlığı & EDA**: Kaggle veri yükleme, sentetik veri simülasyonu, sınıf dağılımı (IR) hesaplama, Amount ve Time özet istatistikleri, dağılım grafikleri ve otomatik EDA raporlama. |
| **ÖĞRENCİ 2** | `src/imbalance.py` | **Sınıf Dengesizliği Çözümleri**: Random Undersampling, SMOTE sentetik aşırı örnekleme, Class Weights ve `scale_pos_weight` hesaplama. **Data Leakage Guard**: Test setinin el değmemiş kalmasını garanti eden sızıntı önleme mimarisi. |
| **ÖĞRENCİ 3** | `src/spark_features.py` | **PySpark Özellik Mühendisliği**: Müşteri bazlı kayan pencereler (Rolling Windows: son 1, 6 ve 24 saatlik işlem adetleri), müşteri ortalama işlem tutarı, tutar z-skoru/sapması, işlemler arası geçen süre (Delta Time), VectorAssembler ML pipeline. |
| **ÖĞRENCİ 4** | `src/modeling.py` | **Modelleme & Metrik Değerlendirmesi**: LightGBM, XGBoost, Random Forest ve Naive Baseline. **Accuracy Paradox (Doğruluk Paradoksu)** analizi, Precision, Recall, F1, F2, AUPRC (PR-AUC), ROC-AUC, Confusion Matrix ve PR Eğrisi çizimleri. |
| **ÖĞRENCİ 5** | `src/experiments.py` | **Ölçek ve Dengesizlik Etki Analizi**: Araştırma sorusunu yanıtlayan kontrollü deney simülasyonu (%10, %25, %50, %100 ölçek), Eğitim Süresi vs AUPRC/Recall trade-off analizi, Pareto doyum grafikleri ve sonuç raporu. |

---

## 🏗️ 3. Proje Dizin Yapısı

```bash
Bva_kredi_karti_dolandirilicik_analizi/
│
├── data/
│   └── creditcard.csv                 # Kaggle formatındaki veri seti (otomatik üretilir/yüklenir)
│
├── reports/
│   ├── figures/                       # Üretilen yüksek çözünürlüklü analiz grafikleri
│   │   ├── 01_class_distribution.png
│   │   ├── 02_amount_time_distribution.png
│   │   ├── 03_feature_correlations.png
│   │   ├── 04_confusion_matrices.png
│   │   ├── 05_pr_and_roc_curves.png
│   │   └── 06_scale_experiment_results.png
│   └── metrics/                       # JSON, CSV ve TXT metrik raporları
│       ├── eda_class_distribution.json
│       ├── eda_summary_statistics.csv
│       ├── model_comparison_results.csv
│       ├── scaling_experiment_results.csv
│       └── research_question_report.txt
│
├── src/
│   ├── __init__.py
│   ├── data_prep.py                   # Öğrenci 1: Veri Yükleme, Dağılım & EDA
│   ├── imbalance.py                   # Öğrenci 2: Dengesizlik Çözümleri & Sızıntı Koruması
│   ├── spark_features.py              # Öğrenci 3: PySpark Pencere & Davranışsal Özellikler
│   ├── modeling.py                    # Öğrenci 4: Modeller, Accuracy Paradox & PR-AUC
│   └── experiments.py                 # Öğrenci 5: Veri Ölçeği Simülasyonu & Analiz
│
├── main.py                            # Tüm adımları uçtan uca çalıştıran ana orkestrasyon dosyası
├── run_experiments.sh                 # Linux/Mac çalıştırma betiği
├── run_experiments.bat                # Windows çalıştırma betiği
├── requirements.txt                   # Gerekli kütüphaneler listesi
└── README.md                          # Proje teknik ve akademik dokümantasyonu
```

---

## ⚡ 4. Kurulum ve Çalıştırma

### 4.1. Sanal Ortam ve Bağımlılıkların Kurulumu
```bash
# Bağımlılıkları yükleyin
pip install -r requirements.txt
```

### 4.2. Tek Komutla Uçtan Uca Çalıştırma
```bash
# Tam pipeline'ı çalıştırma (120,000 işlem, algoritmik ağırlıklandırma)
python main.py

# Hızlı doğrulama (Smoke test)
python main.py --quick

# Belirli bir dengeleme stratejisi seçerek çalıştırma (none | undersample | smote)
python main.py --sampling smote --samples 150000

# Shell / Batch betikleriyle çalıştırma
bash run_experiments.sh       # Linux / macOS
run_experiments.bat           # Windows
```

---

## 📊 5. Kritik Teknik Bulgular ve Metrik Analizi

### 5.1. Accuracy Paradox (Doğruluk Paradoksu)
Dolandırıcılık tespitinde **Accuracy (Doğruluk)** metriği **en tehlikeli ve yanıltıcı** metriktir. 
* Verideki işlemlerin %99.83'ü Normal, yalnızca %0.17'si Fraud olduğunda; her gelen işleme körü körüne "Normal (0)" diyen bir **Naive Model**, matematiksel olarak **%99.83 doğruluk (Accuracy)** elde eder.
* Ancak bu modelin yakaladığı dolandırıcılık adedi **0 (Sıfır)**'dır (Recall = 0.0). Banka açısından yüz milyonlarca liralık zarar anlamına gelir.
* Bu nedenle model başarımında **AUPRC (Area Under Precision-Recall Curve)**, **Recall (Yakalama Oranı)** ve **F2-Score** (kaçırılan dolandırıcılıklara daha fazla ceza veren metrik) birincil başarı kriteri olarak benimsenmiştir.

### 5.2. Neden ROC-AUC Değil de PR-AUC (AUPRC)?
* **ROC Eğrisi**: False Positive Rate ($FPR = \frac{FP}{FP + TN}$) kullanır. Negatif sınıf ($TN$) yüz binlerce işlemden oluştuğunda, payda çok büyük olduğu için yüzlerce yanlış alarm ($FP$) verilse dahi $FPR$ neredeyse sıfıra yakın kalır. Bu da ROC-AUC skorunun yanıltıcı şekilde %99+ çıkmasına neden olur.
* **PR Eğrisi (Precision-Recall)**: Doğrudan Precision ($\frac{TP}{TP + FP}$) ve Recall ($\frac{TP}{TP + FN}$) arasındaki ilişkiyi inceler. Negatif sınıfın devasa büyüklüğünü denkleme sokmaz; yanlış pozitifleri sert bir şekilde cezalandırır. Uç dengesizlikte modelin gerçek gücünü gösteren tek tarafsız göstergedir.

### 5.3. Veri Sızıntısı (Data Leakage) Önleme İlkesi
SMOTE veya Undersampling gibi yöntemler verinin tamamına uygulanıp ardından train/test ayrımı yapılırsa, sentetik veriler veya silinmiş dağılım bilgileri test setine sızar. Bu durum test skorlarını yapay olarak mükemmel gösterir ancak model gerçek dünyada çöker. 
Geliştirilen `DataLeakageGuard` katmanı ile:
1. Veri ilk olarak **Stratified** şekilde Train (%80) ve Test (%20) olarak bölünür.
2. Dengeleme teknikleri **YALNIZCA Train** setine uygulanır.
3. Test seti, gerçek hayatın acımasız dengesizliğini yansıtacak şekilde kesinlikle doğal oranında bırakılır.

---

## 🔬 6. Araştırma Sorusunun Cevabı ve Deney Sonuçları

### ❓ Soru: "Çok veri olması, nadir fraud olaylarını bulmayı gerçekten kolaylaştırıyor mu?"

### 💡 Ampirik Yanıt:
**"Kısmen Evet, Ancak Belirli Bir Eşikten Sonra Doyuma Ulaşır (Diminishing Returns)."**

Yapılan ölçek deneylerinde (%10 -> %25 -> %50 -> %100 veri hacmi) şu temel dinamikler gözlemlenmiştir:

1. **Mutlak Azınlık Çeşitliliği:** Veri boyutu 10 katına çıktığında eğitim kümesindeki mutlak fraud işlem sayısı artar (örneğin 30 işlemden 300 işleme). Bu artış, modelin dolandırıcı davranış varyansını ve sınırlarını daha iyi genelleştirmesini sağlayarak AUPRC skorunu yükseltir.
2. **False Positive Gürültü Baskısı:** Ancak veri hacmi artarken **dengesizlik oranı (1:580)** sabit kalmaktadır. Yani her 1 yeni fraud örneğine karşılık 580 yeni normal işlem gelmektedir. Bu devasa normal işlem havuzu, model için yeni karar belirsizlikleri ve yanlış alarm riskleri (False Positives) üretmeye devam eder.
3. **Maliyet vs Fayda Eğrisi (Trade-Off):** 
   - Veri boyutu %10'dan %50'ye çıkarken model başarımında belirgin sıçrama görülür.
   - %50'den %100'e geçildiğinde ise AUPRC artışı logaritmik olarak yavaşlarken (plato/doyum noktası), eğitim süresi ve bellek tüketimi katlanarak artar.
4. **Stratejik Sonuç:** Dolandırıcılık tespitinde sadece daha fazla "ham işlem" toplamak çözüm değildir. **Asıl farkı yaratan faktörler**:
   - PySpark ile zaman ve müşteri ekseninde davranışsal özellikler üretmek (Öğrenci 3),
   - `scale_pos_weight` ve SMOTE ile sınıf ağırlıklarını dengelemek (Öğrenci 2),
   - Karar eşiğini (Decision Threshold) iş modelinin risk toleransına göre optimize etmektir (Öğrenci 4).

---

## 📈 7. Üretilen Görseller ve Rapor Dosyaları

Pipeline çalıştırıldığında `reports/` dizini altında aşağıdaki çıktılar otomatik üretilir:
* `01_class_distribution.png`: Lineer ve logaritmik sınıf dengesizliği kıyaslaması.
* `02_amount_time_distribution.png`: Normal vs Fraud işlemlerin tutar (log-scale) ve 24 saatlik gün içi zaman yoğunluğu.
* `03_feature_correlations.png`: Dolandırıcılık sınıfı ile en güçlü pozitif ve negatif korelasyona sahip özellikler.
* `04_confusion_matrices.png`: Modellerin Karmaşıklık Matrisleri (Doğru ve Yanlış Tahminler).
* `05_pr_and_roc_curves.png`: Precision-Recall ve ROC eğrilerinin yan yana karşılaştırmalı grafiği.
* `06_scale_experiment_results.png`: Veri Ölçeği vs AUPRC, Eğitim Süresi ve Maliyet-Fayda Analiz grafikleri.
* `reports/metrics/*.csv & *.json`: Sayısal sonuç tabloları ve akademik araştırma sorusu raporu.

---
*Geliştirici Ekip: Veri Bilimi ve Büyük Veri Mühendisliği Çalışma Grubu (Öğrenci 1 - 5)*
