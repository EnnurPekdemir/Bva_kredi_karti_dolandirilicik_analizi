# 🎓 PROJE 4: KREDİ KARTI DOLANDIRICILIK ANALİZİ
## 5 Öğrenci İçin Kapsamlı Eğitim, Konu Anlatımı ve Birebir Sunum Konuşma Rehberi

> **Problem:** Çok büyük işlem verisinde az sayıdaki fraud (dolandırıcılık) işleminin tespiti.  
> **Araştırma Sorusu:** *"Çok veri olması, nadir fraud olaylarını bulmayı gerçekten kolaylaştırıyor mu?"*

---

## 🧠 BÖLÜM 0: HEPİNİZİN BİLMESİ GEREKEN "ANA HİKAYE" (Büyük Resim)

Sunuma çıkmadan önce 5'iniz de şu mantığı adınız gibi bilmelisiniz:
1. **Olay ne?** Bir bankada saniyede binlerce kredi kartı harcaması yapılıyor. Bunların **%99.83'ü normal vatandaşın harcaması**, sadece **%0.17'si (yani her 588 işlemden sadece 1 tanesi) dolandırıcılık (Fraud)**.
2. **Zorluk ne?** Buna veri biliminde **Extreme Class Imbalance (Aşırı Sınıf Dengesizliği)** denir. Samanlıkta iğne aramaktır. Eğer standart bir yapay zeka modeline bu veriyi düz verirseniz, model tembellik yapar ve gelen her işleme *"Normal"* der geçer.
3. **Biz ne yaptık?** 5 kişilik bir veri bilimi ve büyük veri mühendisliği ekibi olarak uçtan uca bir boru hattı (pipeline) kurduk:
   - **Öğrenci 1** veriyi hazırladı ve dolandırıcıların gece saatlerinde & farklı tutarlarda harcama yaptığını keşfetti.
   - **Öğrenci 2** bu 1'e 587'lik dengesizliği çözdü ve test verisine bilgi sızmasını (**Data Leakage**) engelledi.
   - **Öğrenci 3** PySpark kullanarak müşterinin geçmiş davranışlarını (*son 1 saatte kaç işlem yaptı, normal ortalamasından ne kadar saptı?*) çıkardı.
   - **Öğrenci 4** modelleri eğitti ve neden **Accuracy (Doğruluk)** metriğinin bu problemde büyük bir tuzak olduğunu (**Accuracy Paradox**) kanıtlayıp **PR-AUC (AUPRC)** ve **Recall** metrikleriyle gerçek başarıyı ölçtü.
   - **Öğrenci 5** ise hocanın sorduğu ana araştırma sorusunu deneylerle test etti: Veriyi %10, %30, %60 ve %100 boyutlarında modele verdi ve **"Çok veri olmasının bir noktaya kadar işe yaradığını ama belirli bir eşikten sonra doyuma (plato noktasına) ulaştığını"** kanıtladı.

---

# 🧑‍🎓 ÖĞRENCİ 1: İşlem Veri Setini Hazırlar ve Veri Dağılımını İnceler
* **Sorumlu Olduğu Kod Dosyası:** `src/data_prep.py`
* **Arayüzdeki Sayfası:** Sayfa 1 (`Öğrenci 1: Veri & EDA`)
* **Göstereceği Grafikler:** `01_class_distribution.png` ve `02_amount_time_distribution.png`

### 📚 1. ÖNCE SEN ÖĞREN (Teknik Arka Plan - Bilmen Gerekenler)
* **Veri Setinde Neler Var?** Kaggle Kredi Kartı Dolandırıcılık formatında 32 sütunumuz var:
  - `Time`: İlk işlemden itibaren geçen saniye (Toplam 48 saatlik yani 2 günlük işlem akışı).
  - `V1, V2, ..., V28`: Banka müşterilerinin gizliliğini (KVKK / GDPR) korumak için gerçek özelliklerin (isim, lokasyon, mağaza kodu vb.) **PCA (Principal Component Analysis - Temel Bileşen Analizi)** ile şifrelenmiş/boyut indirgenmiş sayısal halleri.
  - `Amount`: İşlem tutarı (TL / Dolar).
  - `Class`: Hedef değişkenimiz. `0` = Normal İşlem, `1` = Dolandırıcılık (Fraud).
  - `customer_id`: PySpark'ta müşteri bazlı geçmiş analizi yapabilmek için simüle ettiğimiz müşteri kimliği.
* **Neden Logaritmik Ölçek (`log1p`) Kullandık?**
  - Normal işlemler 29.949 adet iken Fraud işlemleri sadece 51 adet. Normal bir grafik (Lineer ölçek) çizdiğinde 29.949'luk çubuk tavan yapıyor, 51'lik çubuk ise yerde görünmüyor bile! Bu yüzden grafiği **Logaritmik Ölçekte (`Log10`)** çizdik ki iki sınıfı aynı grafikte net görebilelim.
  - Aynı şekilde işlem tutarları (`Amount`) çok çarpık (sağa çarpık / right-skewed): Çoğu işlem 10-50 birim iken, bazı işlemler 10.000 birim. Bu çarpıklığı düzeltmek için $\log(1 + \text{Amount})$ dönüşümü uyguladık.

### 🎤 2. SUNUMDA BİREBİR KONUŞMA METNİ (Sıra Sana Geldiğinde Böyle Anlat)
> *"Merhaba hocam ve arkadaşlar. Projemizin ilk aşaması olan **Veri Hazırlığı ve Keşifsel Veri Analizi (EDA)** adımını ben gerçekleştirdim (`src/data_prep.py`).*
>
> *İlk olarak büyük ölçekli kredi kartı işlem verimizi incelediğimizde, bu problemin neden klasik bir sınıflandırma problemi olmadığını hemen görüyoruz. Veri setimizde işlemlerin **%99.83'ü normal (Class 0)** iken, yalnızca **%0.17'si dolandırıcılık (Class 1)** işlemidir. Yani **Dengesizlik Oranımız (Imbalance Ratio) 1'e 587'dir.** Her 1 dolandırıcı işleme karşılık 587 adet dürüst harcama bulunmaktadır.*
>
> *(Ekranda 1. Grafiği göstererek)* *İlk grafiğimizin sol tarafında gördüğünüz gibi, lineer ölçekte baktığımızda dolandırıcılık işlemleri neredeyse sıfır çizgisinde kaybolmaktadır. Bu yüzden sağ tarafta logaritmik ölçek kullanarak azınlık sınıfını görünür hale getirdik.*
>
> *(Ekranda 2. Grafiği göstererek)* *Veri dağılımını `Amount` (Tutar) ve `Time` (Zaman) ekseninde incelediğimizde ise çok kritik iki bulgu yakaladık:*
> 1. *Birincisi; normal harcamalar gün içinde sabah ve akşam saatlerinde düzenli bir çan eğrisi çizerken, **dolandırıcılık işlemleri özellikle gece yarısı ve sabaha karşı (kart sahipleri uyurken)** ciddi bir yoğunluk göstermektedir.*
> 2. *İkincisi; `Amount` değişkeni aşırı sağa çarpık bir dağılıma sahip olduğu için $\log(1+\text{Amount})$ dönüşümüyle inceledik. Dolandırıcıların ya kartın çalışıp çalışmadığını denemek için **çok ufak mikro tutarlar** çektiğini ya da limiti boşaltmak için **müşteri ortalamasının çok üstünde tutarlar** harcadığını tespit ettik.*
>
> *Ayrıca korelasyon analizimizde `V4` ve `V11` bileşenlerinin fraud ile pozitif, `V12`, `V14` ve `V17` bileşenlerinin ise güçlü negatif korelasyona sahip olduğunu belirledik. Elde ettiğim bu bulgular, benden sonraki arkadaşlarımın hangi dengeleme ve özellik mühendisliği adımlarını atacağına ışık tuttu. Şimdi sözü sınıf dengesizliği çözümleri için 2. arkadaşımıza bırakıyorum."*

### 🛡️ 3. HOCA SORARSA ŞOV YAPACAĞIN CEVAPLAR
* **Hoca:** *"Verideki V1..V28 nedir, neden açık isimleri yok?"*
  * **Sen:** *"Hocam, finansal verilerde kişisel verilerin gizliliği nedeniyle ham özellikler paylaşılmaz. Banka bu özellikleri PCA (Temel Bileşen Analizi) ile ortogonal bileşenlere dönüştürmüştür. Yalnızca `Time` ve `Amount` orijinal ölçeğinde bırakılmıştır."*
* **Hoca:** *"Amount değişkeninde neden ortalama yerine medyan ve logaritma baktınız?"*
  * **Sen:** *"Çünkü hocam işlem tutarlarında aşırı uç değerler (outliers) var. Çarpık dağılımlarda aritmetik ortalama yanıltıcı olur; bu yüzden çeyrekliklere (IQR/Medyan) ve logaritmik dağılıma baktık."*

---

# 🧑‍🎓 ÖĞRENCİ 2: Class Imbalance Problemini Analiz Eder ve Uygun Dengeleme Yaklaşımını Uygular
* **Sorumlu Olduğu Kod Dosyası:** `src/imbalance.py`
* **Arayüzdeki Sayfası:** Sayfa 2 (`Öğrenci 2: Dengesizlik & Sızıntı`)

### 📚 1. ÖNCE SEN ÖĞREN (Teknik Arka Plan - Bilmen Gerekenler)
* **Model Neden Bozulur?** Makine öğrenimi algoritmaları toplam hatayı (Loss) minimize etmeye çalışır. Verinin %99.83'ü `0` (Normal) olduğu için, model hiçbir şey öğrenmeyip herkese `0` dese bile hatası sadece %0.17 çıkar! Model azınlık sınıfını (Fraud) "gürültü" sanıp yok sayar.
* **3 Dengeleme Tekniğimiz Nedir?**
  1. **Random Undersampling (Rastgele Alt Örnekleme):** Çoğunluk sınıfından (Normal işlemlerden) rastgele kayıt silerek sayıyı dolandırıcılık sayısına yaklaştırmaktır. *Avantajı:* Eğitim çok hızlanır. *Dezavantajı:* Silinen normal işlemlerin içindeki önemli bilgileri kaybedebiliriz (Information Loss).
  2. **SMOTE (Synthetic Minority Over-sampling Technique):** Azınlık sınıfını (Fraud) kopyalamak yerine, uzayda birbirine yakın iki fraud işleminin arasına matematiksel olarak ($k$-NN en yakın komşuluk ile) **yeni sentetik fraud örnekleri** üretir. *Avantajı:* Bilgi kaybı olmaz. *Dezavantajı:* Eğitim verisi büyür ve bazen normal işlemlerin bölgesine taşarak yanlış alarmları artırabilir.
  3. **Algoritmik Dengeleme (`scale_pos_weight` / `class_weight`):** Veriyi hiç silmeyiz ve sentetik veri de üretmeyiz! Bunun yerine algoritmanın ceza fonksiyonuna (Loss function) deriz ki: *"Bak model, 1 tane Fraud işlemini kaçırırsan sana 1 normal işlemin **584 katı ceza** keserim!"* ($\text{scale\_pos\_weight} = \frac{\text{Negatif Sayısı}}{\text{Pozitif Sayısı}} = 584.37$).
* **EN KRİTİK KAVRAM: Data Leakage (Veri Sızıntısı) Nedir?**
  - Çoğu acemi öğrenci SMOTE'u tüm veri setine uygular, sonra `train_test_split` yapar. **BU BÜYÜK BİR HATADIR!**
  - Çünkü SMOTE tüm veriye uygulanırsa, test setindeki verilerin kopyaları eğitim setine sızar ve test setinin doğal 1:587 dengesizliği bozulur.
  - **Doğrusu:** Önce veriyi `%80 Train` ve `%20 Test` diye **Stratified (sınıf oranlarını koruyarak)** bölersin. Dengelemeyi **SADECE TRAIN** setine yaparsın! Test setine asla dokunmazsın.

### 🎤 2. SUNUMDA BİREBİR KONUŞMA METNİ (Sıra Sana Geldiğinde Böyle Anlat)
> *"Teşekkürler. İlk arkadaşımızın ortaya koyduğu **1'e 587'lik aşırı sınıf dengesizliği (Extreme Class Imbalance)** problemini analiz etmek ve çözmek benim görevimdi (`src/imbalance.py`).*
>
> *Eğer bu veriyi doğrudan modele verirsek, model çoğunluk sınıfı olan normal işlemlere aşırı uyum sağlar (bias) ve nadir dolandırıcılık işlemlerini göz ardı eder. Bunu aşmak için modülümde 3 farklı yaklaşımı karşılaştırmalı olarak kurguladım:*
> 1. *Birincisi **Random Undersampling**: Çoğunluk sınıfındaki normal işlemleri alt örnekleyerek denge sağlamak.*
> 2. *İkincisi **SMOTE**: Azınlık sınıfındaki dolandırıcılık örneklerinin $k$-en yakın komşularını kullanarak uzayda sentetik yeni fraud örnekleri üretmek.*
> 3. *Üçüncüsü ise **Algoritmik Maliyet Ağırlıklandırma (`scale_pos_weight` ve `balanced class_weight`)**: Veri hacmini yapay olarak değiştirmeden, karar ağaçlarının kayıp fonksiyonunda azınlık sınıfına **584.37 kat** daha yüksek hata cezası vermektir.*
>
> *Ancak hocam, benim bu modülde kurduğum en kritik mühendislik yapısı **`DataLeakageGuard` (Veri Sızıntısı Koruma)** katmanıdır. Literatürde yapılan en yaygın hata, SMOTE gibi dengeleme işlemlerini tüm veriye uygulayıp sonrasında Train-Test ayrımı yapmaktır. Bu yapıldığında hem eğitimden test kümesine bilgi sızar hem de test kümesinin gerçek dünya dağılımı bozulur.*
>
> *Bizim boru hattımızda ise veri önce **Stratified (katmanlı)** olarak %80 Eğitim ve %20 Test şeklinde izole edilmekte; SMOTE veya Undersampling **KESİNLİKLE yalnızca Eğitim (Train) kümesine** uygulanmaktadır. Test kümemiz ise gerçek dünyadaki bir bankanın canlı veri akışını temsil edecek şekilde el değmemiş 1:587 oranında bırakılmıştır. Şimdi sözü PySpark özellik mühendisliği için 3. arkadaşımıza devrediyorum."*

### 🛡️ 3. HOCA SORARSA ŞOV YAPACAĞIN CEVAPLAR
* **Hoca:** *"Peki bu 3 yöntemden hangisi en iyi sonucu verdi?"*
  * **Sen:** *"Hocam, ağaç tabanlı boosting modellerinde (LightGBM ve XGBoost) verinin orijinal dağılımını bozmayan **`scale_pos_weight` (Algoritmik Ağırlıklandırma)** ile **SMOTE (%10 stratejisi)** en yüksek AUPRC başarımını verdi. Özellikle `scale_pos_weight`, ekstra sentetik veri üretip eğitim süresini uzatmadığı için büyük veride en verimli yöntem oldu."*
* **Hoca:** *"Stratified Split ne demek, neden normal rastgele bölmediniz?"*
  * **Sen:** *"Hocam verimizde zaten çok az fraud var. Rastgele bölseydik şans eseri tüm fraudlar eğitim setine düşüp test setinde hiç fraud kalmayabilirdi. Stratified split, hem eğitim hem de test setinde %0.172'lik fraud oranının birebir aynı kalmasını garanti eder."*

---

# 🧑‍🎓 ÖĞRENCİ 3: Spark Üzerinde Müşteri ve Zaman Bazlı Davranış Özellikleri Üretir
* **Sorumlu Olduğu Kod Dosyası:** `src/spark_features.py`
* **Arayüzdeki Sayfası:** Sayfa 3 (`Öğrenci 3: PySpark Özellikler`)

### 📚 1. ÖNCE SEN ÖĞREN (Teknik Arka Plan - Bilmen Gerekenler)
* **Neden Davranışsal Özellik (Behavioral Feature Engineering) Üretiyoruz?**
  - Ham veride her işlem tek başına bir satırdır. Örneğin: *"Ahmet 5.000 TL harcadı."*
  - Sadece "5.000 TL" bilgisine bakarak bunun dolandırıcılık olup olmadığını bilemezsin! Eğer Ahmet her gün 10.000 TL harcayan zengin bir iş insanıysa bu normaldir. Ama Ahmet her gün 50 TL kantin harcaması yapan bir öğrenciyse ve gece 03:00'te üst üste 4 kere 5.000 TL çekildiyse bu **%100 dolandırıcılıktır!**
  - İşte müşterinin geçmişini o anki işleme bağlamaya **Davranışsal Özellik Mühendisliği** denir.
* **PySpark Window (Pencereleme) Fonksiyonları Nasıl Çalışır?**
  - Büyük veride (milyonlarca satırda) her müşteriyi kendi içinde (`partitionBy("customer_id")`) zamana göre sıraya dizeriz (`orderBy("Time_sec")`).
  - Ürettiğimiz **7 Yeni Özellik** şunlardır:
    1. `tx_count_last_1h`: Müşterinin son 1 saat (3600 sn) içinde yaptığı işlem sayısı (`rangeBetween(-3600, 0)`).
    2. `tx_count_last_6h`: Son 6 saat (21600 sn) içindeki işlem sayısı.
    3. `tx_count_last_24h`: Son 24 saat (86400 sn) içindeki işlem sayısı. *(Neden? Kartı çalan dolandırıcı, kart bloke edilmeden önce peş peşe hızlı işlemler yapar - Velocity Attack).*
    4. `customer_avg_amount`: Müşterinin o ana kadarki tüm geçmiş işlemlerinin ortalama tutarı.
    5. `amount_deviation`: Şu anki işlem tutarının, müşterinin geçmiş ortalamasından farkı ($\text{Amount} - \text{Avg}$).
    6. `amount_zscore`: Bu farkın standart sapmaya bölünmüş hali ($Z = \frac{\text{Amount} - \mu}{\sigma}$). Yani *"Bu harcama müşterinin normal karakterinden kaç kat saptı?"*
    7. `delta_time_sec`: `F.lag("Time_sec", 1)` fonksiyonu ile müşterinin bir önceki işleminden bu yana geçen saniye farkı. (Örn: İki işlem arası 3 saniye ise bot/kopya kart saldırısıdır).
* **PySpark ML Pipeline:** Son olarak `VectorAssembler` ve `StandardScaler` ile bu 37 özelliği makine öğrenimi modellerinin okuyabileceği vektör formatına çevirdik.

### 🎤 2. SUNUMDA BİREBİR KONUŞMA METNİ (Sıra Sana Geldiğinde Böyle Anlat)
> *"Teşekkür ederim. Benim görevim, ham işlem kayıtlarından PySpark kullanarak **müşteri ve zaman bazlı davranışsal özellikler (Behavioral Features)** üretmekti (`src/spark_features.py`).*
>
> *Gerçek dünya bankacılığında bir işlemin fraud olup olmadığını sadece o işlemin tutarına bakarak anlayamayız. Örneğin 5.000 TL'lik bir işlem bir müşteri için çok sıradanken, başka bir müşteri için büyük bir anomalidir. Dolayısıyla modelin müşterinin **geçmiş harcama karakterini** bilmesi gerekir.*
>
> *Büyük veri ölçeğinde bu hesaplamaları dağıtık yapabilmek için **PySpark SQL ve Window (Pencereleme)** mimarisini kullandım. Veriyi `customer_id` bazında bölümleyip (`partitionBy`) işlem zamanına göre sıralayarak (`orderBy`) şu 3 kritik davranış grubunda toplam **7 yeni özellik** türettim:*
>
> 1. ***Birincisi; Kayan Zaman Pencereleri (Rolling Window Counts):*** *`rangeBetween` kullanarak müşterinin **son 1 saat, son 6 saat ve son 24 saat** içinde yaptığı işlem sayılarını hesapladım. Çünkü kredi kartı çalındığında, dolandırıcılar kart iptal edilmeden önce kısa sürede çok sayıda ardışık işlem (Velocity Attack) gerçekleştirirler.*
> 2. ***İkincisi; Müşteri Bazlı Tutar Sapması ve Z-Skoru:*** *Her müşterinin o işlem anına kadarki kümülatif harcama ortalamasını (`customer_avg_amount`) ve standart sapmasını hesaplayarak, güncel işlemin müşterinin kendi geçmişinden ne kadar saptığını gösteren **Z-Score (`amount_zscore`)** özelliğini ürettim.*
> 3. ***Üçüncüsü; İşlemler Arası Geçen Süre (`delta_time_sec`):*** *PySpark `lag` fonksiyonu ile müşterinin bir önceki işlemiyle şu anki işlemi arasında geçen saniye farkını çıkardım.*
>
> *Son aşamada PySpark `VectorAssembler` ve `StandardScaler` kullanarak toplam özellik sayımızı 30'dan **37 zengin özelliğe** çıkardım ve modelleme aşamasına hazır hale getirdim. Şimdi sözü modelleme ve metrik analizi için 4. arkadaşımıza bırakıyorum."*

### 🛡️ 3. HOCA SORARSA ŞOV YAPACAĞIN CEVAPLAR
* **Hoca:** *"Müşteri ortalamasını hesaplarken tüm verinin ortalamasını mı aldınız?"*
  * **Sen:** *"Hayır hocam! Tüm zamanların ortalamasını alsaydık geleceğe ait işlemleri bugünkü işleme katmış olurduk (Look-ahead bias / zaman sızıntısı). Bunun yerine `rowsBetween(Window.unboundedPreceding, 0)` yani sadece o işlem anına kadar olan geçmiş işlemlerin kümülatif ortalamasını aldık."*
* **Hoca:** *"Neden PySpark kullandınız?"*
  * **Sen:** *"Çünkü hocam büyük ölçekli akan (streaming/batch) banka verilerinde milyonlarca kartın kayan pencere (rolling window) hesaplamaları tek çekirdekli yapılarda darboğaz yaratır. PySpark Window API, her müşterinin penceresini farklı executor'larda paralel olarak hesaplayabilmektedir."*

---

# 🧑‍🎓 ÖĞRENCİ 4: Fraud Sınıflandırma Modeli Geliştirir; Accuracy Yerine Precision, Recall, F1, AUPRC İnceler
* **Sorumlu Olduğu Kod Dosyası:** `src/modeling.py`
* **Arayüzdeki Sayfası:** Sayfa 4 (`Öğrenci 4: Modeller & Paradoks`)
* **Göstereceği Grafikler:** `04_confusion_matrices.png` ve `05_pr_and_roc_curves.png`

### 📚 1. ÖNCE SEN ÖĞREN (Teknik Arka Plan - Bilmen Gerekenler)
* **Hangi Modelleri Eğittik?**
  1. `Naive Majority Baseline`: Hiçbir şey öğrenmeyen, gelen her işleme körü körüne *"0 (Normal)"* diyen kukla model. (Doğruluk Paradoksunu kanıtlamak için kurduk).
  2. `Random Forest`: Dengeli sınıf ağırlıklı (`class_weight="balanced"`) paralel karar ağaçları topluluğu (Bagging).
  3. `LightGBM`: Microsoft'un geliştirdiği, yaprak bazlı (leaf-wise) büyüyen çok hızlı Gradient Boosting modeli.
  4. `XGBoost`: Seviye bazlı (depth-wise) büyüyen, regülarizasyonu çok güçlü endüstri standardı Gradient Boosting modeli.
* **Accuracy Paradox (Doğruluk Paradoksu) Nedir? (SUNUMUN EN HAVALI KISMI!)**
  - Test setimizde 6.000 işlem var: Bunun **5.990 tanesi Normal**, **10 tanesi Fraud**.
  - Bizim kukla `Naive Model` gelen 6.000 işlemin hepsine *"Normal"* diyor.
  - Şimdi Accuracy (Doğruluk) hesaplayalım: $\frac{5990}{6000} = \%99.833$!
  - Kağıt üstünde model **%99.83 doğru** çalışıyor görünüyor! Ama yakaladığı dolandırıcı sayısı kaç? **0 (Sıfır)!** 10 dolandırıcının 10'u da bankayı soyup soğana çevirdi!
  - İşte doğruluğun (%99.83) çok yüksek çıkıp modelin aslında tamamen işe yaramaz olmasına **Accuracy Paradox (Doğruluk Paradoksu)** denir!
* **Peki Doğru Metrikler Nelerdir?**
  - **Recall (Duyarlılık / Yakalama Oranı):** $\frac{TP}{TP + FN}$. *"Gerçek dolandırıcıların yüzde kaçını yakaladık?"* (Bankanın soyulmaması için en kritik metrik! Bizim XGBoost ve Random Forest modellerimiz **%100 Recall** ile 10 fraudun 10'unu da yakaladı, LightGBM %90 yakaladı).
  - **Precision (Kesinlik / Hassasiyet):** $\frac{TP}{TP + FP}$. *"Modelin 'Bu işlem dolandırıcı!' diye bloke ettiği kartların yüzde kaçı gerçekten dolandırıcı?"* (Düşük olursa dürüst müşterinin kartını kasada bloke ederiz, müşteri bankaya kızar).
  - **F1-Score ve F2-Score:** Precision ve Recall'un harmonik ortalamasıdır. F2-Score, Recall'a (dolandırıcıyı kaçırmamaya) Precision'dan 2 kat daha fazla önem verir.
  - **Neden ROC-AUC Değil de AUPRC (PR-AUC - Precision-Recall Eğrisi Alanı)?**
    * ROC eğrisi, **False Positive Rate** ($FPR = \frac{FP}{FP + TN}$) kullanır. Paydada $TN$ (Normal işlemler = 5.990) var. Payda çok devasa olduğu için model 100 tane yanlış alarm ($FP=100$) verse bile $FPR = \frac{100}{6000} = 0.016$ gibi çok küçük çıkar ve **ROC-AUC %98-99 gibi yanıltıcı derecede yüksek görünür**.
    * **AUPRC (PR-AUC)** ise formülüne $TN$'yi (Normal işlemleri) hiç sokmaz! Sadece Precision ve Recall'a bakar. Yanlış alarmları acımasızca cezalandırır. Bu yüzden dengesiz veride **altın standart AUPRC'dir!**

### 🎤 2. SUNUMDA BİREBİR KONUŞMA METNİ (Sıra Sana Geldiğinde Böyle Anlat)
> *"Teşekkürler. İlk üç arkadaşımın hazırladığı, dengelediği ve PySpark ile zenginleştirdiği veri seti üzerinde **Sınıflandırma Modellerini Geliştirmek ve Kapsamlı Metrik Analizi Yapmak** benim görevimdi (`src/modeling.py`).*
>
> *Bu aşamada endüstri standardı olan **Random Forest, LightGBM ve XGBoost** algoritmalarını eğittim. Ayrıca değerlendirme metriklerimizin önemini kanıtlamak için her işleme istisnasız 'Normal' tahmininde bulunan bir **Naive Majority (Taban)** modeli kurdum.*
>
> *(Ekranda metrikleri ve Confusion Matrix'i göstererek)* *Hocam, sonuçlarımız literatürdeki **Accuracy Paradox (Doğruluk Paradoksu)** kavramını çarpıcı bir biçimde kanıtlamaktadır: Hiçbir makine öğrenimi yapmayan Naive modelimiz, test setindeki 6.000 işlemin tamamına 'Normal' diyerek **%99.83 Accuracy (Doğruluk)** elde etmiştir. Ancak Confusion Matrix'e baktığımızda test setindeki **10 dolandırıcılığın 0 tanesini yakalamış, tamamını kaçırmıştır (Recall = %0.0)**.*
>
> *Bu nedenle kredi kartı dolandırıcılığında Accuracy metriği tamamen anlamsızdır. Biz modellerimizi değerlendirirken 4 kritik ölçüte odaklandık:*
> 1. ***Recall (Yakalama Oranı):*** *Gerçek dolandırıcılıkların kaçını yakaladık? (Kaçırılan her fraud doğrudan finansal kayıptır).*
> 2. ***Precision (Kesinlik):*** *Fraud dediğimiz işlemlerin kaçı gerçekten fraud? (Yanlış alarmlar müşteri memnuniyetsizliği yaratır).*
> 3. ***F1 ve F2-Score:*** *Precision ve Recall arasındaki denge.*
> 4. ***Ve en önemlisi AUPRC (Area Under Precision-Recall Curve):*** *(Ekranda 5. Grafiği - PR ve ROC eğrilerini göstererek)* *Sağdaki ROC eğrisine baktığımızda tüm modeller %95-100 arası çok yüksek görünmektedir; çünkü ROC eğrisi devasa sayıdaki True Negative'lerden (Normal işlemlerden) etkilenip iyimser bir tablo çizer. Soldaki **Precision-Recall (AUPRC)** eğrimiz ise sadece pozitif sınıfa odaklanır. Burada **XGBoost (%100 Recall, %90.9 Precision, 1.00 AUPRC)** ve **Random Forest** modellerimizin en üstün performansı gösterdiğini net olarak kanıtladık. Şimdi sözü araştırma sorumuzun deneyleri için 5. arkadaşımıza bırakıyorum."*

### 🛡️ 3. HOCA SORARSA ŞOV YAPACAĞIN CEVAPLAR
* **Hoca:** *"Banka yöneticisi olsan Precision'ı mı daha çok önemserdin Recall'u mu?"*
  * **Sen:** *"Hocam, finansal maliyete göre **Recall** önceliklidir; çünkü bir dolandırıcılığı kaçırmanın (False Negative) maliyeti binlerce dolarlık kayıptır. Yanlış alarmın (False Positive) maliyeti ise müşteriye giden bir 'İşlemi siz mi yaptınız?' SMS onay mesajıdır. Bu yüzden biz raporumuzda Recall'a iki kat ağırlık veren **F2-Score**'u da hesapladık."*
* **Hoca:** *"LightGBM neden XGBoost'tan biraz daha düşük Precision verdi?"*
  * **Sen:** *"Çünkü hocam LightGBM yaprak bazlı (leaf-wise) agresif büyüdüğü ve `scale_pos_weight=584` ile azınlık sınıfına çok sert odaklandığı için 10 fraudun 9'unu yakalarken 7 adet de şüpheli normal işleme alarm (False Positive) verdi. XGBoost ise derinlik bazlı (depth-wise) regülarizasyonu sayesinde sadece 1 yanlış alarm ile 10'da 10 yakalama sağladı."*

---

# 🧑‍🎓 ÖĞRENCİ 5: Veri Büyüklüğü ve Sınıf Dengesizliğinin Eğitim Süresi ve Model Performansına Etkisini Analiz Eder
* **Sorumlu Olduğu Kod Dosyası:** `src/experiments.py`
* **Arayüzdeki Sayfası:** Sayfa 5 (`Öğrenci 5: Ölçek & Araştırma`)
* **Göstereceği Grafik ve Tablo:** `06_scale_experiment_results.png` ve Deney Kademeleri Tablosu

### 📚 1. ÖNCE SEN ÖĞREN (Teknik Arka Plan - Bilmen Gerekenler)
* **Projemizin Kalbi (Araştırma Sorusu):**
  > *"Çok veri olması, nadir fraud olaylarını bulmayı gerçekten kolaylaştırıyor mu?"*
* **Bu Soruyu Nasıl Test Ettik? (Kontrollü Deney Tasarımı):**
  - Eğitim verimizi sınıf dengesizliği oranını (1:587) sabit tutarak 4 farklı ölçekte alt kümelere böldük:
    1. **%10 Veri Ölçeği:** 2.400 işlem (İçinde sadece **4 adet** Fraud var!)
    2. **%30 Veri Ölçeği:** 7.200 işlem (İçinde **12 adet** Fraud var)
    3. **%60 Veri Ölçeği:** 14.400 işlem (İçinde **25 adet** Fraud var)
    4. **%100 Veri Ölçeği:** 24.000 işlem (İçinde **41 adet** Fraud var)
  - Her ölçekte modeli sıfırdan eğittik, **Eğitim Süresini (Training Latency)**, **Çıkarım Süresini (Inference ms)** ve aynı test seti üzerindeki **Recall / AUPRC** skorlarını ölçtük.
* **Deneyden Çıkan 3 Büyük Bilimsel Bulgu (Bunu Anlattığında Hoca Hayran Kalacak!):**
  1. **Küçük Veride (%10 - %30) Kararsızlık ve Ezber:** Veri %10 iken modelin önünde sadece 4 tane fraud örneği var. Model genel kuralı öğrenemiyor; Recall %60'ta kalıyor (%40 dolandırıcı kaçıyor).
  2. **Kritik Sıçrama (%60 Ölçeği):** Veri %60'a çıktığında eğitimdeki fraud sayısı 25'e ulaşıyor. Model artık dolandırıcılık örüntülerinin varyansını (çeşitliliğini) kavrıyor ve **Recall %100'e sıçrıyor!**
  3. **Doyum Noktası / Azalan Verimler Yasası (%60 -> %100 Diminishing Returns):** Veriyi %60'tan %100'e çıkardığımızda eğitim süresi **0.58 saniyeden 2.33 saniyeye (4 katına)** fırlıyor! Peki başarım (AUPRC / Recall) 4 kat artıyor mu? **HAYIR!** Başarım plato yapıyor (hatta eklenen binlerce yeni normal işlemin yarattığı False Positive gürültüsü yüzünden AUPRC 0.55'ten 0.50'ye hafif geriliyor).
* **Neden Çok Veri Tek Başına Yetmez?**
  - Çünkü veriyi 10 kat büyüttüğünde dolandırıcı sayısı 10 kat artıyor ama **normal işlem sayısı da 10 kat artıyor!** Dengesizlik oranı (1:587) hiç değişmiyor. Sisteme milyonlarca yeni normal işlem girdikçe, içlerinden dolandırıcıya benzeyen tuhaf normal işlemler (False Positive gürültüsü) çıkıyor ve modelin Precision değerini baskılıyor.

### 🎤 2. SUNUMDA BİREBİR KONUŞMA METNİ (Sıra Sana Geldiğinde Böyle Anlat)
> *"Teşekkürler. Projemizin son aşamasında, tüm çalışmamızın temelini oluşturan araştırma sorusunu deneysel olarak test ettim (`src/experiments.py`):*  
> ***'Çok veri olması, nadir fraud olaylarını bulmayı gerçekten kolaylaştırıyor mu?'***
>
> *Bu soruya bilimsel ve kanıta dayalı bir yanıt verebilmek için kontrollü bir ölçek deneyi tasarladım. Eğitim veri setimizi, 1'e 587 olan sınıf dengesizliği oranını koruyarak **%10, %30, %60 ve %100** olmak üzere 4 farklı büyüklük kademesinde eğittim ve her kademede **Eğitim Süresi (Latency)** ile **Model Başarımını (Recall ve AUPRC)** ölçtüm.*
>
> *(Ekranda 6. Grafiği - 3'lü Ölçek Analizini ve Tabloyu göstererek)* *Elde ettiğimiz sonuçlar bize çok net bir mühendislik gerçeğini göstermektedir:*
>
> ***Araştırma sorumuzun cevabı: KISMEN EVET, ANCAK BELİRLİ BİR EŞİKTEN SONRA DOYUMA ULAŞMAKTADIR (Diminishing Returns)!***
>
> *Bunun 3 temel nedeni vardır:*
> 1. ***Birincisi; Mutlak Azınlık Sayısı Etkisi:*** *Tablomuzda görüldüğü üzere %10 ve %30 veri ölçeğinde model yalnızca 4 ila 12 adet fraud örneği görmektedir. Bu kadar az örnekle karar sınırları tam öğrenilememekte ve Recall %60 seviyesinde kalmaktadır. Veri %60 ölçeğine (25 fraud örneğine) ulaştığında ise model örüntüyü yakalamakta ve **Recall %100'e** ulaşmaktadır.*
> 2. ***İkincisi; Gürültü ve False Positive Baskısı:*** *Veri büyüklüğünü artırmak sınıf dengesizliği oranını (1:587) değiştirmemektedir. Sisteme gelen her 1 yeni fraud işlemine karşılık 587 yeni normal işlem gelmektedir. Bu devasa normal işlem havuzu, sınır bölgelerde yanlış alarmları (False Positives) tetikleyerek Precision ve AUPRC değerini baskılamaktadır.*
> 3. ***Üçüncüsü; Maliyet vs Performans Doyumu (Trade-off):*** *Grafiğimizin ortasında ve sağında gördüğünüz gibi, %60 ölçeğinden %100 ölçeğine geçildiğinde eğitim süresi **4 katına (0.58 saniyeden 2.33 saniyeye)** çıkarken, model başarımı plato yapmakta ve ek bir kazanç sağlamamaktadır.*
>
> ***Sonuç olarak hocam;*** *nadir dolandırıcılık tespitinde sadece 'daha fazla ham veri biriktirmek' tek başına işi kolaylaştırmaz ve hesaplama maliyetini şişirir. Asıl başarı; **3. arkadaşımızın yaptığı gibi PySpark ile müşteri davranış özellikleri üretmek, 2. arkadaşımızın yaptığı gibi akıllı ağırlıklandırma kullanmak ve 4. arkadaşımızın gösterdiği gibi AUPRC odaklı model kurmaktan** geçmektedir. Bizi dinlediğiniz için teşekkür ederiz, sorularınız varsa memnuniyetle yanıtlayabiliriz."*

### 🛡️ 3. HOCA SORARSA ŞOV YAPACAĞIN CEVAPLAR
* **Hoca:** *"O zaman bankalara ne önerirsiniz? Milyarlarca satır verinin hepsini eğitime sokmasınlar mı?"*
  * **Sen:** *"Kesinlikle hocam! Tüm geçmiş yılların milyarlarca normal işlemini ham olarak modele yığmak hem eğitimi saatlerce yavaşlatır hem de eski yıllardaki enflasyon/harcama alışkanlıkları değiştiği için (Concept Drift) modeli bozar. Bunun yerine yakın geçmişteki optimal veri penceresini alıp, PySpark ile özet davranış özelliklerine dönüştürerek eğitmek hem çok daha hızlı hem de çok daha yüksek AUPRC sağlar."*

---

## ⚡ SUNUM GÜNÜ BİLGİSAYAR BAŞINDA "ADIM ADIM" AKIŞ PLANI

1. **Projeksiyona Bağlanın:**
   - Proje klasöründeki `run_dashboard.bat` dosyasına çift tıklayın (veya terminalde `python app.py` yazın).
   - Tarayıcıda **5 Sayfalık Sunum Arayüzümüz** açılacak.
2. **Görünümü Ayarlayın:**
   - Sağ üstteki **"Tam Ekran (F11)"** butonuna basın.
   - Eğer sınıf çok aydınlıksa ve projeksiyon soluk görünüyorsa sağ üstteki **"Projektör Açık Modu"** butonuna basın (bembeyaz, çok net bir ekrana dönüşür).
3. **Sırayla Anlatın:**
   - **Öğrenci 1** Sayfa 1'de konuşmasını yapar -> Klavyeden **Sağ Ok (→)** tuşuna basar.
   - **Öğrenci 2** Sayfa 2'de konuşmasını yapar -> **Sağ Ok (→)** tuşuna basar.
   - **Öğrenci 3** Sayfa 3'te konuşmasını yapar -> **Sağ Ok (→)** tuşuna basar.
   - **Öğrenci 4** Sayfa 4'te konuşmasını yapar -> **Sağ Ok (→)** tuşuna basar.
   - **Öğrenci 5** Sayfa 5'te araştırma sorusunun cevabını verir ve kapanışı yapar.
4. **Final Şovu (Canlı Çalıştırma):**
   - Hoca *"Kodun çalıştığını görelim"* derse sağ üstteki yeşil **"⚡ Canlı Çalıştır (Live Demo)"** butonuna basın, ardından **"Canlı Pipeline'ı Başlat"** deyin. Ekranda 5 adımın canlı çalıştığını ve metrik tablosunun basıldığını gösterip tam puanı alın!
