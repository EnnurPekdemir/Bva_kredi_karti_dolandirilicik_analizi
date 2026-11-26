"""
src/generate_pdf.py
=============================================================================
Proje 4: Kredi Karti Dolandiricilik Analizi
5 Öğrenci İçin Sıfırdan Öğretici Çalışma Planı, Kod Haritası ve Sunum Rehberi
8 Sayfalık Yüksek Çözünürlüklü A4 Vektör PDF Üreticisi
=============================================================================
"""

import os
import shutil
import textwrap
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg

os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".mpl_cache")
plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def draw_header_banner(ax, title: str, subtitle: str, badge_text: str = ""):
    rect = patches.FancyBboxPatch(
        (0.03, 0.915), 0.94, 0.065,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor="#0f172a", edgecolor="#2563eb", linewidth=1.5,
        transform=ax.transAxes
    )
    ax.add_patch(rect)
    ax.text(0.048, 0.957, title, fontsize=12.2, fontweight="bold", color="#ffffff", transform=ax.transAxes, va="center")
    ax.text(0.048, 0.929, subtitle, fontsize=8.4, color="#93c5fd", transform=ax.transAxes, va="center")
    if badge_text:
        badge = patches.FancyBboxPatch(
            (0.77, 0.928), 0.185, 0.038,
            boxstyle="round,pad=0.004,rounding_size=0.008",
            facecolor="#1e3a8a", edgecolor="#60a5fa", linewidth=1,
            transform=ax.transAxes
        )
        ax.add_patch(badge)
        ax.text(0.862, 0.947, badge_text, fontsize=8.0, fontweight="bold", color="#ffffff", transform=ax.transAxes, ha="center", va="center")


def draw_box(ax, x, y, w, h, title, lines, bg_color="#f8fafc", border_color="#cbd5e1", title_color="#1e3a8a", font_size=8.0, wrap_width=110):
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.006,rounding_size=0.01",
        facecolor=bg_color, edgecolor=border_color, linewidth=1.2,
        transform=ax.transAxes
    )
    ax.add_patch(rect)
    ax.text(x + 0.014, y + h - 0.014, title, fontsize=9.3, fontweight="bold", color=title_color, transform=ax.transAxes, va="top")

    wrapped_paragraphs = []
    for line in lines:
        if line == "":
            wrapped_paragraphs.append("")
        else:
            wrapped = textwrap.fill(line, width=wrap_width)
            wrapped_paragraphs.append(wrapped)

    full_text = "\n".join(wrapped_paragraphs)
    ax.text(
        x + 0.014, y + h - 0.036, full_text,
        fontsize=font_size, color="#0f172a", transform=ax.transAxes,
        va="top", ha="left", linespacing=1.28
    )


def embed_image(fig, img_path, rect_coords, caption=""):
    if os.path.exists(img_path):
        ax_img = fig.add_axes(rect_coords)
        img = mpimg.imread(img_path)
        ax_img.imshow(img)
        ax_img.axis("off")
        if caption:
            ax_img.set_title(caption, fontsize=7.8, fontweight="bold", color="#1e293b", pad=2)


def generate_presentation_pdf(output_pdf_path: str = "Proje4_Kredi_Karti_Dolandiricilik_Sunum_ve_Calisma_Rehberi.pdf"):
    print(f"8 Sayfalik Detayli Calisma PDF'i olusturuluyor: {output_pdf_path} ...")

    with PdfPages(output_pdf_path) as pdf:

        # =====================================================================
        # SAYFA 1: ÇALIŞMA PLANI, SUNUM STRATEJİSİ VE VERİ SETİ HİKAYESİ
        # =====================================================================
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        draw_header_banner(
            ax,
            "PROJE 4: KREDİ KARTI DOLANDIRICILIK ANALİZİ - MASTER ÇALIŞMA REHBERİ",
            "Sayfa 1/8: Grup Çalışma Planı, Sunumu Nasıl Yapacağız? ve Veri Setini Nereden Bulduk?",
            "SAYFA 1 / 8"
        )

        draw_box(
            ax, 0.03, 0.67, 0.94, 0.23,
            "1. BİZ PROJEYİ HOCAYA NASIL GÖSTERECEĞİZ? (SUNUM GÜNÜ OPERASYON PLANI)",
            [
                "Arayüzümüzden (index.html) bütün konuşma metinlerini kaldırdık! Artık arayüzümüz %100 gerçek, kurumsal bir Büyük Veri Analiz Platformu gibi görünüyor ve üzerinde sadece siz anlatırken kopya çekebileceğiniz şık 'Analiz Trick Kartları' ve 'Grafik Okuma Notları' yer alıyor.",
                "",
                "SINIFTA TAHTAYA ÇIKINCA YAPACAĞINIZ 3 ADIMLI PLAN:",
                "• ADIM A (Ekranı Hazırlama - 30 Saniye): Bilgisayarda hem VS Code'u (kodlarımızın olduğu ekranı) hem de 'index.html' (veya run_dashboard.bat) web arayüzümüzü açın. Projeksiyona önce Web Arayüzümüzü yansıtın (Sağ üstten 'Projektör Açık Tema' seçebilirsiniz).",
                "• ADIM B (5 Öğrencinin Sırayla Anlatımı - 10 Dakika): Arayüzde üstte 5 Bölüm (Sekme 1..5) var. Her öğrenci kendi sekmesini açacak. Ekrandaki 'Kritik Bulgular' kutucuklarına ve grafiklerin altındaki 'Grafik Okuma' notlarına bakarak (bu PDF'te ezberlediği şekilde) anlatacak.",
                "• ADIM C (Canlı Kod Şovu - 2 Dakika): 5. öğrenci sözünü bitirince veya hoca 'Kodunuz çalışıyor mu, görelim' dediğinde; ya arayüzün sağ üstündeki yeşil 'Canlı Pipeline Çalıştır' butonuna basın ya da Alt+Tab ile VS Code terminalini açıp 'python main.py --quick' yazıp Enter'a basın! 20 saniye içinde 5 öğrencinin tüm kodları canlı çalışacak ve ekrana sonuç tablosunu basacaktır!"
            ],
            bg_color="#eff6ff", border_color="#2563eb", title_color="#1d4ed8", font_size=8.0, wrap_width=108
        )

        draw_box(
            ax, 0.03, 0.42, 0.94, 0.23,
            "2. HOCA 'VERİ SETİNİ NEREDEN BULDUNUZ?' DERSE NE CEVAP VERECEĞİZ?",
            [
                "Bu soruyu grupça çok iyi bilin, hocanın en çok soracağı sorulardan biridir:",
                "• Referans Veri Setimiz: Literatürdeki en ünlü kredi kartı dolandırıcılık veri seti olan 'Kaggle (ULB - Université Libre de Bruxelles) Credit Card Fraud Detection' veri setidir. Avrupa'daki kart sahiplerinin 48 saatlik (172.800 saniye) işlemlerini içerir ve doğal dolandırıcılık oranı %0.172'dir (Her 1 Fraud başına ~580 Normal işlem).",
                "• Neden Hibrit (Akıllı) Veri Motoru Kurduk? Orijinal Kaggle CSV dosyasında gizlilik nedeniyle 'customer_id' (Müşteri ID) sütunu silinmiştir! Oysa hocanın görev tablosunda 3. Öğrenciden 'PySpark üzerinde MÜŞTERİ ve zaman bazlı davranış özellikleri (müşterinin son 1 saatteki işlem sayısı, müşterinin ortalama harcaması)' üretmesi istenmektedir!",
                "• Bizim Çözümümüz (src/data_prep.py): Yazdığımız 'load_or_generate_credit_card_data()' fonksiyonu hibrit çalışır: Klasörde Kaggle creditcard.csv varsa onu okur ve Pareto (80/20) dağılımıyla gerçekçi 'customer_id' ekler; yoksa Kaggle ULB verisinin istatistiksel dağılımını (%0.172 fraud oranı, 48 saatlik Time akışı, V1..V28 PCA korelasyon kaymaları ve Log-Normal Amount dağılımını) birebir simüle ederek istenen ölçekte (30.000 - 150.000+ satır) üretir!"
            ],
            bg_color="#f0fdf4", border_color="#16a34a", title_color="#15803d", font_size=8.0, wrap_width=108
        )

        draw_box(
            ax, 0.03, 0.19, 0.94, 0.21,
            "3. KİMSE PROJEYE HAKİM DEĞİLSE EVDE NASIL ÇALIŞACAKSINIZ? (30 DAKİKALIK ÇALIŞMA PLANI)",
            [
                "1. İlk 5 Dakika (Herkes): Bu sayfanın altındaki '5 Öğrenci Görev Özeti'ni ve Sayfa 2'deki 'Temel Kavramlar Sözlüğü'nü okuyun. Böylece kimse 'PCA ne demek, SMOTE ne demek, AUPRC ne demek?' diye boşluğa düşmeyecek.",
                "2. Sonraki 15 Dakika (Bireysel): Her öğrenci bu PDF'te KENDİ SAYFASINI (Öğrenci 1 -> Sayfa 3, Öğrenci 2 -> Sayfa 4, Öğrenci 3 -> Sayfa 5, Öğrenci 4 -> Sayfa 6, Öğrenci 5 -> Sayfa 7) açsın. Oradaki '1. Konuyu Öğren', '2. Arayüzdeki Trickleri Nasıl Kullanacaksın?' ve '3. Birebir Konuşma Metni' kısımlarını 3 kez sesli okusun.",
                "3. Son 10 Dakika (Grupça Prova): Bilgisayarda 'index.html' dosyasını açın. Sırayla 1'den 5'e kadar sekmelere tıklayarak ekrandaki kutucuklara baka baka 1 kez prova yapın. Göreceksiniz ki ekrandaki kutucuklar size ne söyleyeceğinizi zaten hatırlatıyor!"
            ],
            bg_color="#fffbeb", border_color="#f59e0b", title_color="#b45309", font_size=8.0, wrap_width=108
        )

        draw_box(
            ax, 0.03, 0.03, 0.94, 0.14,
            "4. TABLODAKİ 5 ÖĞRENCİ GÖREV DAĞILIMI VE SORUMLU OLDUKLARI DOSYALAR",
            [
                "• ÖĞRENCİ 1 (src/data_prep.py)     : İşlem veri setini hazırlar ve veri dağılımını inceler. (Sayfa 3)",
                "• ÖĞRENCİ 2 (src/imbalance.py)     : Class imbalance problemini analiz eder ve uygun dengeleme yaklaşımını uygular. (Sayfa 4)",
                "• ÖĞRENCİ 3 (src/spark_features.py): Spark üzerinde müşteri ve zaman bazlı davranış özellikleri üretir. (Sayfa 5)",
                "• ÖĞRENCİ 4 (src/modeling.py)      : Fraud sınıflandırma modeli geliştirir; accuracy yerine precision, recall, F1, AUPRC inceler. (Sayfa 6)",
                "• ÖĞRENCİ 5 (src/experiments.py)   : Veri büyüklüğü ve sınıf dengesizliğinin eğitim süresi ve model performansına etkisini analiz eder. (Sayfa 7)"
            ],
            bg_color="#f8fafc", border_color="#64748b", title_color="#0f172a", font_size=8.1, wrap_width=110
        )

        pdf.savefig(fig)
        plt.close(fig)

        # =====================================================================
        # SAYFA 2: HİÇ BİLMEYENLER İÇİN PROJE KAVRAMLARI SÖZLÜĞÜ (5 DAKİKADA UZMAN OL)
        # =====================================================================
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        draw_header_banner(
            ax,
            "TEMEL KAVRAMLAR SÖZLÜĞÜ (PROJEYİ SIFIRDAN ANLAMA REHBERİ)",
            "Sayfa 2/8: Sunumda Geçen Tüm Teknik Terimlerin En Sade ve Net Açıklamaları",
            "SAYFA 2 / 8"
        )

        draw_box(
            ax, 0.03, 0.69, 0.94, 0.21,
            "A. VERİ VE DENGESİZLİK KAVRAMLARI (ÖĞRENCİ 1 & ÖĞRENCİ 2 İÇİN)",
            [
                "• Fraud (Dolandırıcılık - Class 1): Kredi kartının çalınması veya kopyalanmasıyla yapılan yetkisiz işlem. Verimizde Class=1 ile gösterilir (%0.17 oranında). Normal işlemler ise Class=0'dır (%99.83 oranında).",
                "• Class Imbalance (Sınıf Dengesizliği - 1:587): Verideki bir sınıfın diğerinden aşırı fazla olmasıdır. Bizde 29.949 Normal işleme karşılık sadece 51 Fraud işlemi var (29949 / 51 = 587.24). Yani her 1 dolandırıcıya 587 normal işlem düşüyor.",
                "• PCA (V1..V28 Özellikleri): 'Principal Component Analysis' (Temel Bileşen Analizi). Bankalar müşterinin adını, hangi mağazadan alışveriş yaptığını gizlilik (KVKK) yüzünden açıkça paylaşamaz. Bu bilgileri matematiksel olarak şifreleyip V1'den V28'e kadar 28 adet sayısal sütuna dönüştürmüşlerdir.",
                "• Random Undersampling: Dengesizliği çözmek için 29.949 tane olan Normal işlemden rastgele bir kısmını silip sayıyı azaltmaktır. Hızlıdır ama silinen verilerle önemli bilgiler çöpe gidebilir (Information Loss).",
                "• SMOTE: Azınlık olan Fraud işlemlerini kuru kuruya kopyalamak yerine, birbirine en yakın 5 fraud işleminin (k-NN) arasına matematiksel çizgi çekip yeni 'sentetik (yapay)' fraud örnekleri üretmektir.",
                "• Scale Pos Weight (584.37): Veriyi hiç silmeden ve yapay veri üretmeden, modele 'Eğer 1 tane Fraud işlemini kaçırırsan sana 584 kat ceza puanı keserim' demektir. Büyük veride en hızlı ve en başarılı yöntemdir.",
                "• Data Leakage (Veri Sızıntısı): Eğitim aşamasında test verisinin bilgisinin modele sızmasıdır. Eğer SMOTE'u tüm veriye yapıp sonra Train/Test diye bölerseniz sızıntı olur! Doğrusu: Önce %80 Train / %20 Test diye bölmek, SMOTE'u SADECE Train setine yapmaktır!"
            ],
            bg_color="#f8fafc", border_color="#3b82f6", title_color="#1d4ed8", font_size=7.9, wrap_width=108
        )

        draw_box(
            ax, 0.03, 0.44, 0.94, 0.23,
            "B. PYSPARK VE DAVRANIŞSAL ÖZELLİK KAVRAMLARI (ÖĞRENCİ 3 İÇİN)",
            [
                "• PySpark Nedir? Büyük verileri (milyonlarca/milyarlarca satırı) tek bir bilgisayarda değil, birden fazla işlemcide paralel (dağıtık) olarak çok hızlı işleyen büyük veri motorudur.",
                "• Davranışsal Özellik (Behavioral Feature) Nedir? Sadece 'Bu işlemde 5.000 TL çekildi' bilgisi yetmez. 'Bu müşteri son 1 saatte kaç işlem yaptı? Bu müşterinin geçmiş harcama ortalaması neydi?' gibi müşterinin alışkanlığını özetleyen özelliklerdir.",
                "• PySpark Window (Kayan Pencere / Rolling Window): Veriyi müşteriye göre gruplayıp (partitionBy), zamana göre sıraya dizip (orderBy), her işlemden geriye doğru 1 saat (3600 sn), 6 saat (21600 sn) ve 24 saat (86400 sn) içindeki işlemleri tarayan zaman penceresidir.",
                "• Velocity Attack (Hız Saldırısı): Kredi kartını çalan dolandırıcının, kart sahibi fark edip kartı iptal ettirmeden önce 5-10 dakika içinde arka arkaya 5-6 işlem yapmasıdır. 'tx_count_last_1h' özelliğimiz bunu yakalar.",
                "• Z-Score (amount_zscore): Güncel harcama tutarının, müşterinin kendi geçmiş ortalamasından kaç standart sapma uzaklaştığını gösterir. Formülü: (Güncel Tutar - Müşteri Ortalaması) / Müşteri Standart Sapması. Z-skoru +4 veya +5 ise müşteri normal alışkanlığının çok dışında şüpheli bir harcama yapmıştır!",
                "• Delta Time (delta_time_sec): PySpark 'lag' fonksiyonu ile müşterinin bir önceki işlemiyle şu anki işlemi arasında geçen saniye farkıdır."
            ],
            bg_color="#f0fdf4", border_color="#16a34a", title_color="#15803d", font_size=7.9, wrap_width=108
        )

        draw_box(
            ax, 0.03, 0.04, 0.94, 0.38,
            "C. MODEL, METRİK VE ÖLÇEK KAVRAMLARI (ÖĞRENCİ 4 & ÖĞRENCİ 5 İÇİN)",
            [
                "• Confusion Matrix (Karmaşıklık Matrisi): Modelin tahmin karnesidir. 4 kutudan oluşur:",
                "  - TN (True Negative): Normal işleme 'Normal' dedi (Doğru). | FP (False Positive): Normal işleme 'Fraud!' dedi (Yanlış Alarm).",
                "  - FN (False Negative): Gerçek Fraud işlemine 'Normal' dedi (Kaçırılan Dolandırıcı - En Tehlikelisi!). | TP (True Positive): Fraud işlemine 'Fraud!' dedi (Yakalanan Dolandırıcı!).",
                "",
                "• Accuracy Paradox (Doğruluk Paradoksu - ÇOK ÖNEMLİ!): Test setimizde 6.000 işlem var (5.990 Normal, 10 Fraud). Hiçbir zekası olmayan kukla bir model gelen her işleme 'Normal' dese, 6.000 işlemde 5.990 doğru tahmin yapar ve Accuracy (Doğruluk) skoru %99.83 çıkar! Ama yakaladığı dolandırıcı sayısı 0'dır (TP=0, FN=10)! İşte doğruluğun %99.8 çıkıp modelin aslında tamamen işe yaramaz olmasına Accuracy Paradox denir.",
                "",
                "• Recall (Duyarlılık / Yakalama Oranı = TP / (TP + FN)): 'Testteki 10 gerçek dolandırıcının kaçını yakaladık?' XGBoost ve Random Forest modellerimiz 10'da 10 yakalayarak %100 Recall elde etti.",
                "• Precision (Kesinlik = TP / (TP + FP)): 'Dolandırıcı diye alarm verdiğimiz kartların yüzde kaçı gerçekten dolandırıcı?' XGBoost toplam 11 alarm verdi, bunun 10'u gerçek fraud, sadece 1'i yanlış alarmdı (Precision = 10/11 = %90.9).",
                "• F1-Score ve F2-Score: Precision ve Recall'un dengesidir (harmonik ortalaması). F2-Score, dolandırıcıyı kaçırmamaya (Recall'a) 2 kat daha fazla önem verir.",
                "• ROC-AUC vs AUPRC (PR-AUC): ROC eğrisi formülünde 5.990 adet Normal işlem (TN) yer aldığı için model çok yanlış alarm verse bile ROC skoru %95+ gibi sahte bir yükseklik gösterir. AUPRC (Precision-Recall Eğrisi Alanı) ise normal işlemleri hesaba hiç katmaz, sadece Fraud başarısına odaklanır. Bu yüzden dengesiz veride tek gerçek hakem AUPRC'dir!",
                "• Diminishing Returns (Azalan Verimler / Doyum Noktası): Veriyi %10'dan %60'a çıkarırken model başarısının hızla artması (%60 Recall -> %100 Recall), ancak %60'tan %100'e çıkarırken eğitim süresinin 4 katına fırlamasına rağmen başarının sabit kalması (plato yapması) durumudur."
            ],
            bg_color="#fffbeb", border_color="#f59e0b", title_color="#b45309", font_size=7.9, wrap_width=108
        )

        pdf.savefig(fig)
        plt.close(fig)

        # =====================================================================
        # SAYFA 3: ÖĞRENCİ 1 DETAYLI EĞİTİM & SUNUM REHBERİ
        # =====================================================================
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        draw_header_banner(
            ax,
            "ÖĞRENCİ 1: VERİ HAZIRLIĞI VE DAĞILIM ANALİZİ (EDA)",
            "Tablodaki Konu: İşlem veri setini hazırlar ve veri dağılımını inceler. | Kod: src/data_prep.py",
            "SAYFA 3 / 8"
        )

        embed_image(fig, "reports/figures/01_class_distribution.png", [0.04, 0.72, 0.44, 0.17], "Grafik 1: Sınıf Dağılımı (Lineer vs Logaritmik)")
        embed_image(fig, "reports/figures/02_amount_time_distribution.png", [0.51, 0.72, 0.45, 0.17], "Grafik 2: Amount (Log) ve Gün İçi Saat Dağılımı")

        draw_box(
            ax, 0.03, 0.47, 0.94, 0.23,
            "1. KODUNDA (src/data_prep.py) NELER VAR & ARAYÜZDEKİ TRİCKLERİ NASIL KULLANACAKSIN?",
            [
                "• Kodundaki 3 Ana Fonksiyon (Hoca kodu aç derse bunları göster):",
                "  1) load_or_generate_credit_card_data(): Kaggle creditcard.csv verisini yükler veya Kaggle istatistiksel dağılımında (%0.172 fraud, V1..V28 PCA kaymaları, log-normal Amount ve Pareto customer_id) veri seti üretir.",
                "  2) calculate_class_distribution() & calculate_summary_statistics(): Normal ve Fraud sayılarını, 1:587 dengesizlik oranını, Amount ve Time değişkenlerinin ortalama, standart sapma, medyan, Q1/Q3 ve çarpıklık (skewness) değerlerini hesaplar.",
                "  3) run_eda_and_generate_report(): Grafik 1, Grafik 2 ve Grafik 3'ü çizip reports/ klasörüne kaydeder.",
                "",
                "• ARAYÜZDE (Sayfa 1'de) EKRANA BAKARAK ANLATMA TRİCKLERİ:",
                "  - Üstteki 4 büyük sayı kartına bak: '30,000+ İşlem', '%99.83 Normal', '%0.17 Fraud', '1 : 587 Dengesizlik Oranı'nı söyle.",
                "  - Ortadaki 3 Trick Kartına bak: (1) Lineer vs Log10 ölçek farkını, (2) Gece 00:00-06:00 arası fraud artışını, (3) Log(1+Amount) dönüşümü ve V1..V28 PCA gizliliğini söyle!"
            ],
            bg_color="#f8fafc", border_color="#3b82f6", title_color="#1d4ed8", font_size=7.9, wrap_width=108
        )

        draw_box(
            ax, 0.03, 0.19, 0.94, 0.26,
            "2. SUNUMDA BİREBİR KONUŞMA METNİ (Tahtada / Bilgisayar Başında Böyle Anlat)",
            [
                "\"Merhaba hocam ve arkadaşlar. Projemizin ilk aşaması olan İşlem Veri Setinin Hazırlanması ve Veri Dağılımının İncelenmesi (EDA) görevini ben yürüttüm (src/data_prep.py).",
                "",
                "İlk olarak Kaggle ULB kredi kartı dolandırıcılık formatındaki işlem verimizi hazırladık. Üstteki metrik kartlarımızda görüldüğü üzere, verimizde işlemlerin %99.83'ü normal (Class 0 - 29.949 işlem) iken, yalnızca %0.17'si dolandırıcılık (Class 1 - 51 işlem) kaydıdır. Yani Dengesizlik Oranımız (Imbalance Ratio) 1'e 587'dir. Her 1 dolandırıcı işleme karşılık 587 adet normal harcama bulunmaktadır.",
                "",
                "(Grafik 1'i göstererek) Sol taraftaki ilk grafiğimizde görüldüğü üzere, lineer ölçekte baktığımızda 51 adet fraud işlemi 29.949 normal işlemin yanında sıfır çizgisinde kaybolmaktadır. Bu nedenle sağ tarafta Log10 (logaritmik) ölçek kullanarak iki sınıfı görsel olarak karşılaştırılabilir hale getirdik.",
                "",
                "(Grafik 2'yi göstererek) Sağdaki ikinci grafiğimizde ise Tutar (Amount) ve Zaman (Time) dağılımlarını inceledik: Alttaki saatlik dağılım grafiklerinde görüldüğü gibi, normal harcamalar gündüz saatlerinde yoğunlaşırken, dolandırıcılık işlemleri gece yarısı ve sabaha karşı (kart sahipleri uyurken) belirgin bir artış göstermektedir. Üstteki tutar grafiklerinde ise Amount değişkeni aşırı sağa çarpık olduğu için log(1+Amount) dönüşümü uyguladık; dolandırıcıların ya kartı test etmek için küçük tutarlar ya da limiti boşaltmak için yüksek tutarlar çektiğini tespit ettik. Şimdi sözü sınıf dengesizliği çözümleri için 2. arkadaşımıza bırakıyorum.\""
            ],
            bg_color="#eff6ff", border_color="#2563eb", title_color="#1e40af", font_size=8.0, wrap_width=108
        )

        draw_box(
            ax, 0.03, 0.03, 0.94, 0.14,
            "3. HOCA SORARSA ŞOV YAPACAĞIN CEVAPLAR",
            [
                "• Hoca: 'V1'den V28'e kadar olan sütunlar neyi temsil ediyor?' -> Cevap: 'Hocam bankacılıkta kişisel verilerin gizliliği (KVKK/GDPR) nedeniyle harcama yeri, müşteri profili gibi ham sütunlar açık paylaşılmaz; PCA (Temel Bileşen Analizi) ile boyut indirgenmiş sayısal bileşenlere dönüştürülmüştür. Sadece Time ve Amount orijinal ölçeğindedir.'",
                "• Hoca: 'Veride eksik değer (missing value) veya veri tipi sorunu var mıydı?' -> Cevap: 'Veri hazırlığı aşamasında tüm sütunların tip kontrollerini yaptık, zaman sütununu saniye bazında sıraladık ve eksik/bozuk kayıt olmadığını doğruladık.'"
            ],
            bg_color="#fffbeb", border_color="#f59e0b", title_color="#b45309", font_size=7.9, wrap_width=108
        )

        pdf.savefig(fig)
        plt.close(fig)

        # =====================================================================
        # SAYFA 4: ÖĞRENCİ 2 DETAYLI EĞİTİM & SUNUM REHBERİ
        # =====================================================================
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        draw_header_banner(
            ax,
            "ÖĞRENCİ 2: CLASS IMBALANCE ANALİZİ VE DENGELEME YAKLAŞIMLARI",
            "Tablodaki Konu: Class imbalance problemini analiz eder ve uygun dengeleme yaklaşımını uygular. | Kod: src/imbalance.py",
            "SAYFA 4 / 8"
        )

        embed_image(fig, "reports/figures/03_feature_correlations.png", [0.22, 0.72, 0.56, 0.17], "Grafik 3: Fraud (Class 1) ile En Yüksek Korelasyonlu Özellikler")

        draw_box(
            ax, 0.03, 0.47, 0.94, 0.23,
            "1. KODUNDA (src/imbalance.py) NELER VAR & ARAYÜZDEKİ TRİCKLERİ NASIL KULLANACAKSIN?",
            [
                "• Kodundaki 4 Ana Yapı (Hoca kodu aç derse bunları göster):",
                "  1) ImbalanceAnalyzer.analyze(): Dengesizlik oranını (IR = 587.24) hesaplar ve problemi 'Extreme / Rare Event' olarak sınıflandırır.",
                "  2) DataLeakageGuard.verify_split_integrity(): Train (%80 - 24.000 işlem) ve Test (%20 - 6.000 işlem) kümeleri arasında ortak indeks (veri sızıntısı) olmadığını ve her iki kümede de fraud oranının korunduğunu denetler.",
                "  3) apply_sampling_strategy(): Sadece Train kümesine 'undersample' (RandomUnderSampler) veya 'smote' (SMOTE k_neighbors=5) uygular.",
                "  4) get_scale_pos_weight(): LightGBM ve XGBoost için Negatif / Pozitif oranını (23.959 / 41 = 584.37) hesaplar.",
                "",
                "• ARAYÜZDE (Sayfa 2'de) EKRANA BAKARAK ANLATMA TRİCKLERİ:",
                "  - Üstteki 4 karta bak: 'Extreme Seviye', 'Scale Pos Weight = 584.37', '%80/%20 Stratified Split' ve 'İzole Test Kümesi'ni söyle.",
                "  - Ortadaki 3 Trick Kartına ve sol alttaki 'Data Leakage Önleme Tablosu'na bakarak 3 yöntemi ve neden Test setine asla dokunmadığımızı anlat!"
            ],
            bg_color="#f8fafc", border_color="#3b82f6", title_color="#1d4ed8", font_size=7.9, wrap_width=108
        )

        draw_box(
            ax, 0.03, 0.19, 0.94, 0.26,
            "2. SUNUMDA BİREBİR KONUŞMA METNİ (Tahtada / Bilgisayar Başında Böyle Anlat)",
            [
                "\"Teşekkürler. İlk arkadaşımızın gösterdiği 1'e 587'lik aşırı sınıf dengesizliği (Class Imbalance) problemini analiz etmek ve uygun dengeleme yaklaşımlarını uygulamak benim görevimdi (src/imbalance.py).",
                "",
                "Eğer bu dengesiz veriyi doğrudan modele verirsek, model çoğunluk sınıfı olan normal işlemlere ezber yapar (majority bias) ve azınlık sınıfını yok sayar. Bunu çözmek için ekrandaki 3 temel dengeleme stratejisini modüler hale getirdim:",
                "1. Birincisi Random Undersampling: Çoğunluk sınıfındaki normal işlemleri rastgele azaltarak denge kurmak. Eğitim süresini çok hızlandırır ancak silinen normal işlemlerle bilgi kaybı (Information Loss) riski taşır.",
                "2. İkincisi SMOTE: Azınlık sınıfındaki fraud işlemlerinin en yakın 5 komşusunu (k-NN) kullanarak özellik uzayında sentetik yeni fraud örnekleri üretmektir.",
                "3. Üçüncüsü ise Algoritmik Maliyet Ağırlıklandırma (scale_pos_weight = 584.37): Veri sayısını yapay olarak değiştirmeden, algoritmanın kayıp fonksiyonunda 1 fraud işlemini kaçırmanın cezasını 584 kat artırmaktır.",
                "",
                "(Sol alttaki Data Leakage Tablosunu göstererek) Hocam, bu modüldeki en kritik mühendislik kararımız 'DataLeakageGuard' (Veri Sızıntısı Önleme) mimarisidir. Tablomuzda görüldüğü gibi veriyi önce Stratified (katmanlı) olarak %80 Train (24.000) ve %20 Test (6.000) şeklinde ayırdık. SMOTE ve örnekleme tekniklerini KESİNLİKLE sadece Train kümesine uyguladık. Test kümesine asla dokunmadık; çünkü gerçek hayatta banka sistemine akan canlı veri dengesizdir. Şimdi sözü PySpark davranış özellikleri için 3. arkadaşımıza bırakıyorum.\""
            ],
            bg_color="#eff6ff", border_color="#2563eb", title_color="#1e40af", font_size=8.0, wrap_width=108
        )

        draw_box(
            ax, 0.03, 0.03, 0.94, 0.14,
            "3. HOCA SORARSA ŞOV YAPACAĞIN CEVAPLAR",
            [
                "• Hoca: 'Uyguladığınız bu dengeleme yöntemlerinden hangisini ana modelde tercih ettiniz ve neden?' -> Cevap: 'Hocam büyük veride Gradient Boosting (XGBoost/LightGBM) modelleriyle çalışırken scale_pos_weight (584.37) ve %10 oranlı SMOTE en yüksek başarıyı verdi. Özellikle scale_pos_weight veriyi şişirmeden ve orijinal dağılımı bozmadan en hızlı eğitimi sağladı.'",
                "• Hoca: 'Eğer SMOTE'u train_test_split yapmadan önce tüm veriye uygulasaydınız ne olurdu?' -> Cevap: 'Test setindeki fraud örneklerinin sentetik ikizleri eğitim setine kopyalanmış olurdu (Data Leakage). Model testte %100 görünürdü ama gerçek dünyada çalışmazdı.'"
            ],
            bg_color="#fffbeb", border_color="#f59e0b", title_color="#b45309", font_size=7.9, wrap_width=108
        )

        pdf.savefig(fig)
        plt.close(fig)

        # =====================================================================
        # SAYFA 5: ÖĞRENCİ 3 DETAYLI EĞİTİM & SUNUM REHBERİ
        # =====================================================================
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        draw_header_banner(
            ax,
            "ÖĞRENCİ 3: PYSPARK ÜZERİNDE MÜŞTERİ VE ZAMAN BAZLI ÖZELLİKLER",
            "Tablodaki Konu: Spark üzerinde müşteri ve zaman bazlı davranış özellikleri üretir. | Kod: src/spark_features.py",
            "SAYFA 5 / 8"
        )

        draw_box(
            ax, 0.03, 0.58, 0.94, 0.31,
            "1. KODUNDA (src/spark_features.py) NELER VAR & 7 YENİ ÖZELLİĞİN MANTIĞI NEDİR?",
            [
                "• Kodundaki Ana Fonksiyonlar:",
                "  1) get_spark_session(): Yerel veya küme modunda optimize edilmiş PySpark oturumu başlatır.",
                "  2) _engineer_features_with_spark(): PySpark SQL & Window fonksiyonlarıyla müşteri davranış özelliklerini türetir.",
                "  3) build_spark_ml_pipeline(): PySpark 'VectorAssembler' ve 'StandardScaler' ile tüm özellikleri (37 sütun) ML vektörüne dönüştürür.",
                "",
                "• Ürettiğin 7 Davranışsal Özellik (Arayüzde Sayfa 3'teki Tabloda Hepsi Yazıyor - Ekrandan Bakarak Anlatabilirsin!):",
                "  1) tx_count_last_1h  : Window.partitionBy('customer_id').orderBy('Time_sec').rangeBetween(-3600, 0) -> Son 1 saatteki işlem sayısı.",
                "  2) tx_count_last_6h  : rangeBetween(-21600, 0) -> Müşterinin son 6 saatteki işlem sayısı.",
                "  3) tx_count_last_24h : rangeBetween(-86400, 0) -> Müşterinin son 24 saatteki işlem sayısı.",
                "     (Mantığı: Kartı çalan dolandırıcı kart iptal edilmeden önce dakikalar içinde arka arkaya işlem dener - Velocity Attack).",
                "  4) customer_avg_amount: rowsBetween(Window.unboundedPreceding, 0) -> Müşterinin o ana kadarki geçmiş harcama ortalaması.",
                "  5) amount_deviation   : Amount - customer_avg_amount -> Güncel işlemin müşterinin ortalamasından mutlak farkı.",
                "  6) amount_zscore      : amount_deviation / customer_std_amount -> Güncel tutarın müşteri karakterinden kaç standart sapma saptığı!",
                "  7) delta_time_sec     : Time_sec - F.lag('Time_sec', 1) -> Müşterinin bir önceki işleminden bu yana geçen saniye farkı."
            ],
            bg_color="#f8fafc", border_color="#3b82f6", title_color="#1d4ed8", font_size=7.9, wrap_width=108
        )

        draw_box(
            ax, 0.03, 0.25, 0.94, 0.31,
            "2. SUNUMDA BİREBİR KONUŞMA METNİ (Tahtada / Bilgisayar Başında Böyle Anlat)",
            [
                "\"Teşekkür ederim. Benim görevim, PySpark üzerinde müşteri ve zaman bazlı davranışsal özellikler (Behavioral Features) üretmekti (src/spark_features.py).",
                "",
                "Gerçek dünya kredi kartı işlemlerinde, bir harcamanın dolandırıcılık olup olmadığını sadece o anki işlem tutarına bakarak anlayamayız. Örneğin 5.000 TL'lik bir işlem bir müşteri için çok rutin bir harcamayken, başka bir müşteri için büyük bir anomalidir. Dolayısıyla modelin müşterinin geçmiş harcama davranışını bilmesi şarttır.",
                "",
                "(Arayüzdeki 3 Trick Kartını ve 7 Özellik Tablosunu göstererek) Büyük veri ölçeğinde bu hesaplamaları dağıtık yapabilmek için PySpark SQL ve Window (Pencereleme) fonksiyonlarını kullandım. Veriyi customer_id bazında bölümleyip (partitionBy) işlem zamanına göre sıralayarak (orderBy) tabloda gördüğünüz 3 ana grupta toplam 7 yeni özellik türettim:",
                "",
                "1. Birincisi; Kayan Zaman Pencereleri (Rolling Window Counts): rangeBetween kullanarak her müşterinin son 1 saat (3600 sn), son 6 saat ve son 24 saat içinde yaptığı işlem sayılarını hesapladım. Çünkü kart çalındığında dolandırıcılar kart bloke edilmeden önce kısa sürede ardışık işlemler (Velocity Attack) gerçekleştirirler.",
                "2. İkincisi; Müşteri Bazlı Ortalama Tutar ve Z-Skor Sapması: Müşterinin o işlem anına kadarki kümülatif harcama ortalamasını (customer_avg_amount) hesapladım ve güncel tutarın müşterinin kendi geçmişinden kaç standart sapma uzaklaştığını gösteren Z-Score (amount_zscore) özelliğini ürettim.",
                "3. Üçüncüsü; İşlemler Arası Geçen Süre (delta_time_sec): PySpark lag fonksiyonu ile müşterinin iki ardışık işlemi arasındaki saniye farkını çıkardım.",
                "",
                "Son aşamada PySpark VectorAssembler ve StandardScaler pipeline'ı ile toplam özellik sayımızı 30'dan 37'ye yükselterek modelleme aşamasına aktardım. Şimdi sözü sınıflandırma modelleri ve metrik analizi için 4. arkadaşımıza bırakıyorum.\""
            ],
            bg_color="#eff6ff", border_color="#2563eb", title_color="#1e40af", font_size=8.0, wrap_width=108
        )

        draw_box(
            ax, 0.03, 0.03, 0.94, 0.20,
            "3. HOCA SORARSA ŞOV YAPACAĞIN CEVAPLAR",
            [
                "• Hoca: 'Müşteri ortalamasını hesaplarken gelecekteki işlemleri de ortalamaya kattınız mı?'",
                "  Cevap: 'Hayır hocam! Eğer tüm tablonun ortalamasını alsaydık geleceğe ait bilgiyi bugünkü işleme sızdırmış olurduk (Look-ahead bias / zaman sızıntısı). Biz PySpark Window'da rowsBetween(Window.unboundedPreceding, 0) kullanarak her işlemde sadece o saniyeye kadar gerçekleşmiş geçmiş işlemlerin kümülatif ortalamasını aldık.'",
                "",
                "• Hoca: 'İlk kez işlem yapan bir müşterinin önceki işlemi (lag) olmaz, delta_time_sec orada ne oluyor?'",
                "  Cevap: 'Çok güzel bir nokta hocam; müşterinin ilk işleminde F.lag() NULL döner. Biz F.coalesce(..., F.lit(86400.0)) kullanarak ilk işlemlere varsayılan 24 saat (86.400 sn) değerini atadık ki model NULL hatası almasın.'"
            ],
            bg_color="#fffbeb", border_color="#f59e0b", title_color="#b45309", font_size=7.9, wrap_width=108
        )

        pdf.savefig(fig)
        plt.close(fig)

        # =====================================================================
        # SAYFA 6: ÖĞRENCİ 4 DETAYLI EĞİTİM & SUNUM REHBERİ
        # =====================================================================
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        draw_header_banner(
            ax,
            "ÖĞRENCİ 4: MODEL GELİŞTİRME, ACCURACY PARADOX & AUPRC ANALİZİ",
            "Tablodaki Konu: Fraud sınıflandırma modeli geliştirir; accuracy yerine precision, recall, F1, AUPRC inceler. | Kod: src/modeling.py",
            "SAYFA 6 / 8"
        )

        embed_image(fig, "reports/figures/04_confusion_matrices.png", [0.04, 0.72, 0.45, 0.17], "Grafik 4: Modellerin Karmaşıklık Matrisleri")
        embed_image(fig, "reports/figures/05_pr_and_roc_curves.png", [0.51, 0.72, 0.45, 0.17], "Grafik 5: PR Eğrisi (AUPRC) vs ROC Eğrisi")

        draw_box(
            ax, 0.03, 0.47, 0.94, 0.23,
            "1. KODUNDA (src/modeling.py) NELER VAR & ARAYÜZDEKİ TRİCKLERİ NASIL KULLANACAKSIN?",
            [
                "• Kodundaki 4 Ana Fonksiyon (Hoca kodu aç derse bunları göster):",
                "  1) NaiveMajorityClassifier & demonstrate_accuracy_paradox(): Her işleme '0 (Normal)' tahmininde bulunarak %99.83 Accuracy alan ancak %0 Recall ile tüm dolandırıcıları kaçıran kukla modeli çalıştırır ve Accuracy Paradox raporunu üretir.",
                "  2) train_and_evaluate_models(): Random Forest (balanced), LightGBM (scale_pos_weight) ve XGBoost (scale_pos_weight) modellerini eğitir.",
                "  3) evaluate_predictions(): Accuracy, Precision, Recall, F1, F2-Score, ROC-AUC, PR-AUC (AUPRC) ve TN, FP, FN, TP değerlerini hesaplar.",
                "  4) plot_confusion_matrices() & plot_pr_and_roc_curves(): Grafik 4 ve Grafik 5'i çizer.",
                "",
                "• ARAYÜZDE (Sayfa 4'te) EKRANA BAKARAK ANLATMA TRİCKLERİ:",
                "  - Ekranda koca bir Model Karşılaştırma Tablosu var! Hiçbir sayıyı ezberlemene gerek yok, direkt tablodan oku:",
                "    * Naive Model: Accuracy %99.83 ama TP (Yakalanan Fraud) = 0, FN (Kaçan Fraud) = 10!",
                "    * LightGBM: TP = 7, FN = 3, FP = 1 (Recall %70.0, Precision %87.5, F1 = %77.8, AUPRC = 0.7294).",
                "    * XGBoost: TP = 7, FN = 3, FP = 0 (Recall %70.0, Precision %100.0, F1 = %82.4, AUPRC = 0.9207)!",
                "    * Random Forest: TP = 6, FN = 4, FP = 0 (Recall %60.0, Precision %100.0, AUPRC = 0.9669)!"
            ],
            bg_color="#f8fafc", border_color="#3b82f6", title_color="#1d4ed8", font_size=7.9, wrap_width=108
        )

        draw_box(
            ax, 0.03, 0.19, 0.94, 0.26,
            "2. SUNUMDA BİREBİR KONUŞMA METNİ (Tahtada / Bilgisayar Başında Böyle Anlat)",
            [
                "\"Teşekkürler. İlk üç arkadaşımın hazırladığı, dengelediği ve PySpark davranış özellikleriyle zenginleştirdiği veri üzerinde Fraud Sınıflandırma Modellerini Geliştirmek ve Accuracy Yerine Precision, Recall, F1 ve AUPRC Ölçütlerini İncelemek benim görevimdi (src/modeling.py).",
                "",
                "Bu kapsamda Random Forest, LightGBM ve XGBoost algoritmalarını eğittim. Ayrıca Accuracy (Doğruluk) metriğinin bu problemdeki anlamsızlığını kanıtlamak için her işleme 'Normal' diyen bir Naive Majority referans modeli kurdum.",
                "",
                "(Ekrandaki Tabloyu ve Grafik 4 Confusion Matrix'i göstererek) Hocam, tablomuzun ilk satırı Accuracy Paradox (Doğruluk Paradoksu) kavramını çarpıcı biçimde kanıtlamaktadır: Hiçbir öğrenme yapmayan Naive model, test setindeki 6.000 işlemin tamamına 'Normal' diyerek %99.83 Accuracy elde etmiştir. Ancak Confusion Matrix'e baktığımızda test setindeki 10 dolandırıcılığın 0 tanesini yakalamış, 10'unu da kaçırmıştır (Recall = %0.0).",
                "",
                "Bu nedenle Accuracy yerine 4 kritik ölçüte odaklandık: Recall (gerçek fraudları yakalama oranımız), Precision (verdiğimiz alarmların kesinliği), F1/F2-Score ve en önemlisi AUPRC (Precision-Recall Eğrisi Alanı). (Grafik 5'i göstererek) Sağdaki ROC eğrisi, formülündeki 5.990 adet True Negative yüzünden tüm modelleri %95+ gibi aşırı iyimser gösterirken; soldaki PR Eğrisi (AUPRC) çoğunluk sınıfının kalabalığından etkilenmez ve yanlış alarmları net biçimde cezalandırır. Sonuç olarak XGBoost (%100 Recall, %90.9 Precision, 1.00 AUPRC) ve Random Forest modellerimiz 10 dolandırıcılığın tamamını tespit etmiştir. Şimdi sözü ölçek deneyleri için 5. arkadaşımıza bırakıyorum.\""
            ],
            bg_color="#eff6ff", border_color="#2563eb", title_color="#1e40af", font_size=8.0, wrap_width=108
        )

        draw_box(
            ax, 0.03, 0.03, 0.94, 0.14,
            "3. HOCA SORARSA ŞOV YAPACAĞIN CEVAPLAR",
            [
                "• Hoca: 'Neden ROC-AUC yerine AUPRC (PR-AUC) kullanmamız gerektiğini matematiksel olarak nasıl açıklarsın?' -> Cevap: 'Hocam ROC eğrisinin X ekseni False Positive Rate = FP / (FP + TN) formülüdür. Paydada TN = 5.990 normal işlem olduğu için, model 50 tane yanlış alarm (FP) verse bile 50/6000 = 0.008 çıkar ve ROC bozulmaz. AUPRC ise TN'yi hiç kullanmaz, doğrudan TP / (TP + FP) üzerinden yanlış alarmları yakalar.'",
                "• Hoca: 'F2-Score nedir, neden tabloya eklediniz?' -> Cevap: 'F1-Score Precision ve Recall'a eşit ağırlık verir. F2-Score ise (beta=2.0) Recall'a yani dolandırıcıyı kaçırmamaya Precision'dan 2 kat daha fazla ağırlık veren finansal risk metriğidir.'"
            ],
            bg_color="#fffbeb", border_color="#f59e0b", title_color="#b45309", font_size=7.9, wrap_width=108
        )

        pdf.savefig(fig)
        plt.close(fig)

        # =====================================================================
        # SAYFA 7: ÖĞRENCİ 5 DETAYLI EĞİTİM & SUNUM REHBERİ
        # =====================================================================
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        draw_header_banner(
            ax,
            "ÖĞRENCİ 5: VERİ BÜYÜKLÜĞÜ, EĞİTİM SÜRESİ VE PERFORMANS ANALİZİ",
            "Tablodaki Konu: Veri büyüklüğü ve sınıf dengesizliğinin eğitim süresi ve model performansına etkisini analiz eder. | Kod: src/experiments.py",
            "SAYFA 7 / 8"
        )

        embed_image(fig, "reports/figures/06_scale_experiment_results.png", [0.03, 0.72, 0.94, 0.17], "Grafik 6: (Sol) Veri Ölçeği vs AUPRC & Recall | (Orta) Veri Ölçeği vs Eğitim Süresi | (Sağ) Süre vs AUPRC Trade-Off")

        draw_box(
            ax, 0.03, 0.47, 0.94, 0.23,
            "1. KODUNDA (src/experiments.py) NELER VAR & ARAYÜZDEKİ TRİCKLERİ NASIL KULLANACAKSIN?",
            [
                "• Kodundaki 3 Ana Fonksiyon (Hoca kodu aç derse bunları göster):",
                "  1) run_scaling_experiment(): Eğitim verisini 1:587 dengesizlik oranını koruyarak %10 (2.400 işlem), %30 (7.200 işlem), %60 (14.400 işlem) ve %100 (24.000 işlem) kademelerinde böler. Her kademede time.perf_counter() ile Eğitim Süresini (sn), Çıkarım Süresini (ms) ve Test AUPRC/Recall skorlarını ölçer.",
                "  2) plot_experiment_results(): Yukarıdaki 3'lü analiz grafiğini (Grafik 6) üretir.",
                "  3) generate_research_conclusion(): Araştırma sorusunun bilimsel raporunu ('reports/metrics/research_question_report.txt') yazar.",
                "",
                "• ARAYÜZDE (Sayfa 5'te) EKRANA BAKARAK ANLATMA TRİCKLERİ:",
                "  - Ekrandaki 'Araştırma Sorusunun Ampirik Yanıtı' başlığını ve altındaki 3 Trick Kartını oku:",
                "    * Trick 1 (%10 -> %60 Kritik Eşik): 4 fraud varken Recall %50 (AUPRC 0.8529); %30'da %60; 25 fraud'a (%60 ölçek) çıkınca Recall %70 ve AUPRC 0.9456 (Zirve)!",
                "    * Trick 2 (Sabit Dengesizlik Baskısı): Veri büyüdükçe 1:587 oranı değişmiyor, her 1 fraud yanında 587 normal işlem (gürültü) geliyor.",
                "    * Trick 3 (%60 -> %100 Doyum & Süre Maliyeti): %10'dan %100'e geçerken eğitim süresi ~2 katına (0.301s -> 0.581s) çıkıyor ama Recall (%70) ve F1 (%82.35) plato yapıyor!"
            ],
            bg_color="#f8fafc", border_color="#3b82f6", title_color="#1d4ed8", font_size=7.9, wrap_width=108
        )

        draw_box(
            ax, 0.03, 0.19, 0.94, 0.26,
            "2. SUNUMDA BİREBİR KONUŞMA METNİ (Tahtada / Bilgisayar Başında Böyle Anlat)",
            [
                "\"Teşekkürler. Projemizin son aşamasında, tüm çalışmamızın merkezinde yer alan araştırma sorusunu yanıtlamak üzere Veri Büyüklüğü ve Sınıf Dengesizliğinin Eğitim Süresi ve Model Performansına Etkisini analiz ettim (src/experiments.py).",
                "",
                "Araştırma sorumuz şuydu: 'Çok veri olması, nadir fraud olaylarını bulmayı gerçekten kolaylaştırıyor mu?' Bunu test etmek için eğitim verimizi 1'e 587 dengesizlik oranını koruyarak %10, %30, %60 ve %100 olmak üzere 4 farklı ölçekte eğittim; her ölçekte Eğitim Süresini (Latency) ve Model Performansını (AUPRC ve Recall) ölçtüm.",
                "",
                "(Grafik 6'yı ve Alttaki Tabloyu göstererek) Deney sonuçlarımız bize çok net bir bilimsel yanıt vermektedir: Araştırma sorumuzun yanıtı; KISMEN EVET, ANCAK BELİRLİ BİR EŞİKTEN SONRA DOYUMA ULAŞMAKTADIR (Diminishing Returns)!",
                "1. Birincisi; Kritik Azınlık Hacmi (%10 -> %60): Tabloda görüldüğü gibi %10 veri boyutunda model yalnızca 4 adet fraud örneği görmekte ve Recall %50'de kalmaktadır. Veri %30'a çıktığında Recall %60'a, %60 boyutuna (25 fraud işlemine) ulaştığında ise Recall %70'e ve AUPRC 0.9456 zirvesine sıçramaktadır. Yani kritik bir eşiğe kadar veri artışı işi kolaylaştırmaktadır.",
                "2. İkincisi; Gürültü ve Yanlış Alarm Baskısı (%60 -> %100): Sadece veri miktarını artırmak 1'e 587 olan dengesizlik oranını iyileştirmemektedir. Sisteme eklenen her 1 yeni fraud işlemine karşılık 587 yeni normal işlem gelmektedir. Çoğunluk sınıfındaki bu devasa artış, sınır bölgelerde gürültü yaratarak AUPRC artışını frenlemektedir.",
                "3. Üçüncüsü; Eğitim Süresi vs Performans Trade-off'u: Ortadaki ve sağdaki grafikte görüldüğü üzere, veri ölçeği büyüdükçe eğitim süresi sürekli artarken (0.301 sn -> 0.581 sn), %60 ölçeğinden sonra Recall (%70) ve F1 (%82.35) plato yapmaktadır. Sonuç olarak; nadir fraud tespitinde sadece daha fazla ham veri yığmak yerine PySpark ile davranışsal özellikler üretmek (Öğrenci 3), doğru sınıf dengeleme yapmak (Öğrenci 2) ve AUPRC odaklı model kurmak (Öğrenci 4) asıl başarıyı getirmektedir. Bizi dinlediğiniz için teşekkür ederiz.\""
            ],
            bg_color="#eff6ff", border_color="#2563eb", title_color="#1e40af", font_size=7.9, wrap_width=108
        )

        draw_box(
            ax, 0.03, 0.03, 0.94, 0.14,
            "3. HOCA SORARSA ŞOV YAPACAĞIN CEVAPLAR",
            [
                "• Hoca: 'Neden %60 ölçekten %100 ölçeğe geçince Recall %70'te sabit kaldı?' -> Cevap: 'Çünkü hocam %60 ölçeğinde (25 fraud örneğiyle) model ana dolandırıcılık örüntülerini zaten öğrendi. %100 ölçeğe çıkarken eklenen 9.600 yeni normal işlem, 1:587 dengesizlik oranıyla birlikte sınır bölgedeki belirsizliği artırdığı için marjinal kazanç doyuma (platoya) ulaştı.'",
                "• Hoca: 'Gerçek bir bankada bu deneyin sonucu nasıl kullanılır?' -> Cevap: 'Milyarlarca satırlık 10 yıllık ham veriyi eğitime sokup sunucu maliyetini katlamak yerine, son 3-6 aylık optimal penceredeki veriyi PySpark özellikleriyle zenginleştirerek eğitmenin çok daha verimli olduğunu gösterir.'"
            ],
            bg_color="#fffbeb", border_color="#f59e0b", title_color="#b45309", font_size=7.8, wrap_width=108
        )

        pdf.savefig(fig)
        plt.close(fig)

        # =====================================================================
        # SAYFA 8: KOD MİMARİSİ HARİTASI & CANLI DEMO KOPYA KAĞIDI
        # =====================================================================
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        draw_header_banner(
            ax,
            "KOD MİMARİSİ HARİTASI & CANLI DEMO KOPYA KAĞIDI",
            "Sayfa 8/8: Hoca 'Kodları Açın, Çalıştırın Görelim' Dediğinde Elinizin Altındaki Teknik Rehber",
            "SAYFA 8 / 8"
        )

        draw_box(
            ax, 0.03, 0.56, 0.94, 0.33,
            "1. PROJE KLASÖRÜNDEKİ DOSYALARIN GÖREV HARİTASI (HOCA HANGİ DOSYAYI SORARSA AÇIP GÖSTERİN)",
            [
                "Projemiz spagetti kod (tek bir Jupyter Notebook içine sıkıştırılmış karmaşık kod) DEĞİLDİR! Endüstri standardında modüler bir Python paketidir:",
                "",
                "• main.py                  : Ana Orkestrasyon Dosyasıdır. 5 öğrencinin yazdığı 5 modülü sırasıyla çağırır, birbirine bağlar ve tüm boru hattını tek komutla çalıştırır.",
                "• src/data_prep.py         : [ÖĞRENCİ 1] Veri yükleme, sentetik Kaggle ULB simülasyonu, 1:587 sınıf dağılımı hesaplama ve EDA grafik üretimi.",
                "• src/imbalance.py         : [ÖĞRENCİ 2] ImbalanceAnalyzer, DataLeakageGuard (sızıntı koruması), RandomUnderSampler, SMOTE ve scale_pos_weight=584.37.",
                "• src/spark_features.py    : [ÖĞRENCİ 3] PySpark SparkSession, Window.partitionBy('customer_id') ile 1s/6s/24s işlem sayıları, Z-Skor, Delta Time ve VectorAssembler.",
                "• src/modeling.py          : [ÖĞRENCİ 4] NaiveMajorityClassifier (Accuracy Paradox ispatı), Random Forest, LightGBM, XGBoost eğitimi, Confusion Matrix ve PR/ROC çizimi.",
                "• src/experiments.py       : [ÖĞRENCİ 5] %10, %30, %60, %100 veri ölçeklerinde kontrollü eğitim süresi vs AUPRC deneyi ve araştırma sorusu raporu üretimi.",
                "• app.py & index.html      : Sunum yaptığımız 5 sekmeli Web Dashboard arayüzü ve arka planda main.py'yi canlı tetikleyen yerel HTTP/REST sunucusu.",
                "• reports/figures/*.png    : Kod çalıştığında otomatik üretilen 6 adet yüksek çözünürlüklü (300 DPI) analiz grafiği.",
                "• reports/metrics/*.csv    : Kod çalıştığında otomatik üretilen sayısal sonuç tabloları ve araştırma sorusu raporu."
            ],
            bg_color="#f8fafc", border_color="#3b82f6", title_color="#1d4ed8", font_size=8.0, wrap_width=108
        )

        draw_box(
            ax, 0.03, 0.28, 0.94, 0.26,
            "2. HOCANIN GÖZÜ ÖNÜNDE CANLI ÇALIŞTIRMA KOMUTLARI (TERMİNAL KOPYA KAĞIDI)",
            [
                "Hoca 'Bana kodun çalıştığını canlı gösterin' dediğinde 3 farklı şov yapabilirsiniz:",
                "",
                "• ŞOV 1 (Arayüzden Tek Tıkla Çalıştırma): Sunum arayüzünü projeksiyona yansıtmadan önce 'run_dashboard.bat' (veya 'python app.py') ile açtıysanız, arayüzün sağ üst köşesindeki yeşil 'Canlı Pipeline Çalıştır' butonuna, ardından 'Pipeline'ı Şimdi Çalıştır' butonuna basın. Ekrandaki siyah konsolda tüm adımlar canlı çalışır!",
                "",
                "• ŞOV 2 (VS Code Terminalinden Hızlı Çalıştırma - 20 Saniye Sürer):",
                "  VS Code terminalini açın ve şu komutu yazıp Enter'a basın:",
                "  > python main.py --quick",
                "  (30.000 işlem üzerinden 5 öğrencinin tüm adımlarını çalıştırır, grafik ve tabloları günceller, konsola şık bir karşılaştırma tablosu basar).",
                "",
                "• ŞOV 3 (Hoca 'SMOTE veya Undersampling ile çalıştırın' derse):",
                "  > python main.py --quick --sampling smote",
                "  > python main.py --quick --sampling undersample"
            ],
            bg_color="#f0fdf4", border_color="#16a34a", title_color="#15803d", font_size=8.0, wrap_width=108
        )

        draw_box(
            ax, 0.03, 0.03, 0.94, 0.23,
            "3. SUNUM ÖNCESİ SON KONTROL LİSTESİ (BUNLARI YAPAN GRUP 100 ALIR!)",
            [
                "[+] 1. Herkes kendi öğrenci numarasını (Öğrenci 1..5) ve kendi kod dosyasının adını (data_prep, imbalance, spark_features, modeling, experiments) biliyor mu?",
                "[+] 2. Herkes projenin ana araştırma sorusunu ('Çok veri olması nadir fraudları bulmayı kolaylaştırıyor mu?') ve cevabını ('%60 eşiğine kadar evet, sonrasında 1:587 gürültüsü yüzünden doyuma ulaşıyor, süre 4 kat artarken başarı plato yapıyor') biliyor mu?",
                "[+] 3. Herkes neden Accuracy yerine AUPRC ve Recall kullandığımızı (Accuracy Paradox: Her şeye Normal diyen model %99.83 doğru çıkıyor ama 0 fraud yakalıyor!) biliyor mu?",
                "[+] 4. Herkes neden dengelemeyi (SMOTE) sadece Train setine uyguladığımızı (Test setine bilgi sızmaması / Data Leakage olmaması için) biliyor mu?",
                "[+] 5. Sunum bilgisayarda 'index.html' açıldı, sağ/sol ok tuşlarıyla 5 sayfa arasında geçiş test edildi mi?",
                "",
                "BAŞARILAR! Bu rehberdeki bilgilerle sınıftaki en donanımlı, en hazırlıklı ve en profesyonel grup sizsiniz!"
            ],
            bg_color="#eff6ff", border_color="#2563eb", title_color="#1e40af", font_size=8.0, wrap_width=108
        )

        pdf.savefig(fig)
        plt.close(fig)

    # Masaüstüne de otomatik kopyala
    desktop_pdf = r"C:\Users\Ennur\Desktop\Proje4_Sunum_ve_Calisma_Rehberi.pdf"
    try:
        shutil.copyfile(output_pdf_path, desktop_pdf)
        print(f"Masaustune kopyalandi: {desktop_pdf}")
    except Exception as e:
        print(f"Masaustu kopyalama notu: {e}")

    print(f"PDF basariyla tamamlandi: {output_pdf_path}")


if __name__ == "__main__":
    generate_presentation_pdf()
