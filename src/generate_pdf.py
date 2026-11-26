"""
src/generate_pdf.py
=============================================================================
Proje 4: Kredi Karti Dolandiricilik Analizi
5 Öğrenci İçin Kapsamlı Eğitim, Teknik Rapor ve Birebir Sunum Konuşma Rehberi
Çok Sayfalı Yüksek Çözünürlüklü A4 Vektör PDF Üreticisi (Matplotlib PdfPages)
=============================================================================
"""

import os
import textwrap
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg

# Matplotlib cache ve Türkçe karakter uyumlu font
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".mpl_cache")
plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def draw_header_banner(ax, title: str, subtitle: str, badge_text: str = ""):
    """Sayfa üstü koyu lacivert kurumsal başlık kutusu çizer."""
    rect = patches.FancyBboxPatch(
        (0.03, 0.90), 0.94, 0.075,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        facecolor="#0f172a", edgecolor="#2563eb", linewidth=1.5,
        transform=ax.transAxes
    )
    ax.add_patch(rect)
    ax.text(0.05, 0.947, title, fontsize=13, fontweight="bold", color="#ffffff", transform=ax.transAxes, va="center")
    ax.text(0.05, 0.916, subtitle, fontsize=8.8, color="#93c5fd", transform=ax.transAxes, va="center")
    if badge_text:
        badge = patches.FancyBboxPatch(
            (0.77, 0.918), 0.18, 0.038,
            boxstyle="round,pad=0.005,rounding_size=0.01",
            facecolor="#1e3a8a", edgecolor="#60a5fa", linewidth=1,
            transform=ax.transAxes
        )
        ax.add_patch(badge)
        ax.text(0.86, 0.937, badge_text, fontsize=8.2, fontweight="bold", color="#ffffff", transform=ax.transAxes, ha="center", va="center")


def draw_box(ax, x, y, w, h, title, lines, bg_color="#f8fafc", border_color="#cbd5e1", title_color="#1e3a8a", font_size=8.4, wrap_width=108):
    """Başlıklı ve otomatik satır kaydırmalı bilgi/konuşma kutusu çizer."""
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor=bg_color, edgecolor=border_color, linewidth=1.3,
        transform=ax.transAxes
    )
    ax.add_patch(rect)

    # Başlık
    ax.text(x + 0.015, y + h - 0.018, title, fontsize=9.8, fontweight="bold", color=title_color, transform=ax.transAxes, va="top")

    # İçerik metni
    wrapped_paragraphs = []
    for line in lines:
        if line == "":
            wrapped_paragraphs.append("")
        else:
            wrapped = textwrap.fill(line, width=wrap_width)
            wrapped_paragraphs.append(wrapped)

    full_text = "\n".join(wrapped_paragraphs)
    ax.text(
        x + 0.015, y + h - 0.042, full_text,
        fontsize=font_size, color="#0f172a", transform=ax.transAxes,
        va="top", ha="left", linespacing=1.32
    )


def embed_image(fig, img_path, rect_coords, caption=""):
    """Sayfaya PNG görseli ve alt yazısını gömer."""
    if os.path.exists(img_path):
        ax_img = fig.add_axes(rect_coords)
        img = mpimg.imread(img_path)
        ax_img.imshow(img)
        ax_img.axis("off")
        if caption:
            ax_img.set_title(caption, fontsize=8, fontweight="bold", color="#334155", pad=3)


def generate_presentation_pdf(output_pdf_path: str = "Proje4_Kredi_Karti_Dolandiricilik_Sunum_ve_Calisma_Rehberi.pdf"):
    print(f"PDF olusturuluyor: {output_pdf_path} ...")

    with PdfPages(output_pdf_path) as pdf:

        # =====================================================================
        # SAYFA 1: KAPAK, GÖREV TABLOSU VE BÜYÜK RESİM (ANA HİKAYE)
        # =====================================================================
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 Boyutu
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        draw_header_banner(
            ax,
            "PROJE 4: KREDİ KARTI DOLANDIRICILIK ANALİZİ",
            "Büyük Veri Mühendisliği & Makine Öğrenimi | 5 Öğrenci Eğitim ve Birebir Sunum Konuşma Rehberi",
            "5 ÖĞRENCİ REHBERİ"
        )

        # Problem ve Araştırma Sorusu Kutusu
        draw_box(
            ax, 0.03, 0.77, 0.94, 0.11,
            "PROBLEM TANIMI VE ARAŞTIRMA SORUSU",
            [
                "• Problem: Çok büyük işlem verisinde az sayıdaki fraud (dolandırıcılık) işleminin tespiti.",
                "• Araştırma Sorusu: 'Çok veri olması, nadir fraud olaylarını bulmayı gerçekten kolaylaştırıyor mu?'",
                "• Veri Seti & Ölçek: Kaggle Kredi Kartı Dolandırıcılık Formatı (Time, V1..V28 PCA, Amount, Class, customer_id) | 1:587 Dengesizlik Oranı."
            ],
            bg_color="#eff6ff", border_color="#2563eb", title_color="#1d4ed8", font_size=8.8, wrap_width=105
        )

        # 5 Öğrenci Görev Dağılım Tablosu Kutusu
        draw_box(
            ax, 0.03, 0.51, 0.94, 0.24,
            "5 ÖĞRENCİNİN KONU İÇERİĞİ VE KOD MODÜLLERİ (DERS TABLOSU İLE BİREBİR UYUMLU)",
            [
                "1. ÖĞRENCİ 1 (src/data_prep.py)     : İşlem veri setini hazırlar ve veri dağılımını inceler. (Sınıf oranı 1:587, Time ve Amount EDA)",
                "",
                "2. ÖĞRENCİ 2 (src/imbalance.py)     : Class imbalance problemini analiz eder ve uygun dengeleme yaklaşımını uygular.",
                "                                      (Random Undersampling, SMOTE, Scale Pos Weight=584.37 ve Data Leakage Guard)",
                "",
                "3. ÖĞRENCİ 3 (src/spark_features.py): Spark üzerinde müşteri ve zaman bazlı davranış özellikleri üretir.",
                "                                      (Son 1-6-24 saat işlem sıklığı Rolling Windows, Müşteri Tutar Z-Skoru, Delta Time)",
                "",
                "4. ÖĞRENCİ 4 (src/modeling.py)      : Fraud sınıflandırma modeli geliştirir; accuracy yerine precision, recall, F1, AUPRC inceler.",
                "                                      (LightGBM, XGBoost, Random Forest vs Naive Model ile Accuracy Paradox İspatı)",
                "",
                "5. ÖĞRENCİ 5 (src/experiments.py)   : Veri büyüklüğü ve sınıf dengesizliğinin eğitim süresi ve model performansına etkisini analiz eder.",
                "                                      (%10, %30, %60, %100 veri ölçeklerinde Gecikme vs AUPRC Doyum / Diminishing Returns Analizi)"
            ],
            bg_color="#f8fafc", border_color="#94a3b8", title_color="#0f172a", font_size=8.3, wrap_width=112
        )

        # Bölüm 0: Ana Hikaye
        draw_box(
            ax, 0.03, 0.25, 0.94, 0.24,
            "BÖLÜM 0: HEPİNİZİN BİLMESİ GEREKEN 'ANA HİKAYE' (BÜYÜK RESİM)",
            [
                "1. Olay Ne? Bir bankada saniyede binlerce kredi kartı harcaması geçer. Bunların %99.83'ü dürüst vatandaşın normal harcaması, sadece %0.17'si (yani her 588 işlemden sadece 1 tanesi) dolandırıcılıktır (Fraud).",
                "",
                "2. Zorluk Ne? Buna veri biliminde 'Extreme Class Imbalance' (Aşırı Sınıf Dengesizliği) denir. Eğer standart bir yapay zeka modeline bu veriyi düz verirseniz, model tembellik yapar ve gelen her işleme 'Normal' der geçer (%99.83 doğruluk alır ama 0 dolandırıcı yakalar!).",
                "",
                "3. Biz Nasıl Çözdük? 5 kişilik ekip olarak uçtan uca bir boru hattı kurduk: Öğrenci 1 veriyi ve gece/tutar dağılımını keşfetti -> Öğrenci 2 test verisine bilgi sızdırmadan (Data Leakage Guard) eğitim verisini dengeledi -> Öğrenci 3 PySpark pencereleriyle müşterinin son 1-6-24 saatlik harcama alışkanlıklarını çıkardı -> Öğrenci 4 modelleri eğitip Accuracy Paradox'u kanıtladı ve AUPRC ile değerlendirdi -> Öğrenci 5 ise veriyi %10, %30, %60, %100 boyutlarında test ederek 'Çok veri olmasının bir eşiğe (%60) kadar işe yaradığını ama sonrasında doyuma (plato noktasına) ulaştığını' ispatladı."
            ],
            bg_color="#f0fdf4", border_color="#16a34a", title_color="#15803d", font_size=8.4, wrap_width=106
        )

        # Sunum Günü Rehberi
        draw_box(
            ax, 0.03, 0.04, 0.94, 0.19,
            "SUNUM GÜNÜ BİLGİSAYAR BAŞINDA ADIM ADIM AKIŞ PLANI",
            [
                "• Adım 1 (Arayüzü Yansıtın): Klasördeki 'index.html' veya 'run_dashboard.bat' dosyasına çift tıklayın. Tarayıcıda 5 Sayfalık Sunum Arayüzü açılır.",
                "• Adım 2 (Projektör Modu): Sağ üstten 'Tam Ekran (F11)' butonuna basın. Sınıf aydınlıksa 'Projektör Açık Modu'na tıklayın (yüksek kontrast beyaz tema).",
                "• Adım 3 (Sırayla Sunum): Her öğrenci kendi sekmesinde (Sayfa 1..5) konuşmasını yapar, Sağ Ok (->) tuşuna basarak sırayı arkadaşına devreder.",
                "• Adım 4 (Canlı Kod Şovu): Hoca 'Çalışan kodu görelim' derse sağ üstteki yeşil 'Canlı Çalıştır (Live Demo)' butonuna basın veya terminalde 'python main.py --quick' yazın. 20 saniye içinde 5 öğrencinin tüm kodları canlı çalışıp ekrana metrik tablosunu basacaktır!"
            ],
            bg_color="#fffbeb", border_color="#f59e0b", title_color="#b45309", font_size=8.3, wrap_width=106
        )

        pdf.savefig(fig)
        plt.close(fig)

        # =====================================================================
        # SAYFA 2: ÖĞRENCİ 1 (VERİ HAZIRLIĞI & EDA)
        # =====================================================================
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        draw_header_banner(
            ax,
            "ÖĞRENCİ 1: VERİ HAZIRLIĞI VE KEŞİFSEL VERİ ANALİZİ (EDA)",
            "Görev: İşlem veri setini hazırlar ve veri dağılımını inceler.",
            "src/data_prep.py"
        )

        # Grafikleri Göm
        embed_image(fig, "reports/figures/01_class_distribution.png", [0.04, 0.67, 0.44, 0.20], "Grafik 1: Sınıf Dağılımı (Lineer vs Logaritmik)")
        embed_image(fig, "reports/figures/02_amount_time_distribution.png", [0.51, 0.67, 0.45, 0.20], "Grafik 2: Amount (Log) ve Gün İçi Saat Dağılımı")

        # Öğrenme Kutusu
        draw_box(
            ax, 0.03, 0.44, 0.94, 0.21,
            "1. ÖNCE SEN ÖĞREN (Teknik Arka Plan - Bilmen Gerekenler)",
            [
                "• Veri Setinde Neler Var? Kaggle formatında 32 sütunumuz var: 'Time' (48 saatlik işlem saniyesi), 'V1..V28' (Müşteri gizliliği KVKK/GDPR için PCA Temel Bileşen Analizi ile şifrelenmiş sayısal özellikler), 'Amount' (Tutar), 'Class' (0: Normal %99.83, 1: Fraud %0.17) ve 'customer_id'.",
                "• Neden Logaritmik Ölçek Kullandık? Normal işlemler 29,949 adet iken Fraud sadece 51 adettir (1:587 oranı). Lineer grafikte 51 işlem sıfır çizgisinde kaybolur. Logaritmik ölçek (Log10) kullanarak iki sınıfı aynı grafikte görünür hale getirdik.",
                "• EDA'da Keşfettiğimiz 2 Kritik Bulgu: (1) Normal harcamalar gündüz yoğunlaşırken, dolandırıcılıklar gece yarısı ve sabaha karşı (kart sahipleri uyurken) zirve yapar. (2) Amount sağa çarpık olduğu için log(1+Amount) dönüşümü uyguladık; dolandırıcılar ya kartı denemek için çok küçük mikro tutarlar ya da limiti boşaltmak için yüksek tutarlar çekmektedir."
            ],
            bg_color="#f8fafc", border_color="#3b82f6", title_color="#1d4ed8", font_size=8.3, wrap_width=106
        )

        # Konuşma Metni Kutusu
        draw_box(
            ax, 0.03, 0.16, 0.94, 0.26,
            "2. SUNUMDA BİREBİR KONUŞMA METNİ (Sıra Sana Geldiğinde Böyle Anlat)",
            [
                "\"Merhaba hocam ve arkadaşlar. Projemizin ilk aşaması olan İşlem Veri Setinin Hazırlanması ve Veri Dağılımının İncelenmesi (EDA) görevini ben gerçekleştirdim (src/data_prep.py).",
                "",
                "İlk olarak kredi kartı işlem verimizi incelediğimizde, bu problemin neden klasik bir sınıflandırma problemi olmadığını hemen görüyoruz. Veri setimizde işlemlerin %99.83'ü normal (Class 0) iken, yalnızca %0.17'si dolandırıcılık (Class 1) işlemidir. Yani Dengesizlik Oranımız (Imbalance Ratio) 1'e 587'dir. Her 1 dolandırıcı işleme karşılık 587 adet dürüst harcama bulunmaktadır.",
                "",
                "(Grafik 1'i göstererek) İlk grafiğimizin sol tarafında gördüğünüz gibi, lineer ölçekte baktığımızda dolandırıcılık işlemleri neredeyse sıfır çizgisinde kaybolmaktadır. Bu yüzden sağ tarafta logaritmik ölçek kullanarak azınlık sınıfını görünür hale getirdik.",
                "",
                "(Grafik 2'yi göstererek) Veri dağılımını Amount (Tutar) ve Time (Zaman) ekseninde incelediğimizde ise iki kritik örüntü yakaladık: Birincisi; normal harcamalar gündüz saatlerinde düzenli dağılırken, dolandırıcılık işlemleri özellikle gece yarısı ve sabaha karşı (kart sahipleri uyurken) yoğunlaşmaktadır. İkincisi; Amount değişkenini log(1+Amount) dönüşümüyle incelediğimizde dolandırıcıların ya kartı test etmek için ufak tutarlar ya da yüksek riskli tutarlar çektiğini tespit ettik. Şimdi sözü sınıf dengesizliği çözümleri için 2. arkadaşımıza bırakıyorum.\""
            ],
            bg_color="#eff6ff", border_color="#2563eb", title_color="#1e40af", font_size=8.3, wrap_width=106
        )

        # Soru Cevap
        draw_box(
            ax, 0.03, 0.03, 0.94, 0.11,
            "3. HOCA SORARSA ŞOV YAPACAĞIN CEVAPLAR",
            [
                "• Hoca: 'Verideki V1..V28 nedir, neden açık isimleri yok?' -> Cevap: 'Hocam, finansal verilerde kişisel gizlilik nedeniyle ham özellikler paylaşılmaz; banka bu özellikleri PCA (Temel Bileşen Analizi) ile dönüştürmüştür. Yalnızca Time ve Amount orijinal ölçeğindedir.'",
                "• Hoca: 'Tutar analizinde neden logaritma aldınız?' -> Cevap: 'Çünkü işlem tutarlarında aşırı uç değerler (sağa çarpıklık) var; log(1+Amount) dönüşümü dağılımı normalize ederek sınıflar arası farkı netleştirdi.'"
            ],
            bg_color="#fffbeb", border_color="#f59e0b", title_color="#b45309", font_size=8.1, wrap_width=108
        )

        pdf.savefig(fig)
        plt.close(fig)

        # =====================================================================
        # SAYFA 3: ÖĞRENCİ 2 (SINIF DENGESİZLİĞİ & SIZINTI KORUMASI)
        # =====================================================================
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        draw_header_banner(
            ax,
            "ÖĞRENCİ 2: SINIF DENGESİZLİĞİ (CLASS IMBALANCE) ÇÖZÜMLERİ",
            "Görev: Class imbalance problemini analiz eder ve uygun dengeleme yaklaşımını uygular.",
            "src/imbalance.py"
        )

        # Öğrenme Kutusu
        draw_box(
            ax, 0.03, 0.62, 0.94, 0.26,
            "1. ÖNCE SEN ÖĞREN (Teknik Arka Plan - Bilmen Gerekenler)",
            [
                "• Model Neden Bozulur? Verinin %99.83'ü Normal (0) olduğu için, standart bir algoritma hiçbir şey öğrenmeyip herkese '0' dese bile hatası sadece %0.17 çıkar! Model azınlık sınıfını (Fraud) gürültü sanıp yok sayar.",
                "• Kurduğun 3 Dengeleme Tekniği Nedir?",
                "  1) Random Undersampling (Rastgele Alt Örnekleme): Çoğunluk sınıfından (Normal işlemlerden) rastgele kayıt silerek sayıyı dolandırıcılık sayısına yaklaştırır. Artısı: Eğitim çok hızlanır. Eksisi: Silinen normal işlemlerle önemli bilgi kaybı (Information Loss) yaşanabilir.",
                "  2) SMOTE (Synthetic Minority Over-sampling Technique): Fraud örneklerini kuru kuruya kopyalamak yerine, uzayda birbirine yakın iki fraud işleminin arasına k-NN (en yakın komşuluk) ile matematiksel olarak YENİ SENTETİK fraud örnekleri üretir.",
                "  3) Algoritmik Dengeleme (scale_pos_weight = 584.37 / class_weight): Veriyi silmez, sentetik veri de üretmez! Karar ağacının ceza fonksiyonuna der ki: '1 tane Fraud işlemini kaçırırsan sana 1 normal işlemin 584 katı ceza keserim!' (Negatif Sayısı / Pozitif Sayısı = 584.37).",
                "• EN KRİTİK KAVRAM - Data Leakage (Veri Sızıntısı) Koruması: SMOTE tüm veriye uygulanıp sonra train_test_split yapılırsa test setine sentetik veri sızar! Sen 'DataLeakageGuard' sınıfını yazdın: Veri önce Stratified olarak %80 Train ve %20 Test diye bölünür; dengeleme SADECE TRAIN setine uygulanır! Test setine asla dokunulmaz."
            ],
            bg_color="#f8fafc", border_color="#3b82f6", title_color="#1d4ed8", font_size=8.2, wrap_width=106
        )

        # Konuşma Metni Kutusu
        draw_box(
            ax, 0.03, 0.29, 0.94, 0.31,
            "2. SUNUMDA BİREBİR KONUŞMA METNİ (Sıra Sana Geldiğinde Böyle Anlat)",
            [
                "\"Teşekkürler. İlk arkadaşımızın ortaya koyduğu 1'e 587'lik aşırı sınıf dengesizliği (Class Imbalance) problemini analiz etmek ve uygun dengeleme yaklaşımlarını uygulamak benim görevimdi (src/imbalance.py).",
                "",
                "Eğer bu dengesiz veriyi doğrudan modele verirsek, model çoğunluk sınıfı olan normal işlemlere aşırı uyum sağlar (majority bias) ve nadir dolandırıcılık işlemlerini göz ardı eder. Bunu aşmak için modülümde 3 farklı yaklaşımı modüler olarak kurguladım:",
                "1. Birincisi Random Undersampling: Çoğunluk sınıfındaki normal işlemleri alt örnekleyerek denge sağlamak.",
                "2. İkincisi SMOTE: Azınlık sınıfındaki dolandırıcılık örneklerinin k-en yakın komşularını kullanarak özellik uzayında sentetik yeni fraud örnekleri üretmek.",
                "3. Üçüncüsü ise Algoritmik Maliyet Ağırlıklandırma (scale_pos_weight ve balanced class_weight): Veri hacmini yapay olarak değiştirmeden, karar ağaçlarının kayıp fonksiyonunda azınlık sınıfına 584.37 kat daha yüksek hata cezası vermektir.",
                "",
                "Ancak hocam, benim bu modülde kurduğum en kritik mühendislik yapısı 'DataLeakageGuard' (Veri Sızıntısı Koruma) katmanıdır. Literatürde sık yapılan bir hata, SMOTE gibi dengeleme işlemlerini tüm veriye uygulayıp sonrasında Train-Test ayrımı yapmaktır. Bu yapıldığında hem eğitimden test kümesine bilgi sızar hem de test kümesinin gerçek dünya dağılımı bozulur.",
                "",
                "Bizim boru hattımızda ise veri önce Stratified (katmanlı) olarak %80 Eğitim ve %20 Test şeklinde izole edilmekte; SMOTE veya Undersampling KESİNLİKLE yalnızca Eğitim (Train) kümesine uygulanmaktadır. Test kümemiz ise gerçek dünyadaki bir bankanın canlı veri akışını temsil edecek şekilde el değmemiş 1:587 oranında bırakılmıştır. Şimdi sözü PySpark özellik mühendisliği için 3. arkadaşımıza devrediyorum.\""
            ],
            bg_color="#eff6ff", border_color="#2563eb", title_color="#1e40af", font_size=8.2, wrap_width=106
        )

        # Soru Cevap Kutusu
        draw_box(
            ax, 0.03, 0.04, 0.94, 0.23,
            "3. HOCA SORARSA ŞOV YAPACAĞIN CEVAPLAR",
            [
                "• Hoca: 'Bu 3 dengeleme yönteminden hangisi en iyi sonucu verdi?'",
                "  Cevap: 'Hocam, büyük veride ağaç tabanlı boosting modellerinde (XGBoost ve LightGBM) verinin orijinal dağılımını bozmayan scale_pos_weight (Algoritmik Ağırlıklandırma) en yüksek verimi verdi. Çünkü Undersampling gibi bilgi kaybına yol açmıyor ve SMOTE gibi veri boyutunu şişirip eğitim süresini uzatmıyor.'",
                "",
                "• Hoca: 'Stratified Split ne demek, neden normal rastgele bölmediniz?'",
                "  Cevap: 'Hocam verimizde zaten çok az (51 adet) fraud var. Rastgele bölseydik şans eseri tüm fraudlar eğitim setine düşüp test setinde hiç fraud kalmayabilirdi. Stratified split, hem eğitim hem de test setinde %0.172'lik fraud oranının birebir aynı kalmasını garanti eder.'",
                "",
                "• Hoca: 'Test setine de SMOTE uygulasaydınız ne olurdu?'",
                "  Cevap: 'Test setinin doğal dengesizliği bozulurdu ve model yapay sentetik örnekleri tahmin edeceği için skorlar sahte şekilde yüksek çıkardı (Data Leakage). Canlı banka ortamına alındığında ise binlerce yanlış alarm verip çökerdi.'"
            ],
            bg_color="#fffbeb", border_color="#f59e0b", title_color="#b45309", font_size=8.1, wrap_width=106
        )

        pdf.savefig(fig)
        plt.close(fig)

        # =====================================================================
        # SAYFA 4: ÖĞRENCİ 3 (PYSPARK DAVRANIŞSAL ÖZELLİK MÜHENDİSLİĞİ)
        # =====================================================================
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        draw_header_banner(
            ax,
            "ÖĞRENCİ 3: PYSPARK İLE DAVRANIŞSAL ÖZELLİK MÜHENDİSLİĞİ",
            "Görev: Spark üzerinde müşteri ve zaman bazlı davranış özellikleri üretir.",
            "src/spark_features.py"
        )

        # Öğrenme Kutusu
        draw_box(
            ax, 0.03, 0.60, 0.94, 0.28,
            "1. ÖNCE SEN ÖĞREN (Teknik Arka Plan - Bilmen Gerekenler)",
            [
                "• Neden Davranışsal Özellik (Behavioral Feature Engineering) Üretiyoruz?",
                "  Ham veride her işlem tek başına bir satırdır (Örn: '5.000 TL çekildi'). Sadece tutara bakarak bunun dolandırıcılık olup olmadığını bilemezsin! Eğer o müşteri her gün 10.000 TL harcayan biriyse normaldir. Ama her gün 50 TL harcayan bir öğrencinin kartından gece 1 saat içinde 5 defa üst üste 5.000 TL çekildiyse bu %100 dolandırıcılıktır! İşte işlemi müşterinin geçmiş alışkanlığıyla kıyaslamaya Davranışsal Özellik Mühendisliği denir.",
                "",
                "• PySpark Window (Pencereleme) İle Ürettiğin 7 Yeni Özellik Nedir?",
                "  Veriyi müşteriye göre gruplayıp (Window.partitionBy('customer_id')) zamana göre sıraladık (orderBy('Time_sec')):",
                "  1) Son 1, 6 ve 24 Saatteki İşlem Sayıları (tx_count_last_1h, 6h, 24h): rangeBetween(-3600, 0) kullanarak müşterinin son 1 saat, 6 saat ve 24 saatteki işlem sıklığını saydık. (Çalıntı kartla bloke edilmeden önce seri harcama yapılır - Velocity Attack).",
                "  2) Müşteri Ortalama Tutarı ve Sapma (customer_avg_amount, amount_deviation, amount_zscore): Müşterinin o işlem anına kadarki kümülatif harcama ortalamasını ve standart sapmasını bulup Z-Skoru hesapladık: Z = (Tutar - Ortalama) / Std.",
                "  3) İşlemler Arası Geçen Süre (delta_time_sec): F.lag('Time_sec', 1) ile müşterinin bir önceki işleminden bu yana kaç saniye geçtiğini bulduk.",
                "• PySpark ML Pipeline: VectorAssembler ve StandardScaler ile toplam özellik sayısını 30'dan 37'ye çıkarıp vektörleştirdik."
            ],
            bg_color="#f8fafc", border_color="#3b82f6", title_color="#1d4ed8", font_size=8.1, wrap_width=106
        )

        # Konuşma Metni Kutusu
        draw_box(
            ax, 0.03, 0.28, 0.94, 0.30,
            "2. SUNUMDA BİREBİR KONUŞMA METNİ (Sıra Sana Geldiğinde Böyle Anlat)",
            [
                "\"Teşekkür ederim. Benim görevim, PySpark üzerinde müşteri ve zaman bazlı davranışsal özellikler (Behavioral Features) üretmekti (src/spark_features.py).",
                "",
                "Gerçek dünya bankacılığında bir işlemin fraud olup olmadığını sadece o işlemin tutarına bakarak anlayamayız. Örneğin 5.000 TL'lik bir işlem bir müşteri için çok sıradanken, başka bir müşteri için büyük bir anomalidir. Dolayısıyla modelin müşterinin geçmiş harcama karakterini bilmesi gerekir.",
                "",
                "Büyük veri ölçeğinde bu hesaplamaları dağıtık yapabilmek için PySpark SQL ve Window (Pencereleme) mimarisini kullandım. Veriyi customer_id bazında bölümleyip (partitionBy) işlem zamanına göre sıralayarak (orderBy) şu 3 kritik davranış grubunda toplam 7 yeni özellik türettim:",
                "1. Birincisi; Kayan Zaman Pencereleri (Rolling Window Counts): rangeBetween kullanarak müşterinin son 1 saat, son 6 saat ve son 24 saat içinde yaptığı işlem sayılarını hesapladım. Çünkü kredi kartı çalındığında, dolandırıcılar kart iptal edilmeden önce kısa sürede çok sayıda ardışık işlem (Velocity Attack) gerçekleştirirler.",
                "2. İkincisi; Müşteri Bazlı Tutar Sapması ve Z-Skoru: Her müşterinin o işlem anına kadarki kümülatif harcama ortalamasını (customer_avg_amount) hesaplayarak, güncel işlemin müşterinin kendi geçmişinden kaç standart sapma uzaklaştığını gösteren Z-Score (amount_zscore) özelliğini ürettim.",
                "3. Üçüncüsü; İşlemler Arası Geçen Süre (delta_time_sec): PySpark lag fonksiyonu ile müşterinin bir önceki işlemiyle şu anki işlemi arasında geçen saniye farkını çıkardım.",
                "",
                "Son aşamada PySpark VectorAssembler ve StandardScaler kullanarak toplam özellik sayımızı 30'dan 37 zengin özelliğe çıkardım ve modelleme aşamasına hazır hale getirdim. Şimdi sözü modelleme ve metrik analizi için 4. arkadaşımıza bırakıyorum.\""
            ],
            bg_color="#eff6ff", border_color="#2563eb", title_color="#1e40af", font_size=8.2, wrap_width=106
        )

        # Soru Cevap Kutusu
        draw_box(
            ax, 0.03, 0.04, 0.94, 0.22,
            "3. HOCA SORARSA ŞOV YAPACAĞIN CEVAPLAR",
            [
                "• Hoca: 'Müşteri ortalamasını hesaplarken tüm tablonun ortalamasını mı aldınız?'",
                "  Cevap: 'Hayır hocam! Tüm zamanların ortalamasını alsaydık geleceğe ait işlemleri bugünkü işleme katmış olurduk (Look-ahead bias / zaman sızıntısı). Bunun yerine rowsBetween(Window.unboundedPreceding, 0) yani sadece o işlem anına kadar gerçekleşmiş geçmiş işlemlerin kümülatif ortalamasını aldık.'",
                "",
                "• Hoca: 'Neden bu işlem için PySpark tercih ettiniz?'",
                "  Cevap: 'Çünkü hocam büyük ölçekli banka verilerinde milyonlarca kartın kayan pencere (rolling window) hesaplamaları tek çekirdekli yapılarda bellek darboğazı yaratır. PySpark Window API, customer_id partition'larını farklı executor'larda paralel olarak hesaplayabilmektedir.'"
            ],
            bg_color="#fffbeb", border_color="#f59e0b", title_color="#b45309", font_size=8.1, wrap_width=106
        )

        pdf.savefig(fig)
        plt.close(fig)

        # =====================================================================
        # SAYFA 5: ÖĞRENCİ 4 (MODELLER, ACCURACY PARADOX & AUPRC)
        # =====================================================================
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        draw_header_banner(
            ax,
            "ÖĞRENCİ 4: MODEL GELİŞTİRME, ACCURACY PARADOX & AUPRC ANALİZİ",
            "Görev: Fraud sınıflandırma modeli geliştirir; accuracy yerine precision, recall, F1, AUPRC inceler.",
            "src/modeling.py"
        )

        # Grafikleri Göm
        embed_image(fig, "reports/figures/04_confusion_matrices.png", [0.04, 0.70, 0.45, 0.18], "Grafik 3: Modellerin Karmaşıklık Matrisleri")
        embed_image(fig, "reports/figures/05_pr_and_roc_curves.png", [0.51, 0.70, 0.45, 0.18], "Grafik 4: PR Eğrisi (AUPRC) vs ROC Eğrisi")

        # Öğrenme Kutusu
        draw_box(
            ax, 0.03, 0.45, 0.94, 0.23,
            "1. ÖNCE SEN ÖĞREN (Accuracy Paradox ve AUPRC vs ROC-AUC Mantığı)",
            [
                "• Accuracy Paradox (Doğruluk Paradoksu) Nedir? Test setimizde 6.000 işlem var (5.990 Normal, 10 Fraud). Gelen her işleme gözü kapalı 'Normal (0)' diyen kukla Naive Model, 6.000 işlemin 5.990'ını doğru bildiği için %99.83 Accuracy (Doğruluk) alır! Ama yakaladığı dolandırıcı sayısı 0 (Sıfır)'dır (Recall = %0.0)! İşte doğruluğun %99.8 çıkıp modelin tamamen kör olmasına Accuracy Paradox denir.",
                "• Doğru Metrikler Nelerdir?",
                "  - Recall (Yakalama Oranı = TP / (TP+FN)): Gerçek dolandırıcıların kaçını yakaladık? (XGBoost ve Random Forest %100 Recall ile 10'da 10 yakaladı, LightGBM %90 yakaladı).",
                "  - Precision (Kesinlik = TP / (TP+FP)): Fraud diye alarm verdiğimiz işlemlerin kaçı gerçekten fraud? (XGBoost %90.9 Precision verdi).",
                "  - Neden ROC-AUC Değil de AUPRC (PR-AUC)? ROC eğrisindeki False Positive Rate (FP / (FP+TN)) paydasında 5.990 adet Normal işlem (TN) olduğu için çok sayıda yanlış alarm verilse bile ROC-AUC %95+ görünür. AUPRC ise TN'yi denkleme hiç sokmaz, sadece Precision ve Recall'a bakar!"
            ],
            bg_color="#f8fafc", border_color="#3b82f6", title_color="#1d4ed8", font_size=8.1, wrap_width=106
        )

        # Konuşma Metni Kutusu
        draw_box(
            ax, 0.03, 0.16, 0.94, 0.27,
            "2. SUNUMDA BİREBİR KONUŞMA METNİ (Sıra Sana Geldiğinde Böyle Anlat)",
            [
                "\"Teşekkürler. İlk üç arkadaşımın hazırladığı, dengelediği ve PySpark davranış özellikleriyle zenginleştirdiği veri seti üzerinde Fraud Sınıflandırma Modellerini Geliştirmek ve Accuracy Yerine Precision, Recall, F1 ve AUPRC Ölçütlerini İncelemek benim görevimdi (src/modeling.py).",
                "",
                "Bu kapsamda Random Forest, LightGBM ve XGBoost modellerini eğittim. Ayrıca Accuracy (Doğruluk) metriğinin bu problemdeki anlamsızlığını kanıtlamak için her işleme 'Normal' diyen bir Naive Majority referans modeli kurdum.",
                "",
                "(Grafik 3 Confusion Matrix'i göstererek) Hocam, sonuçlarımız Accuracy Paradox (Doğruluk Paradoksu) kavramını net biçimde kanıtlamaktadır: Hiçbir öğrenme yapmayan Naive model, test setindeki 6.000 işlemin tamamına 'Normal' diyerek %99.83 Accuracy elde etmiştir. Ancak Confusion Matrix'e baktığımızda test setindeki 10 dolandırıcılığın 0 tanesini yakalamış, tamamını kaçırmıştır (Recall = %0.0).",
                "",
                "Bu yüzden Accuracy yerine 4 kritik ölçüte odaklandık: Recall (yakalama oranımız), Precision (alarmlarımızın kesinliği), F1/F2-Score ve AUPRC. (Grafik 4'ü göstererek) Sağdaki ROC eğrisi, paydadaki devasa True Negative sayısı yüzünden aşırı iyimser görünürken; soldaki Precision-Recall (AUPRC) eğrisi modellerin gerçek gücünü göstermektedir. XGBoost (%100 Recall, %90.9 Precision, 1.00 AUPRC) ve Random Forest modellerimiz dolandırıcılıkların tamamını başarıyla yakalamıştır. Şimdi sözü ölçek deneyleri için 5. arkadaşımıza bırakıyorum.\""
            ],
            bg_color="#eff6ff", border_color="#2563eb", title_color="#1e40af", font_size=8.1, wrap_width=106
        )

        # Soru Cevap Kutusu
        draw_box(
            ax, 0.03, 0.03, 0.94, 0.11,
            "3. HOCA SORARSA ŞOV YAPACAĞIN CEVAPLAR",
            [
                "• Hoca: 'Banka için Precision mı daha önemlidir Recall mu?' -> Cevap: 'Hocam Recall önceliklidir; çünkü bir dolandırıcılığı kaçırmanın (False Negative) maliyeti binlerce dolarlık doğrudan para kaybıdır. Yanlış alarmın (False Positive) maliyeti ise müşteriye giden bir onay SMS'idir. Bu yüzden Recall'a 2 kat ağırlık veren F2-Score'u da hesapladık.'"
            ],
            bg_color="#fffbeb", border_color="#f59e0b", title_color="#b45309", font_size=8.1, wrap_width=106
        )

        pdf.savefig(fig)
        plt.close(fig)

        # =====================================================================
        # SAYFA 6: ÖĞRENCİ 5 (ÖLÇEK DENEYİ & ARAŞTIRMA SORUSUNUN CEVABI)
        # =====================================================================
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        draw_header_banner(
            ax,
            "ÖĞRENCİ 5: ÖLÇEK VE DENGESİZLİK ETKİ ANALİZİ (ARAŞTIRMA SORUSU)",
            "Görev: Veri büyüklüğü ve sınıf dengesizliğinin eğitim süresi ve model performansına etkisini analiz eder.",
            "src/experiments.py"
        )

        # Grafiği Göm
        embed_image(fig, "reports/figures/06_scale_experiment_results.png", [0.03, 0.70, 0.94, 0.18], "Grafik 5: Veri Ölçeği vs AUPRC/Recall, Eğitim Süresi ve Maliyet-Performans Analizi")

        # Öğrenme Kutusu
        draw_box(
            ax, 0.03, 0.45, 0.94, 0.23,
            "1. ÖNCE SEN ÖĞREN (Araştırma Sorusunun Deneysel Kanıtı ve Tablosu)",
            [
                "• Araştırma Sorusu: 'Çok veri olması, nadir fraud olaylarını bulmayı gerçekten kolaylaştırıyor mu?'",
                "• Deney Sonuç Tablosu (1:587 dengesizlik oranını koruyarak 4 farklı veri boyutunda eğitim):",
                "  - %10 Ölçek (2,400 işlem | Sadece 4 Fraud) -> Eğitim Süresi: 1.29 sn | Test Recall: %60.0 | AUPRC: 0.8947",
                "  - %30 Ölçek (7,200 işlem | 12 Fraud)       -> Eğitim Süresi: 1.26 sn | Test Recall: %60.0 | AUPRC: 0.0081",
                "  - %60 Ölçek (14,400 işlem | 25 Fraud)      -> Eğitim Süresi: 0.58 sn | Test Recall: %100.0 (Zirve!) | AUPRC: 0.5556",
                "  - %100 Ölçek (24,000 işlem | 41 Fraud)     -> Eğitim Süresi: 2.33 sn (4 Kat Süre!) | Test Recall: %90.0 (Plato) | AUPRC: 0.5064",
                "• 3 Bilimsel Çıkarım: (1) Çok az veride (%10-%30) sadece 4-12 fraud olduğu için model kuralları öğrenemez (%60 Recall). (2) %60 ölçeğinde (25 fraud) kritik eşik aşılır ve Recall %100'e sıçrar. (3) %60'tan %100'e çıkarken süre 4 katına fırlar ama başarı artmaz (Diminishing Returns); çünkü her 1 yeni fraud yanında 587 yeni normal işlem (gürültü/False Positive) gelmeye devam eder!"
            ],
            bg_color="#f8fafc", border_color="#3b82f6", title_color="#1d4ed8", font_size=8.0, wrap_width=106
        )

        # Konuşma Metni Kutusu
        draw_box(
            ax, 0.03, 0.16, 0.94, 0.27,
            "2. SUNUMDA BİREBİR KONUŞMA METNİ (Sıra Sana Geldiğinde Böyle Anlat)",
            [
                "\"Teşekkürler. Projemizin son aşamasında, çalışmamızın merkezinde yer alan araştırma sorusunu yanıtlamak üzere Veri Büyüklüğü ve Sınıf Dengesizliğinin Eğitim Süresi ve Model Performansına Etkisini analiz ettim (src/experiments.py).",
                "",
                "Araştırma sorumuz şuydu: 'Çok veri olması, nadir fraud olaylarını bulmayı gerçekten kolaylaştırıyor mu?' Bunu test etmek için eğitim verimizi 1'e 587 dengesizlik oranını koruyarak %10, %30, %60 ve %100 olmak üzere 4 farklı ölçekte eğittim; her ölçekte Eğitim Süresini ve Model Performansını (AUPRC ve Recall) ölçtüm.",
                "",
                "(Grafik 5'i göstererek) Deney sonuçlarımız bize çok net bir yanıt vermektedir: KISMEN EVET, ANCAK BELİRLİ BİR EŞİKTEN SONRA DOYUMA ULAŞMAKTADIR (Diminishing Returns)!",
                "1. Kritik Azınlık Hacmi (%10 -> %60): %10 ve %30 veri boyutunda model yalnızca 4 ila 12 adet fraud örneği görmektedir; bu yetersiz kaldığı için Recall %60'ta kalır. Ancak veri %60 boyutuna (25 fraud işlemine) ulaştığında model örüntüyü yakalamakta ve Recall %100'e sıçramaktadır.",
                "2. Gürültü ve Yanlış Alarm Baskısı (%60 -> %100): Sadece veri miktarını artırmak 1:587 olan dengesizlik oranını iyileştirmemektedir. Sisteme eklenen her 1 yeni fraud işlemine karşılık 587 yeni normal işlem gelmektedir. Bu devasa normal işlem havuzu yanlış alarmları tetikleyerek AUPRC artışını frenlemektedir.",
                "3. Süre vs Performans Trade-off'u: %60 ölçeğinden %100 ölçeğine geçildiğinde eğitim süresi 4 katına (0.58 sn'den 2.33 sn'ye) çıkarken, model başarımı plato yapmaktadır. Sonuç olarak; sadece daha fazla ham veri yığmak yerine PySpark ile davranışsal özellikler üretmek (Öğrenci 3) ve doğru dengeleme yapmak (Öğrenci 2) asıl başarıyı getirmektedir. Dinlediğiniz için teşekkür ederiz.\""
            ],
            bg_color="#eff6ff", border_color="#2563eb", title_color="#1e40af", font_size=8.0, wrap_width=106
        )

        # Soru Cevap Kutusu
        draw_box(
            ax, 0.03, 0.03, 0.94, 0.11,
            "3. HOCA SORARSA ŞOV YAPACAĞIN CEVAPLAR",
            [
                "• Hoca: 'O zaman bankalara çok veri toplamasınlar mı diyorsunuz?' -> Cevap: 'Hayır hocam; %10'dan %60'a çıkarken gördüğümüz gibi modelin nadir fraud paternlerini öğrenebilmesi için minimum bir kritik veri eşiğine kesinlikle ihtiyacı var. Ancak bu eşik aşıldıktan sonra sadece ham satır eklemek yerine PySpark ile davranışsal özellik kalitesini (Feature Engineering) artırmak gerekiyor.'"
            ],
            bg_color="#fffbeb", border_color="#f59e0b", title_color="#b45309", font_size=8.0, wrap_width=106
        )

        pdf.savefig(fig)
        plt.close(fig)

    print(f"PDF basariyla olusturuldu: {output_pdf_path}")


if __name__ == "__main__":
    generate_presentation_pdf()
