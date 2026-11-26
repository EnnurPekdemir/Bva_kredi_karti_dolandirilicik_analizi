"""
main.py
=============================================================================
PROJE 4: KREDİ KARTI DOLANDIRICILIK ANALİZİ (CREDIT CARD FRAUD DETECTION)
UÇTAN UCA ORKESTRASYON VE ÇALIŞTIRMA PİPELINE'I
=============================================================================
Bu ana betik 5 öğrencinin geliştirdiği modülleri sırasıyla çalıştırarak
tam bir büyük veri ve makine öğrenimi boru hattını (pipeline) icra eder:

1. ÖĞRENCİ 1 (src/data_prep.py)     : Veri Yükleme/Üretme & EDA Raporlaması
2. ÖĞRENCİ 3 (src/spark_features.py): PySpark ile Davranışsal Özellik Mühendisliği
3. ÖĞRENCİ 2 (src/imbalance.py)     : Sınıf Dengesizliği Çözümleri & Sızıntı Koruması
4. ÖĞRENCİ 4 (src/modeling.py)      : Model Eğitimi, Accuracy Paradox & Metrikler
5. ÖĞRENCİ 5 (src/experiments.py)   : Ölçek Deneyi & Araştırma Sorusu Raporu
=============================================================================
"""

import os
import sys

# Türkçe Windows (cp1254) konsollarında UnicodeEncodeError çökmesini önlemek için UTF-8 zorlaması
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Matplotlib cache dizinini proje icine yonlendir ve joblib loky cekirdek uyarisini sustur
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mpl_cache")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import argparse
import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from tabulate import tabulate

from src.data_prep import (
    load_or_generate_credit_card_data,
    run_eda_and_generate_report
)
from src.imbalance import (
    ImbalanceAnalyzer,
    DataLeakageGuard,
    apply_sampling_strategy,
    get_scale_pos_weight
)
from src.spark_features import (
    get_spark_session,
    engineer_features_spark
)
from src.modeling import (
    train_and_evaluate_models,
    demonstrate_accuracy_paradox,
    plot_confusion_matrices,
    plot_pr_and_roc_curves
)
from src.experiments import (
    run_scaling_experiment,
    plot_experiment_results,
    generate_research_conclusion
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("MainPipeline")


def run_pipeline(
    data_path: str = "data/creditcard.csv",
    n_samples: int = 120_000,
    quick_run: bool = False,
    sampling_method: str = "none"
) -> None:
    """
    Tüm boru hattını (pipeline) sırasıyla yürütür.
    """
    print("\n" + "="*80)
    print("PROJE 4: KREDI KARTI DOLANDIRICILIK ANALIZI - PIPELINE BASLATILIYOR")
    print("="*80 + "\n")

    os.makedirs("reports/figures", exist_ok=True)
    os.makedirs("reports/metrics", exist_ok=True)

    # -------------------------------------------------------------------------
    # ADIM 1: ÖĞRENCİ 1 - VERİ HAZIRLIĞI & EDA
    # -------------------------------------------------------------------------
    logger.info(">>> ADIM 1: OGRENCI 1 - Veri Hazirligi ve Kesifsel Veri Analizi (EDA)...")
    sample_size = 30_000 if quick_run else n_samples
    raw_df = load_or_generate_credit_card_data(
        file_path=data_path,
        n_samples=sample_size,
        fraud_ratio=0.00172,
        random_state=42
    )
    run_eda_and_generate_report(raw_df)

    # -------------------------------------------------------------------------
    # ADIM 2: ÖĞRENCİ 3 - PYSPARK İLE BÜYÜK VERİ ÖZELLİK MÜHENDİSLİĞİ
    # -------------------------------------------------------------------------
    logger.info(">>> ADIM 2: OGRENCI 3 - PySpark ile Davranissal Ozellik Muhendisligi...")
    spark = get_spark_session("CreditCardFraudPipeline")
    featured_df, _ = engineer_features_spark(raw_df, spark_session=spark)

    if not isinstance(featured_df, pd.DataFrame):
        featured_df = featured_df.toPandas()

    feature_cols = [c for c in featured_df.columns if c not in ["Class", "customer_id"]]
    X = featured_df[feature_cols]
    y = featured_df["Class"].astype(int)

    logger.info(f"Ozellik muhendisligi sonrasi toplam ozellik sayisi: {len(feature_cols)}")

    # -------------------------------------------------------------------------
    # ADIM 3: ÖĞRENCİ 2 - SINIF DENGESİZLİĞİ VE SIZINTI (LEAKAGE) KORUMASI
    # -------------------------------------------------------------------------
    logger.info(">>> ADIM 3: OGRENCI 2 - Sinif Dengesizligi Analizi & Veri Sizintisi Korumasi...")
    imbalance_report = ImbalanceAnalyzer.analyze(y)
    logger.info(f"Dengesizlik Seviyesi: {imbalance_report['severity_level']} (IR: 1:{imbalance_report['imbalance_ratio_IR']})")

    # Stratified Train-Test Ayrımı (Doğal dengesizlik test setinde korunmalı)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # Veri Sızıntısı (Data Leakage) Güvenlik Doğrulaması
    DataLeakageGuard.verify_split_integrity(X_train, X_test, y_train, y_test)

    # Dengeleme stratejisi (SADECE Train setine uygulanır)
    X_train_res, y_train_res = apply_sampling_strategy(
        X_train, y_train, method=sampling_method, sampling_strategy=0.1, random_state=42
    )

    # Algoritmik ağırlık hesaplama (Dengelenmiş eğitim kümesi üzerinden stabilize edilerek hesaplanır)
    scale_pos_weight = get_scale_pos_weight(y_train_res, dampened=True)

    # -------------------------------------------------------------------------
    # ADIM 4: ÖĞRENCİ 4 - MODELLEME, ACCURACY PARADOX & METRİK DEĞERLENDİRMESİ
    # -------------------------------------------------------------------------
    logger.info(">>> ADIM 4: OGRENCI 4 - Model Egitimi & Kapsamli Metrik Degerlendirmesi...")
    models_dict, results_df, probs_dict = train_and_evaluate_models(
        X_train_res, y_train_res, X_test, y_test,
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )

    # Accuracy Paradox Analizi
    y_naive = models_dict["Naive Majority"].predict(X_test)
    best_model_name = "XGBoost" if "XGBoost" in models_dict else list(models_dict.keys())[-1]
    y_best = models_dict[best_model_name].predict(X_test)
    demonstrate_accuracy_paradox(y_test.values, y_naive, y_best, best_model_name)

    # Grafikleri oluştur
    plot_confusion_matrices(y_test.values, models_dict, X_test)
    plot_pr_and_roc_curves(y_test.values, probs_dict)

    # Metrik tablosunu konsola ve diske yaz (ASCII güvenli grid formatı)
    table_str = tabulate(results_df, headers="keys", tablefmt="grid", showindex=False)
    print("\n" + "="*80)
    print("MODEL PERFORMANS KARSILASTIRMA TABLOSU")
    print("="*80)
    print(table_str)

    results_csv_path = "reports/metrics/model_comparison_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    logger.info(f"Model metrikleri kaydedildi: {results_csv_path}")

    # -------------------------------------------------------------------------
    # ADIM 5: ÖĞRENCİ 5 - ÖLÇEK DENEYİ VE ARAŞTIRMA SORUSUNUN CEVABI
    # -------------------------------------------------------------------------
    logger.info(">>> ADIM 5: OGRENCI 5 - Olcek ve Dengesizlik Etki Analizi (Arastirma Sorusu)...")
    scales = [0.10, 0.30, 0.60, 1.00]
    exp_df = run_scaling_experiment(
        X_train_full=X_train,
        y_train_full=y_train,
        X_test=X_test,
        y_test=y_test,
        data_scales=scales,
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )

    plot_experiment_results(exp_df)
    conclusion_text = generate_research_conclusion(exp_df)
    exp_table_str = tabulate(exp_df, headers="keys", tablefmt="grid", showindex=False)

    print("\n" + "="*80)
    print("OLCEK DENEYI SONUC TABLOSU (%10 - %30 - %60 - %100)")
    print("="*80)
    print(exp_table_str)
    print(conclusion_text)

    # Deney tablosu ve sonucunu kaydet
    exp_csv_path = "reports/metrics/scaling_experiment_results.csv"
    exp_df.to_csv(exp_csv_path, index=False)

    report_txt_path = "reports/metrics/research_question_report.txt"
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("MODEL PERFORMANS KARSILASTIRMA TABLOSU\n")
        f.write(table_str + "\n\n")
        f.write("OLCEK DENEYI SONUC TABLOSU\n")
        f.write(exp_table_str + "\n")
        f.write(conclusion_text)

    print("\n" + "="*80)
    print("TUM BORU HATTI (PIPELINE) BASARIYLA TAMAMLANDI!")
    print("Raporlar ve grafikler 'reports/' dizini altina kaydedilmistir.")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kredi Karti Dolandiricilik Analizi Pipeline")
    parser.add_argument("--data-path", type=str, default="data/creditcard.csv", help="Veri seti dosya yolu")
    parser.add_argument("--samples", type=int, default=120_000, help="Uretilecek/kullanilacak ornek sayisi")
    parser.add_argument("--quick", action="store_true", help="Hizli test calistirmasi (30,000 islem)")
    parser.add_argument("--sampling", type=str, default="none", choices=["none", "undersample", "smote"], help="Dengeleme stratejisi")

    args = parser.parse_args()
    run_pipeline(
        data_path=args.data_path,
        n_samples=args.samples,
        quick_run=args.quick,
        sampling_method=args.sampling
    )
