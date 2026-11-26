"""
src/data_prep.py
=============================================================================
ÖĞRENCİ 1: Veri Hazırlığı & Keşifsel Veri Analizi (EDA)

Bu modül aşağıdaki görevleri yerine getirir:
1. Kaggle formatındaki kredi kartı dolandırıcılık veri setini yükler veya
   büyük ölçekli, dengesiz ve gerçekçi sentetik transaction verisi üretir.
2. Sınıf dağılımını (Class 0 vs Class 1), 'Amount' ve 'Time' değişkenlerinin
   sınıf bazlı özet istatistiklerini hesaplar.
3. İşlem hacmi, zaman dağılımı ve fraud oranlarını görselleştirip
   kapsamlı bir EDA raporu oluşturur.
=============================================================================
"""

import os
import json
import logging
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DataPrep")


def generate_synthetic_credit_card_data(
    n_samples: int = 150_000,
    fraud_ratio: float = 0.00172,
    n_customers: int = 5_000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Kaggle 'Credit Card Fraud Detection' formatına ve istatistiksel dağılımına
    birebir uygun, gerçekçi zorluk derecesine sahip büyük ölçekli veri seti üretir.
    """
    logger.info(f"Sentetik veri seti uretiliyor... (Toplam: {n_samples:,}, Fraud Orani: %{fraud_ratio*100:.3f})")
    rng = np.random.RandomState(random_state)

    n_fraud = max(10, int(n_samples * fraud_ratio))
    n_normal = n_samples - n_fraud

    # 1. Time (0 ile 172800 saniye arasi - 48 saatlik surec)
    time_normal = rng.uniform(0, 172800, size=n_normal)
    # Dolandiricilik islemleri gece saatlerinde (00:00-07:00) daha yogun, ama gunduz de gorulur
    n_f1 = int(n_fraud * 0.45)
    n_f2 = int(n_fraud * 0.35)
    n_f3 = n_fraud - n_f1 - n_f2
    time_fraud = np.concatenate([
        rng.uniform(0, 25200, size=n_f1),
        rng.uniform(86400, 111600, size=n_f2),
        rng.uniform(0, 172800, size=n_f3)
    ])
    time_all = np.concatenate([time_normal, time_fraud])

    # 2. V1 - V28 Özellikleri (Gerçekçi örtüşme ve zorluk seviyesinde PCA dağılımı)
    v_features_normal = rng.randn(n_normal, 28)
    v_features_fraud = rng.randn(n_fraud, 28) * 1.25

    # Fraud alt tipleri: %80 belirgin patern, %20 sınır bölgede (zor yakalanan) fraudlar
    hard_mask = rng.rand(n_fraud) < 0.18
    scale_factor = np.where(hard_mask, 0.70, 1.35)

    # V4 ve V11 pozitif kayma gosterir
    v_features_fraud[:, 3] += rng.normal(2.15, 0.85, size=n_fraud) * scale_factor  # V4
    v_features_fraud[:, 10] += rng.normal(1.95, 0.80, size=n_fraud) * scale_factor # V11
    # V10, V12, V14, V17 negatif kayma gosterir
    v_features_fraud[:, 9] -= rng.normal(2.20, 0.90, size=n_fraud) * scale_factor  # V10
    v_features_fraud[:, 11] -= rng.normal(2.55, 0.95, size=n_fraud) * scale_factor # V12
    v_features_fraud[:, 13] -= rng.normal(2.90, 1.00, size=n_fraud) * scale_factor # V14
    v_features_fraud[:, 16] -= rng.normal(2.40, 0.90, size=n_fraud) * scale_factor # V17

    v_all = np.vstack([v_features_normal, v_features_fraud])

    # 3. Amount (Log-normal dağılım)
    amount_normal = np.round(rng.lognormal(mean=3.2, sigma=1.35, size=n_normal), 2)
    n_a1 = n_fraud // 3
    n_a2 = n_fraud - n_a1
    amount_fraud = np.round(np.concatenate([
        rng.uniform(1.0, 15.0, size=n_a1),
        rng.lognormal(mean=4.8, sigma=1.25, size=n_a2)
    ]), 2)
    amount_all = np.concatenate([amount_normal, amount_fraud])

    # 4. Class (0: Normal, 1: Fraud)
    class_all = np.concatenate([np.zeros(n_normal, dtype=int), np.ones(n_fraud, dtype=int)])

    # 5. Customer ID (Pareto/Zipf dagilimli musteri islemleri)
    effective_customers = min(n_customers, max(500, n_samples // 15))
    customer_weights = 1.0 / (np.arange(1, effective_customers + 1) ** 0.75)
    customer_weights /= customer_weights.sum()
    customers_normal = rng.choice(np.arange(1001, 1001 + effective_customers), size=n_normal, p=customer_weights)

    # Fraud işlemlerinin bir kısmı aynı müşterinin kartından kısa sürede ardışık (velocity attack) yapılır
    fraud_cust_pool = rng.choice(np.arange(1001, 1001 + effective_customers), size=max(5, n_fraud // 2), replace=False)
    customers_fraud = rng.choice(fraud_cust_pool, size=n_fraud, replace=True)

    # Aynı müşteriye ait ardışık fraudların zamanlarını birbirine yaklaştır (son 1-6 saat penceresinde yakalansın)
    for cid in np.unique(customers_fraud):
        c_idx = np.where(customers_fraud == cid)[0]
        if len(c_idx) > 1:
            base_t = time_fraud[c_idx[0]]
            for k in range(1, len(c_idx)):
                time_fraud[c_idx[k]] = min(172790.0, base_t + rng.uniform(30, 1800) * k)

    time_all = np.concatenate([time_normal, time_fraud])
    customers = np.concatenate([customers_normal, customers_fraud])

    # DataFrame Olustur
    data_dict: Dict[str, Any] = {"Time": time_all}
    for i in range(28):
        data_dict[f"V{i+1}"] = v_all[:, i]
    data_dict["Amount"] = amount_all
    data_dict["Class"] = class_all
    data_dict["customer_id"] = customers

    df = pd.DataFrame(data_dict)
    df = df.sort_values(by="Time").reset_index(drop=True)

    logger.info(f"Sentetik veri basariyla uretildi. Boyut: {df.shape}")
    return df


def load_or_generate_credit_card_data(
    file_path: Optional[str] = "data/creditcard.csv",
    n_samples: int = 150_000,
    fraud_ratio: float = 0.00172,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Eğer belirtilen dosya yolu mevcutsa ve istenen örnek boyutuyla uyumluysa CSV'den okur;
    aksi halde istenen ölçekte gerçekçi veri seti oluşturup kaydeder.
    """
    if file_path and os.path.exists(file_path):
        logger.info(f"Mevcut veri seti kontrol ediliyor: {file_path}")
        df = pd.read_csv(file_path)
        if "customer_id" not in df.columns:
            # Gerçek Kaggle creditcard.csv (284,807 satır) yüklendiyse customer_id ekle
            rng = np.random.RandomState(random_state)
            n_customers = max(1000, len(df) // 30)
            df["customer_id"] = rng.randint(1001, 1001 + n_customers, size=len(df))
            return df

        # Eğer mevcut sentetik dosyanın satır sayısı istenen n_samples ile aynıysa doğrudan kullan
        if len(df) == n_samples:
            logger.info(f"Mevcut veri seti boyutu ({len(df):,}) istenen boyutla eslesiyor, dosyadan yuklendi.")
            return df
        else:
            logger.info(
                f"Mevcut dosya boyutu ({len(df):,}) istenen boyuttan ({n_samples:,}) farkli. "
                f"Istenen boyutta yeniden uretiliyor..."
            )

    df = generate_synthetic_credit_card_data(
        n_samples=n_samples,
        fraud_ratio=fraud_ratio,
        random_state=random_state
    )

    if file_path:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.info(f"Uretilen veri seti ({len(df):,} satir) suraya kaydedildi: {file_path}")

    return df


def calculate_class_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Sınıf dağılımını, sayılarını ve dengesizlik oranını (Imbalance Ratio) hesaplar.
    """
    total_count = len(df)
    class_counts = df["Class"].value_counts().to_dict()
    n_normal = class_counts.get(0, 0)
    n_fraud = class_counts.get(1, 0)

    ratio_normal = (n_normal / total_count) * 100 if total_count > 0 else 0.0
    ratio_fraud = (n_fraud / total_count) * 100 if total_count > 0 else 0.0
    imbalance_ratio = (n_normal / n_fraud) if n_fraud > 0 else float("inf")

    dist_summary = {
        "total_transactions": int(total_count),
        "normal_transactions": int(n_normal),
        "fraud_transactions": int(n_fraud),
        "normal_ratio_percent": float(np.round(ratio_normal, 4)),
        "fraud_ratio_percent": float(np.round(ratio_fraud, 4)),
        "imbalance_ratio_1_to_k": float(np.round(imbalance_ratio, 2))
    }
    return dist_summary


def calculate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    'Amount' ve 'Time' değişkenlerinin Class 0 (Normal) ve Class 1 (Fraud)
    bazında detaylı özet istatistiklerini üretir.
    """
    stats_list = []
    for cls_val, cls_label in [(0, "Normal"), (1, "Fraud")]:
        subset = df[df["Class"] == cls_val]
        for col in ["Amount", "Time"]:
            series = subset[col]
            stats_list.append({
                "Class": cls_label,
                "Feature": col,
                "Count": int(series.count()),
                "Mean": float(series.mean()),
                "Std": float(series.std()),
                "Min": float(series.min()),
                "Q1_25%": float(series.quantile(0.25)),
                "Median_50%": float(series.median()),
                "Q3_75%": float(series.quantile(0.75)),
                "Max": float(series.max()),
                "Skewness": float(series.skew())
            })

    summary_df = pd.DataFrame(stats_list)
    return summary_df


def run_eda_and_generate_report(
    df: pd.DataFrame,
    output_fig_dir: str = "reports/figures",
    output_metric_dir: str = "reports/metrics"
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Kapsamlı Keşifsel Veri Analizi (EDA) gerçekleştirir ve görselleri kaydeder.
    """
    os.makedirs(output_fig_dir, exist_ok=True)
    os.makedirs(output_metric_dir, exist_ok=True)

    dist = calculate_class_distribution(df)
    stats_df = calculate_summary_statistics(df)

    logger.info("=== SINIF DAGILIMI VE OZET ISTATISTIKLER ===")
    logger.info(f"Toplam Islem: {dist['total_transactions']:,}")
    logger.info(f"Normal Islem: {dist['normal_transactions']:,} (%{dist['normal_ratio_percent']:.3f})")
    logger.info(f"Fraud Islem : {dist['fraud_transactions']:,} (%{dist['fraud_ratio_percent']:.3f})")
    logger.info(f"Dengesizlik Orani (1 Fraud basina Normal islem): 1:{dist['imbalance_ratio_1_to_k']}")

    # 1. Grafik: Sınıf Dağılımı (Lineer vs Logaritmik Karşılaştırma)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    classes = ["Normal (0)", "Fraud (1)"]
    counts = [dist["normal_transactions"], dist["fraud_transactions"]]
    colors = ["#2b5c8f", "#d9534f"]

    axes[0].bar(classes, counts, color=colors, edgecolor="black", alpha=0.85)
    axes[0].set_title("Sinif Sayilari (Lineer Olcek - Dengesizligin Boyutu)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Islem Sayisi")
    for i, count in enumerate(counts):
        axes[0].text(i, count + (max(counts) * 0.01), f"{count:,}", ha="center", fontweight="bold")

    axes[1].bar(classes, counts, color=colors, edgecolor="black", alpha=0.85)
    axes[1].set_yscale("log")
    axes[1].set_title("Sinif Sayilari (Logaritmik Olcek)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Islem Sayisi (Log10)")
    for i, count in enumerate(counts):
        axes[1].text(i, count * 1.3, f"{count:,}", ha="center", fontweight="bold")

    plt.tight_layout()
    dist_plot_path = os.path.join(output_fig_dir, "01_class_distribution.png")
    fig.savefig(dist_plot_path, dpi=300)
    plt.close(fig)

    # 2. Grafik: Amount ve Time Dağılımı (Normal vs Fraud)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    df_plot = df.copy()
    df_plot["Amount_Log"] = np.log1p(df_plot["Amount"])
    df_plot["Class_Label"] = df_plot["Class"].map({0: "Normal", 1: "Fraud"})

    sns.boxplot(data=df_plot, x="Class_Label", y="Amount_Log", hue="Class_Label", ax=axes[0, 0], palette=["#2b5c8f", "#d9534f"], legend=False)
    axes[0, 0].set_title("Islem Tutari Log1p(Amount) Kutu Grafigi", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("log(1 + Amount)")

    sns.kdeplot(data=df_plot[df_plot["Class"] == 0]["Amount_Log"], ax=axes[0, 1], label="Normal", color="#2b5c8f", fill=True, alpha=0.3)
    sns.kdeplot(data=df_plot[df_plot["Class"] == 1]["Amount_Log"], ax=axes[0, 1], label="Fraud", color="#d9534f", fill=True, alpha=0.3)
    axes[0, 1].set_title("Islem Tutari Yogunluk Grafigi (KDE)", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("log(1 + Amount)")
    axes[0, 1].legend()

    df_plot["Hour"] = (df_plot["Time"] / 3600) % 24
    sns.histplot(data=df_plot[df_plot["Class"] == 0], x="Hour", bins=24, ax=axes[1, 0], color="#2b5c8f", stat="density", kde=True)
    axes[1, 0].set_title("Normal Islemlerin Gun Ici Saat Dagilimi", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Gunun Saati (0 - 24)")

    sns.histplot(data=df_plot[df_plot["Class"] == 1], x="Hour", bins=24, ax=axes[1, 1], color="#d9534f", stat="density", kde=True)
    axes[1, 1].set_title("Fraud Islemlerin Gun Ici Saat Dagilimi", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Gunun Saati (0 - 24)")

    plt.tight_layout()
    amount_time_plot_path = os.path.join(output_fig_dir, "02_amount_time_distribution.png")
    fig.savefig(amount_time_plot_path, dpi=300)
    plt.close(fig)

    # 3. Grafik: Fraud ile En Cok Korelasyona Sahip Ozellikler
    numeric_cols = [c for c in df.columns if c not in ["customer_id"]]
    corr_series = df[numeric_cols].corr()["Class"].drop("Class").sort_values()
    top_corr = pd.concat([corr_series.head(5), corr_series.tail(5)])

    fig, ax = plt.subplots(figsize=(10, 6))
    top_corr.plot(kind="barh", ax=ax, color=["#d9534f" if x < 0 else "#2b5c8f" for x in top_corr])
    ax.set_title("Fraud ('Class') ile En Yuksek Negatif ve Pozitif Korelasyonlu Degiskenler", fontsize=12, fontweight="bold")
    ax.set_xlabel("Pearson Korelasyon Katsayisi")
    plt.tight_layout()
    corr_plot_path = os.path.join(output_fig_dir, "03_feature_correlations.png")
    fig.savefig(corr_plot_path, dpi=300)
    plt.close(fig)

    dist_file = os.path.join(output_metric_dir, "eda_class_distribution.json")
    with open(dist_file, "w", encoding="utf-8") as f:
        json.dump(dist, f, indent=4)

    stats_file = os.path.join(output_metric_dir, "eda_summary_statistics.csv")
    stats_df.to_csv(stats_file, index=False)

    logger.info("EDA gorselleri ve metrikleri basariyla olusturuldu.")
    return dist, stats_df


if __name__ == "__main__":
    data = load_or_generate_credit_card_data(n_samples=30_000)
    run_eda_and_generate_report(data)
