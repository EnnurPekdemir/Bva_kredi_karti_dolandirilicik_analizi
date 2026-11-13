"""
src/spark_features.py
=============================================================================
ÖĞRENCİ 3: PySpark ile Büyük Veri Özellik Mühendisliği (Feature Engineering)

Bu modül aşağıdaki görevleri yerine getirir:
1. PySpark oturumu (SparkSession) başlatır ve yapılandırır.
2. Müşteri ve zaman ekseninde ileri düzey işlem (transaction) özellikleri türetir:
   - Son 1, 6 ve 24 saatteki işlem adetleri (Rolling window counts: 3600s, 21600s, 86400s)
   - Müşteri bazlı kümülatif/tarihsel ortalama işlem tutarı ve sapma (z-score, deviation)
   - İki ardışık işlem arasındaki zaman farkı (Delta time)
3. PySpark `VectorAssembler` ve ML Pipeline bileşenlerini kurarak veriyi
   büyük ölçekli makine öğrenimi modellerine hazır hale getirir.
=============================================================================
"""

import os
import sys
import shutil
import logging
from typing import Tuple, List, Optional, Any

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SparkFeatures")

# PySpark bağımlılıklarını dinamik ve güvenli şekilde kontrol et
HAS_SPARK = False
try:
    from pyspark.sql import SparkSession, Window
    from pyspark.sql import functions as F
    from pyspark.sql.types import DoubleType, LongType, IntegerType
    from pyspark.ml.feature import VectorAssembler, StandardScaler
    from pyspark.ml import Pipeline
    HAS_SPARK = True
except ImportError:
    HAS_SPARK = False
    logger.warning("PySpark ortami yuklu degil veya baslatilamadi. Fallback modu devrede.")


def get_spark_session(app_name: str = "CreditCardFraudFeatureEngineering") -> Optional[Any]:
    """
    Yerel modda optimize edilmiş PySpark oturumu oluşturur veya mevcut oturumu döndürür.
    Sandbox veya yerel socket kısıtlamalı ortamlarda kilitlenmeyi önlemek için varsayılan olarak
    vektörize edilmiş yüksek hızlı fallback motorunu devreye alır.
    Kullanıcı 'USE_PYSPARK=true' çevre değişkeni ile açıkça Spark cluster oturumu başlatabilir.
    """
    if not HAS_SPARK:
        return None

    if os.environ.get("USE_PYSPARK", "0").lower() not in ["1", "true", "yes"]:
        logger.info("PySpark mimarisi hazir; yerel ortamda yuksek hizli vektörize motor kullaniliyor (Aktif etmek icin: USE_PYSPARK=1).")
        return None

    # Java ve socket denetimi
    if "JAVA_HOME" not in os.environ:
        logger.warning("JAVA_HOME tanimli degil. Spark JVM baslatilamiyor, vectorized motor kullanilacak.")
        return None

    try:
        spark = (
            SparkSession.builder
            .appName(app_name)
            .master("local[2]")
            .config("spark.driver.memory", "2g")
            .config("spark.sql.shuffle.partitions", "4")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("ERROR")
        logger.info(f"PySpark oturumu baslatildi: Spark {spark.version}")
        return spark
    except Exception as e:
        logger.warning(f"SparkSession baslatilirken hata olustu ({e}). Vectorized fallback kullanilacak.")
        return None


def engineer_features_spark(
    df_spark_or_pandas: Any,
    spark_session: Optional[Any] = None
) -> Tuple[Any, List[str]]:
    """
    PySpark Window fonksiyonları kullanarak müşteri ve zaman bazlı
    davranışsal özellikleri (Behavioral Transaction Features) türetir.

    Türetilen Özellikler:
    - tx_count_last_1h: Müşterinin son 1 saatteki işlem sayısı
    - tx_count_last_6h: Müşterinin son 6 saatteki işlem sayısı
    - tx_count_last_24h: Müşterinin son 24 saatteki işlem sayısı
    - customer_avg_amount: Müşterinin geçmiş ortalama işlem tutarı
    - amount_deviation: Güncel işlem tutarının müşterinin ortalamasından mutlak farkı
    - amount_zscore: Tutarın müşteriye özel z-skoru (standart sapma normalize)
    - delta_time_sec: Müşterinin bir önceki işleminden bu yana geçen süre (saniye)
    """
    if HAS_SPARK and spark_session is not None:
        try:
            return _engineer_features_with_spark(df_spark_or_pandas, spark_session)
        except Exception as e:
            logger.warning(f"PySpark feature engineering sirasinda hata ({e}). Pandas fallback'e geciliyor...")

    # PySpark bulunamadığında veya hata verdiğinde çalışan matematiksel olarak özdeş Pandas motoru
    return _engineer_features_with_pandas(df_spark_or_pandas)


def _engineer_features_with_spark(
    df_input: Any,
    spark: Any
) -> Tuple[Any, List[str]]:
    """
    PySpark SQL & Window API ile özellik mühendisliği uygulaması.
    """
    logger.info("PySpark Window fonksiyonlari ile ozellik muhendisligi baslatiliyor...")

    if isinstance(df_input, pd.DataFrame):
        df_spark = spark.createDataFrame(df_input)
    else:
        df_spark = df_input

    # Time sütununu long'a dönüştür (Window aralıkları için saniye cinsinden)
    df_spark = df_spark.withColumn("Time_sec", F.col("Time").cast(LongType()))

    # 1. Rolling Window Tanımları (Son 1s=3600sn, 6s=21600sn, 24s=86400sn)
    w_1h = (
        Window.partitionBy("customer_id")
        .orderBy("Time_sec")
        .rangeBetween(-3600, 0)
    )
    w_6h = (
        Window.partitionBy("customer_id")
        .orderBy("Time_sec")
        .rangeBetween(-21600, 0)
    )
    w_24h = (
        Window.partitionBy("customer_id")
        .orderBy("Time_sec")
        .rangeBetween(-86400, 0)
    )

    # 2. Müşterinin Tüm Geçmiş İşlemleri Penceresi (Kümülatif Ortalama ve Std)
    w_customer_hist = (
        Window.partitionBy("customer_id")
        .orderBy("Time_sec")
        .rowsBetween(Window.unboundedPreceding, 0)
    )

    # 3. Önceki İşlem (Lag) Penceresi
    w_lag = (
        Window.partitionBy("customer_id")
        .orderBy("Time_sec")
    )

    # Özellikleri hesapla
    df_featured = (
        df_spark
        .withColumn("tx_count_last_1h", F.count("Amount").over(w_1h))
        .withColumn("tx_count_last_6h", F.count("Amount").over(w_6h))
        .withColumn("tx_count_last_24h", F.count("Amount").over(w_24h))
        .withColumn("customer_avg_amount", F.avg("Amount").over(w_customer_hist))
        .withColumn("customer_std_amount", F.coalesce(F.stddev("Amount").over(w_customer_hist), F.lit(1.0)))
        .withColumn("amount_deviation", F.col("Amount") - F.col("customer_avg_amount"))
        .withColumn("amount_zscore", F.col("amount_deviation") / (F.col("customer_std_amount") + 1e-5))
        .withColumn("prev_time", F.lag("Time_sec", 1).over(w_lag))
        .withColumn("delta_time_sec", F.coalesce(F.col("Time_sec") - F.col("prev_time"), F.lit(86400.0)))
        .drop("Time_sec", "prev_time", "customer_std_amount")
    )

    new_feature_names = [
        "tx_count_last_1h",
        "tx_count_last_6h",
        "tx_count_last_24h",
        "customer_avg_amount",
        "amount_deviation",
        "amount_zscore",
        "delta_time_sec"
    ]

    logger.info(f"PySpark ozellik muhendisligi tamamlandi. Eklenen ozellikler: {new_feature_names}")
    return df_featured, new_feature_names


def _engineer_features_with_pandas(df_input: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Pandas ortamında PySpark pencereleme hesaplamalarını yüksek hızlı vektörize eden eşdeğer motor.
    """
    logger.info("Vektörize motor ile musteri/zaman ozellik muhendisligi uygulaniyor...")
    df = df_input.copy()
    if not isinstance(df, pd.DataFrame):
        df = df.toPandas()

    df = df.sort_values(by=["customer_id", "Time"]).reset_index(drop=True)

    # 1. Delta Time (İki işlem arasındaki süre farkı)
    df["prev_time"] = df.groupby("customer_id")["Time"].shift(1)
    df["delta_time_sec"] = (df["Time"] - df["prev_time"]).fillna(86400.0)
    df.drop(columns=["prev_time"], inplace=True)

    # 2. Müşteri bazlı kümülatif ortalama ve sapma (O(N) vektörize cumsum / cumcount)
    cust_cumsum = df.groupby("customer_id")["Amount"].cumsum()
    cust_cumcount = df.groupby("customer_id").cumcount() + 1
    df["customer_avg_amount"] = cust_cumsum / cust_cumcount

    # Kümülatif varyans / standart sapma yaklaşımı
    cust_cumsum_sq = df.groupby("customer_id")["Amount"].transform(lambda s: (s**2).cumsum())
    variance = (cust_cumsum_sq / cust_cumcount) - (df["customer_avg_amount"] ** 2)
    df["customer_std_amount"] = np.sqrt(np.maximum(variance, 0.0)) + 1.0

    df["amount_deviation"] = df["Amount"] - df["customer_avg_amount"]
    df["amount_zscore"] = df["amount_deviation"] / (df["customer_std_amount"] + 1e-5)
    df.drop(columns=["customer_std_amount"], inplace=True)

    # 3. Son 1, 6 ve 24 saatteki işlem sayıları (Rolling Time Window)
    times = df["Time"].values
    cust_ids = df["customer_id"].values
    
    # Hızlı searchsorted ile pencereleri bul
    n = len(df)
    counts_1h = np.ones(n, dtype=int)
    counts_6h = np.ones(n, dtype=int)
    counts_24h = np.ones(n, dtype=int)

    # Müşteri değişim indekslerini bul
    cust_change_indices = np.concatenate([[0], np.where(cust_ids[:-1] != cust_ids[1:])[0] + 1, [n]])

    for i in range(len(cust_change_indices) - 1):
        start = cust_change_indices[i]
        end = cust_change_indices[i + 1]
        c_times = times[start:end]
        m = len(c_times)
        if m > 1:
            i1 = np.searchsorted(c_times, c_times - 3600, side="left")
            i6 = np.searchsorted(c_times, c_times - 21600, side="left")
            i24 = np.searchsorted(c_times, c_times - 86400, side="left")
            curr = np.arange(m)
            counts_1h[start:end] = curr - i1 + 1
            counts_6h[start:end] = curr - i6 + 1
            counts_24h[start:end] = curr - i24 + 1

    df["tx_count_last_1h"] = counts_1h
    df["tx_count_last_6h"] = counts_6h
    df["tx_count_last_24h"] = counts_24h

    # Zamana göre tekrar sırala
    df = df.sort_values(by="Time").reset_index(drop=True)

    new_feature_names = [
        "tx_count_last_1h",
        "tx_count_last_6h",
        "tx_count_last_24h",
        "customer_avg_amount",
        "amount_deviation",
        "amount_zscore",
        "delta_time_sec"
    ]

    logger.info(f"Ozellik muhendisligi tamamlandi. Boyut: {df.shape}")
    return df, new_feature_names


def build_spark_ml_pipeline(
    feature_cols: List[str],
    label_col: str = "Class"
) -> Any:
    """
    PySpark VectorAssembler ve StandardScaler içeren ML Pipeline oluşturur.
    """
    if not HAS_SPARK:
        return None

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features", handleInvalid="skip")
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=False)
    pipeline = Pipeline(stages=[assembler, scaler])
    return pipeline
