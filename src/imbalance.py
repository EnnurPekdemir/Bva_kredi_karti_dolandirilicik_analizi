"""
src/imbalance.py
=============================================================================
ÖĞRENCİ 2: Sınıf Dengesizliği Çözümleri (Class Imbalance Solutions)

Bu modül aşağıdaki görevleri yerine getirir:
1. Veri setindeki sınıf dengesizliği seviyesini ve oranını analiz eder.
2. Dengesizlik çözüm stratejilerini modüler olarak sunar:
   a) Rastgele Alt Örnekleme (Random Undersampling)
   b) SMOTE (Synthetic Minority Over-sampling Technique)
   c) Algoritmik Dengeleme (Class Weight & Scale Pos Weight)
3. Veri Sızıntısını (Data Leakage) önleyen mekanizma:
   - Dengeleme algoritmalarının KESİNLİKLE sadece eğitim (train) verisine
     uygulanmasını, test/doğrulama setinin ise gerçek dünya dağılımını
     yansıtacak şekilde el değmemiş kalmasını garanti eder.
=============================================================================
"""

import logging
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

# Pylance ve linter uyumlulugu icin dinamik guvenli import
try:
    from imblearn.under_sampling import RandomUnderSampler  # type: ignore
    from imblearn.over_sampling import SMOTE  # type: ignore
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Imbalance")


class ImbalanceAnalyzer:
    """
    Sınıf dengesizliğinin derecesini, oranını ve teorik zorluğunu analiz eden sınıf.
    """

    @staticmethod
    def analyze(y: pd.Series) -> Dict[str, Any]:
        """
        Etiket serisi üzerinde dengesizlik metriklerini hesaplar.
        """
        counts = y.value_counts().to_dict()
        n_maj = counts.get(0, 0)
        n_min = counts.get(1, 0)
        total = len(y)

        ir = (n_maj / n_min) if n_min > 0 else float("inf")
        min_pct = (n_min / total) * 100 if total > 0 else 0.0

        if ir < 3:
            severity = "Hafif Dengesizlik (Mild)"
        elif ir < 10:
            severity = "Orta Seviye Dengesizlik (Moderate)"
        elif ir < 100:
            severity = "Yuksek Dengesizlik (High)"
        else:
            severity = "Asiri Ucbirim Dengesizlik (Extreme / Rare Event)"

        analysis_result = {
            "total_samples": int(total),
            "majority_count_0": int(n_maj),
            "minority_count_1": int(n_min),
            "minority_percentage": float(np.round(min_pct, 4)),
            "imbalance_ratio_IR": float(np.round(ir, 2)),
            "severity_level": severity
        }
        return analysis_result


class DataLeakageGuard:
    """
    Veri sızıntısını önlemek için güvenlik katmanı.
    Eğitim ve Test kümelerinin ayrımını ve dengeleme işlemlerinin
    yalnızca eğitim kümesine sınırlandırıldığını denetler.
    """

    @staticmethod
    def verify_split_integrity(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> None:
        """
        Train ve Test kümeleri arasında indeks çakışması veya boyut uyumsuzluğu
        olup olmadığını denetler.
        """
        # 1. İndeks kesişimi kontrolü
        train_idx = set(X_train.index)
        test_idx = set(X_test.index)
        intersection = train_idx.intersection(test_idx)

        if len(intersection) > 0:
            raise ValueError(
                f"[DATA LEAKAGE HATASI] Train ve Test kumesinde {len(intersection)} adet ortak ornek bulundu!"
            )

        # 2. Test kümesindeki azınlık sınıfı varlığı kontrolü
        if (y_test == 1).sum() == 0:
            logger.warning("[UYARI] Test setinde hic fraud (1) ornegi kalmadi! Stratified split kullanildigindan emin olun.")

        logger.info(
            f"[GÜVENLİ] Train-Test ayrımı doğrulandı. Sızıntı tespit edilmedi. "
            f"(Train boyutu: {len(X_train):,}, Test boyutu: {len(X_test):,})"
        )


def apply_sampling_strategy(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = "none",
    sampling_strategy: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Eğitim veri setine seçilen dengeleme stratejisini uygular.

    Parametreler:
        X_train (pd.DataFrame): Eğitim özellikleri.
        y_train (pd.Series): Eğitim hedef değişkeni.
        method (str): 'none', 'undersample', 'smote'
        sampling_strategy (float): Hedeflenen azınlık/çoğunluk oranı (örn. 0.1 -> 1:10)
        random_state (int): Tekrarlanabilirlik seed değeri.

    Dönüş:
        Tuple[pd.DataFrame, pd.Series]: Dengelenmiş (veya orijinal) X_train, y_train.
    """
    method = method.lower()
    logger.info(f"Dengeleme stratejisi uygulaniyor: '{method.upper()}' (Train Boyutu: {len(X_train):,})")

    if method == "none":
        return X_train.copy(), y_train.copy()

    elif method == "undersample":
        if not HAS_IMBLEARN:
            logger.warning("imblearn yuklu degil, ornekleme yapilamadi.")
            return X_train.copy(), y_train.copy()

        # Rastgele Çoğunluk Alt Örnekleme (Random Undersampling)
        rus = RandomUnderSampler(
            sampling_strategy=sampling_strategy,
            random_state=random_state
        )
        X_res, y_res = rus.fit_resample(X_train, y_train)
        logger.info(
            f"Random Undersampling tamamlandi: Onceki boyut={len(X_train):,} -> Yeni boyut={len(X_res):,} "
            f"(Fraud: {(y_res == 1).sum():,}, Normal: {(y_res == 0).sum():,})"
        )
        return pd.DataFrame(X_res, columns=list(X_train.columns)), pd.Series(y_res, name=y_train.name)

    elif method == "smote":
        if not HAS_IMBLEARN:
            logger.warning("imblearn yuklu degil, SMOTE uygulanamadi.")
            return X_train.copy(), y_train.copy()

        # SMOTE: Sentetik Azınlık Aşırı Örnekleme
        n_minority = int((y_train == 1).sum())
        k_neighbors = min(5, max(1, n_minority - 1))

        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=random_state
        )
        X_res, y_res = smote.fit_resample(X_train, y_train)
        logger.info(
            f"SMOTE tamamlandi: Onceki boyut={len(X_train):,} -> Yeni boyut={len(X_res):,} "
            f"(Fraud: {(y_res == 1).sum():,}, Normal: {(y_res == 0).sum():,})"
        )
        return pd.DataFrame(X_res, columns=list(X_train.columns)), pd.Series(y_res, name=y_train.name)

    else:
        raise ValueError(f"Bilinmeyen dengeleme stratejisi: '{method}'. Desteklenenler: 'none', 'undersample', 'smote'")


def get_balanced_class_weights(y_train: pd.Series) -> Dict[int, float]:
    """
    Scikit-Learn modelleri (Random Forest, Logistic Regression vb.) için
    sınıf frekansının tersiyle orantılı ağırlıkları hesaplar:
    w_j = n_samples / (n_classes * n_samples_j)
    """
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    logger.info(f"Hesaplanan Dengeli Sinif Agirliklari (Class Weights): {weight_dict}")
    return weight_dict


def get_scale_pos_weight(y_train: pd.Series) -> float:
    """
    LightGBM ve XGBoost algoritmaları için 'scale_pos_weight' parametresi:
    scale_pos_weight = count(negatif_sinif_0) / count(pozitif_sinif_1)
    """
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    if n_pos == 0:
        return 1.0
    spw = float(n_neg / n_pos)
    logger.info(f"Hesaplanan scale_pos_weight degeri (LightGBM/XGBoost icin): {spw:.2f}")
    return spw
