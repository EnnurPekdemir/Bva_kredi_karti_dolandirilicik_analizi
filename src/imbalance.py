"""
src/imbalance.py
=============================================================================
ÖĞRENCİ 2: Sınıf Dengesizliği Çözümleri (Class Imbalance Solutions)

Bu modül aşağıdaki görevleri yerine getirir:
1. Veri setindeki sınıf dengesizliği seviyesini ve oranını analiz eder.
2. Dengesizlik çözüm stratejilerini modüler olarak sunar:
   a) Rastgele Alt Örnekleme (Random Undersampling)
   b) SMOTE (Synthetic Minority Over-sampling Technique)
   c) Algoritmik Dengeleme (Class Weight & Stabilized Scale Pos Weight)
3. Veri Sızıntısını (Data Leakage) önleyen mekanizma:
   - Dengeleme algoritmalarının KESİNLİKLE sadece eğitim (train) verisine
     uygulanmasını, test/doğrulama setinin ise gerçek dünya dağılımını
     yansıtacak şekilde el değmemiş kalmasını garanti eder.
=============================================================================
"""

import os
import logging
from typing import Tuple, Dict, Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

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

        return {
            "total_samples": int(total),
            "majority_count_0": int(n_maj),
            "minority_count_1": int(n_min),
            "minority_percentage": float(np.round(min_pct, 4)),
            "imbalance_ratio_IR": float(np.round(ir, 2)),
            "severity_level": severity
        }


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
        train_idx = set(X_train.index)
        test_idx = set(X_test.index)
        intersection = train_idx.intersection(test_idx)

        if len(intersection) > 0:
            raise ValueError(
                f"[DATA LEAKAGE HATASI] Train ve Test kumesinde {len(intersection)} adet ortak ornek bulundu!"
            )

        if (y_test == 1).sum() == 0:
            logger.warning("[UYARI] Test setinde hic fraud (1) ornegi kalmadi! Stratified split kullanildigindan emin olun.")

        logger.info(
            f"[GUVENLI] Train-Test ayrimi dogrulandi. Sizinti tespit edilmedi. "
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
    """
    method = method.lower()
    logger.info(f"Dengeleme stratejisi uygulaniyor: '{method.upper()}' (Train Boyutu: {len(X_train):,})")

    if method == "none":
        return X_train.copy(), y_train.copy()

    elif method == "undersample":
        if not HAS_IMBLEARN:
            logger.warning("imblearn yuklu degil, ornekleme yapilamadi.")
            return X_train.copy(), y_train.copy()

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
    Scikit-Learn modelleri için sınıf frekansının tersiyle orantılı ağırlıkları hesaplar.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    logger.info(f"Hesaplanan Dengeli Sinif Agirliklari (Class Weights): {weight_dict}")
    return weight_dict


def get_scale_pos_weight(y_train: pd.Series, dampened: bool = True) -> float:
    """
    LightGBM ve XGBoost algoritmaları için 'scale_pos_weight' parametresini hesaplar.
    Not: Uç dengesizlikte (örn. 1:585) ham oranın doğrudan verilmesi GBDT ağaçlarında
    hessian/gradyan patlamasına yol açıp Precision'ı çökerttiği için karekök sönümlemesi
    (sqrt(n_neg / n_pos)) ile stabilize edilir.
    """
    n_neg = float((y_train == 0).sum())
    n_pos = float((y_train == 1).sum())
    if n_pos <= 0:
        return 1.0
    raw_ratio = n_neg / n_pos
    if dampened and raw_ratio > 25.0:
        spw = float(np.sqrt(raw_ratio))
        logger.info(
            f"Ham dengesizlik orani: {raw_ratio:.2f} -> Stabilize edilmis scale_pos_weight (sqrt): {spw:.2f}"
        )
    else:
        spw = float(raw_ratio)
        logger.info(f"Hesaplanan scale_pos_weight degeri: {spw:.2f}")
    return spw
