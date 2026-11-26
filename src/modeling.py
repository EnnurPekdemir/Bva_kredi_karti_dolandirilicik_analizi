"""
src/modeling.py
=============================================================================
ÖĞRENCİ 4: Model Geliştirme & Kapsamlı Metrik Değerlendirmesi

Bu modül aşağıdaki görevleri yerine getirir:
1. Sektör standardı modelleri eğitir:
   - LightGBM Classifier (stabilize edilmiş scale_pos_weight destekli)
   - XGBoost Classifier (max_delta_step ve scale_pos_weight destekli)
   - Random Forest Classifier (class_weight='balanced_subsample')
   - Naive Baseline Classifier ("Her işleme Normal/0 de")
2. Doğruluk Paradoksunu (Accuracy Paradox) sayısal ve görsel olarak kanıtlar.
3. Finansal dolandırıcılık için hayati metrikleri hesaplar:
   - Precision, Recall, F1-Score, F2-Score, PR-AUC (AUPRC), ROC-AUC
4. Confusion Matrix ve Precision-Recall Eğrisi görselleştirme fonksiyonlarını sunar.
=============================================================================
"""

import os
import logging
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve
)
from sklearn.ensemble import RandomForestClassifier

try:
    import lightgbm as lgb  # type: ignore
except ImportError:
    lgb = None

try:
    import xgboost as xgb  # type: ignore
except ImportError:
    xgb = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Modeling")


class NaiveMajorityClassifier:
    """
    Her işleme istisnasız çoğunluk sınıfı (0 - Normal) tahmininde bulunan
    referans/taban model. Accuracy Paradox'unu ispatlamak için kullanılır.
    """
    def __init__(self):
        self.classes_ = np.array([0, 1])

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        probs = np.zeros((len(X), 2))
        probs[:, 0] = 1.0
        probs[:, 1] = 0.0
        return probs


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Tüm sınıflandırma ve risk değerlendirme metriklerini kapsamlı hesaplar.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)

    if y_prob is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
        except Exception:
            roc_auc = 0.5
        try:
            pr_auc = average_precision_score(y_true, y_prob)
        except Exception:
            pr_auc = float(np.mean(y_true))
    else:
        roc_auc = 0.5
        pr_auc = float(np.mean(y_true))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (cm[0, 0], 0, 0, 0)

    return {
        "model_name": model_name,
        "accuracy": float(np.round(acc, 5)),
        "precision": float(np.round(prec, 5)),
        "recall": float(np.round(rec, 5)),
        "f1_score": float(np.round(f1, 5)),
        "f2_score": float(np.round(f2, 5)),
        "roc_auc": float(np.round(roc_auc, 5)),
        "pr_auc": float(np.round(pr_auc, 5)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    }


def demonstrate_accuracy_paradox(
    y_test: np.ndarray,
    y_pred_naive: np.ndarray,
    y_pred_trained: np.ndarray,
    trained_model_name: str = "XGBoost"
) -> Dict[str, Any]:
    """
    Doğruluk Paradoksunun (Accuracy Paradox) nedenini ve finansal etkisini
    açıkça belgeleyen karşılaştırma raporu sunar.
    """
    naive_metrics = evaluate_predictions(y_test, y_pred_naive, None, model_name="Naive Majority (Always 0)")
    trained_metrics = evaluate_predictions(y_test, y_pred_trained, None, model_name=trained_model_name)

    report = {
        "naive_accuracy": naive_metrics["accuracy"],
        "naive_recall": naive_metrics["recall"],
        "naive_false_negatives": naive_metrics["false_negatives"],
        "trained_accuracy": trained_metrics["accuracy"],
        "trained_recall": trained_metrics["recall"],
        "trained_false_negatives": trained_metrics["false_negatives"],
        "explanation": (
            f"Naive model hicbir fraud islemini yakalayamadigi halde (Recall={naive_metrics['recall']:.2f}, "
            f"Kacirilan Fraud={naive_metrics['false_negatives']}) sozde yuksek dogruluk "
            f"(Accuracy=%{naive_metrics['accuracy']*100:.2f}) gostermektedir. "
            f"Buna karsilik {trained_model_name} modeli, dolandiriciliklarin %{trained_metrics['recall']*100:.1f}'ini "
            f"basariyla yakalamistir. Bu durum finansal dolandiricilikta Accuracy metriginin kesinlikle TEK BASINA "
            f"kullanilamayacagini, PR-AUC ve Recall'un kritik oldugunu ispatlar."
        )
    }

    logger.info("=== ACCURACY PARADOX ANALIZI ===")
    logger.info(f"Naive Model Accuracy: %{naive_metrics['accuracy']*100:.3f} | Recall: %{naive_metrics['recall']*100:.1f} (Kacirilan Fraud: {naive_metrics['false_negatives']})")
    logger.info(f"{trained_model_name} Accuracy : %{trained_metrics['accuracy']*100:.3f} | Recall: %{trained_metrics['recall']*100:.1f} (Kacirilan Fraud: {trained_metrics['false_negatives']})")
    return report


def train_and_evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scale_pos_weight: float = 1.0,
    random_state: int = 42
) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Birden fazla modeli eğitir, metriklerini hesaplar ve tahmin olasılıklarını toplar.
    """
    models: Dict[str, Any] = {
        "Naive Majority": NaiveMajorityClassifier(),
        "Random Forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=10,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=150,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=6,
            max_delta_step=2,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1
        )
    }

    results = []
    probabilities: Dict[str, np.ndarray] = {}

    for name, model in models.items():
        logger.info(f"Model egitiliyor: {name}...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred.astype(float)

        probabilities[name] = y_prob
        met = evaluate_predictions(y_test.values, y_pred, y_prob, model_name=name)
        results.append(met)
        logger.info(f"[{name}] PR-AUC: {met['pr_auc']:.4f} | Recall: {met['recall']:.4f} | Precision: {met['precision']:.4f} | F1: {met['f1_score']:.4f}")

    results_df = pd.DataFrame(results)
    return models, results_df, probabilities


def plot_confusion_matrices(
    y_test: np.ndarray,
    models_dict: Dict[str, Any],
    X_test: pd.DataFrame,
    output_path: str = "reports/figures/04_confusion_matrices.png"
) -> None:
    """
    Modellerin Karmaşıklık Matrislerini (Confusion Matrices) yan yana görselleştirir.
    """
    n_models = len(models_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models_dict.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
            xticklabels=["Normal (0)", "Fraud (1)"],
            yticklabels=["Normal (0)", "Fraud (1)"]
        )
        ax.set_title(f"{name}\nConfusion Matrix", fontsize=11, fontweight="bold")
        ax.set_xlabel("Tahmin Edilen Sinif")
        ax.set_ylabel("Gercek Sinif")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Confusion Matrix grafigi kaydedildi: {output_path}")


def plot_pr_and_roc_curves(
    y_test: np.ndarray,
    probabilities: Dict[str, np.ndarray],
    output_path: str = "reports/figures/05_pr_and_roc_curves.png"
) -> None:
    """
    Tüm modellerin Precision-Recall (PR) ve ROC Eğrilerini yan yana çizer.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    baseline_pr = (y_test == 1).sum() / len(y_test)
    axes[0].plot([0, 1], [baseline_pr, baseline_pr], linestyle="--", color="gray", label=f"Rastgele Taban (AP = {baseline_pr:.4f})")

    for name, y_prob in probabilities.items():
        if name == "Naive Majority":
            continue
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        axes[0].plot(rec, prec, lw=2.2, label=f"{name} (AUPRC = {ap:.4f})")

    axes[0].set_title("Precision-Recall Egrisi (AUPRC - Asil Odak Metrigi)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Recall (Duyarlilik / Yakalama Orani)")
    axes[0].set_ylabel("Precision (Hassasiyet)")
    axes[0].set_ylim([-0.05, 1.05])
    axes[0].set_xlim([-0.05, 1.05])
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower left")

    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Rastgele (AUC = 0.5000)")
    for name, y_prob in probabilities.items():
        if name == "Naive Majority":
            continue
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        axes[1].plot(fpr, tpr, lw=2.2, label=f"{name} (ROC-AUC = {auc:.4f})")

    axes[1].set_title("ROC Egrisi (Receiver Operating Characteristic)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("False Positive Rate (1 - Ozgulluk)")
    axes[1].set_ylabel("True Positive Rate (Recall)")
    axes[1].set_ylim([-0.05, 1.05])
    axes[1].set_xlim([-0.05, 1.05])
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"PR ve ROC egrileri grafigi kaydedildi: {output_path}")
