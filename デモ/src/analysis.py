import sys
import os
sys.path.append(os.path.abspath("../../src"))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

from data_utils import *
from preprocess import *
from feature_extraction import *

# --- 定数 ---
LABEL_COUNT = 18
SAMPLING_RATE = 70
DATA_FOLDER = "../../data/20250809"
DEMO_FILE = "20250809_161644_data.txt"
MODEL_FILE = "trained_model_best.pkl"


def analysis():
    # --- 学習データの読み込み ---
    print("📘 学習データを読み込み中...")
    all_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.txt') and f != DEMO_FILE]
    row_data, row_csi = load_csi_data(LABEL_COUNT, DATA_FOLDER, all_files)
    _, row_phase = split_amp_phase(row_csi)
    split_data, split_csi = split_by_interval(row_data, row_phase, interval_ms=500)
    merged_split_data, merged_split_csi = merge_split_data(split_data, split_csi)

    # クラスの削減（6クラス: 上下 × 左中右）
    target_labels = [[0, 3, 6], [1, 4, 7], [2, 5, 8], [9, 12, 15], [10, 13, 16], [11, 14, 17]]
    restored_split_csi = group_labels_by_indices(merged_split_csi, target_labels)

    # 特徴量抽出（学習データ）
    X_train, y_train, feature_names = create_feature_matrix(restored_split_csi, SAMPLING_RATE)
    print(f"特徴量数: {len(feature_names)}")

    # --- 検証データ（DEMO_FILE）の読み込み ---
    print("📗 検証データを読み込み中...")
    demo_data, demo_csi = load_csi_data(LABEL_COUNT, DATA_FOLDER, [DEMO_FILE])
    _, demo_phase = split_amp_phase(demo_csi)
    demo_split_data, demo_split_csi = split_by_interval(demo_data, demo_phase, interval_ms=500)
    demo_merged_split_data, demo_merged_split_csi = merge_split_data(demo_split_data, demo_split_csi)
    demo_restored_split_csi = group_labels_by_indices(demo_merged_split_csi, target_labels)

    X_demo, y_demo, _ = create_feature_matrix(demo_restored_split_csi, SAMPLING_RATE)
    

    # --- 基本モデル（全特徴量）で一度学習 ---
    print("🔧 基本モデルを学習中...")
    clf_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf_base.fit(X_train, y_train)
    clf_base.fit(X_train, y_train)

    # 特徴量重要度
    importances = clf_base.feature_importances_
    indices = np.argsort(importances)[::-1]

    # --- デバッグモードかどうかで再学習スキップ ---
    if True:
        print("🛠 デバッグモード: 再学習ループはスキップ")
        y_pred = clf_base.predict(X_demo)
        acc = accuracy_score(y_demo, y_pred)
        report = classification_report(y_demo, y_pred, digits=3)
        print(f"精度: {acc:.3f}\n{report}")

        # 保存
        model_info = {
            "model": clf_base,
            "selected_indices": list(range(len(feature_names))),  # 全特徴量
            "feature_names": feature_names,
            "accuracy": acc,
            "report": report
        }
        joblib.dump(model_info, MODEL_FILE)
        print(f"✅ モデルを保存しました → {MODEL_FILE}")
        return

    # --- 上位特徴量を順に増やして再学習＆評価 ---
    print("🔍 上位特徴量での再学習を実施中...")
    max_acc = 0
    best_model = None
    best_top_n = 0
    best_selected_indices = None
    best_report = ""

    for i in range(len(feature_names)):
        top_n = i + 1
        selected_indices = indices[:top_n]

        # 学習・検証データの特徴量選択
        X_train_s = [[x[j] for j in selected_indices] for x in X_train]
        X_demo_s = [[x[j] for j in selected_indices] for x in X_demo]

        # モデル学習
        clf_s = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf_s.fit(X_train_s, y_train)

        # DEMO_FILE で評価
        y_pred = clf_s.predict(X_demo_s)
        acc = accuracy_score(y_demo, y_pred)

        if acc > max_acc:
            max_acc = acc
            best_model = clf_s
            best_top_n = top_n
            best_selected_indices = selected_indices
            best_report = classification_report(y_demo, y_pred, digits=3)

    # --- ベストモデルを保存 ---
    if best_model is not None:
        model_info = {
            "model": best_model,
            "selected_indices": best_selected_indices,
            "feature_names": [feature_names[i] for i in best_selected_indices],
            "accuracy": max_acc,
            "report": best_report
        }
        joblib.dump(model_info, MODEL_FILE)
        print(f"✅ ベストモデルを保存しました ({best_top_n}特徴量, acc={max_acc:.3f}) → {MODEL_FILE}")

    return


if __name__ == "__main__":
    analysis()
