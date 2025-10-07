import cv2
from PIL import ImageFont, ImageDraw, Image

import sys
import os
sys.path.append(os.path.abspath("../../src"))

from analysis import *
from data_utils import *
from preprocess import *
from feature_extraction import *

# -------------------- グローバル初期化 --------------------
import joblib
import numpy as np

# モデルと使用特徴量の読み込み
model_info = joblib.load(MODEL_FILE)
clf = model_info["model"]
feature_indices = model_info["selected_indices"]

# デモデータの読み込み
demo_data, demo_csi = load_csi_file("../../data/20250809/" + DEMO_FILE)
# window_csi から振幅と位相を分離
amp_window = []
phase_window = []

for sample in demo_csi:
        amp = sample[::2]    # 偶数インデックスが振幅
        phase = sample[1::2] # 奇数インデックスが位相
        amp_window.append(amp)
        phase_window.append(phase)

# 動画開始時間（ms）
video_start_ms = 973550

def put_japanese_text(frame, text, position, color=(255,0,0)):
    """
    frame: OpenCVフレーム
    text: 表示する文字列（日本語可）
    position: (x, y) 左上座標
    color: 文字色 BGR
    """
    font_size = 48

    # OpenCV -> PIL
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)
    font = ImageFont.truetype("/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc", font_size)
    # 文字描画（縁取り付き）
    draw.text(position, text, font=font, fill=color)
    # PIL -> OpenCV
    return cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

def highlight_tooth_area(tooth_img, label_index):
    """
    歯画像上で磨いている箇所を半透明で強調表示
    """
    label = [(395, 390), (280, 470), (165, 390),
             (395, 170), (280, 70), (165, 170)]

    alpha = 0.7
    radius = 80
    overlay = tooth_img.copy()
    cv2.circle(overlay, label[label_index], radius, (0, 255, 0), -1)
    highlighted = cv2.addWeighted(tooth_img, 1-alpha, overlay, alpha, 0)
    return highlighted

def get_correct_label(current_frame, movie_start, label_intervals):
    """
    現在のフレーム番号に対応する正解ラベル番号を返す関数

    Parameters
    ----------
    current_frame : int
        現在の動画フレーム番号
    movie_start : int
        データ開始オフセット（動画の開始フレーム）
    label_intervals : list[list[int, int]]
        各ラベルの開始・終了フレーム区間

    Returns
    -------
    label_index : int | None
        対応するラベル番号（1始まり）。該当なしなら None
    """
    adjusted_frame = current_frame + movie_start
    for i, (start, end) in enumerate(label_intervals, start=1):
        if start <= adjusted_frame <= end:
            return i
    return None

# 18分類ラベルから6分類ラベルに変更
def map_label_to_index(label_num):
    """
    ラベル番号 (1..18) を歯インデックス (0..5) に変換する。
    - 1..9  -> (n-1) % 3        -> 0,1,2 のいずれか
    - 10..18 -> (n-1) % 3 + 3   -> 3,4,5 のいずれか
    該当しない / None の場合は None を返す。
    """
    if label_num is None:
        return None
    n = int(label_num)
    if 1 <= n <= 9:
        return (n - 1) % 3
    if 10 <= n <= 18:
        return (n - 1) % 3 + 3
    return None

def predict_label(current_frame):
    """
    現在のフレームに対応するCSI位相データからラベルを予測
    """
    frame_ms = video_start_ms + int(current_frame * (1000 / SAMPLING_RATE))

    # ±250ms の範囲の CSI をまとめる（位相のみ）
    window_phase = []
    for csi, data in zip(phase_window, demo_data):
        ms = data[1]  # demo_data から timestamp を取得
        if frame_ms - 250 <= ms <= frame_ms + 250:
            window_phase.append(csi)

    if not window_phase:
        return None

    # 特徴量作成（サブキャリアごとの統計量を保持）
    X_window, feature_names = create_feature_matrix_for_window(window_phase)

    # 予測
    pred = clf.predict(X_window)
    return int(pred[0])


def create_feature_matrix_for_window(csi_segments):
    """
    サブキャリアごとに13種類の特徴量を抽出し、特徴量行列を作成する関数。

    Args:
        csi_segments: [サンプル数][サブキャリア数] の2次元リスト

    Returns:
        X: 特徴量行列 [1][サブキャリア数 × 特徴量数]
        feature_names: 特徴量名リスト
    """
    import numpy as np
    from scipy.stats import kurtosis, skew

    # 特徴量関数を定義
    feature_funcs = {
        "平均": lambda x: float(np.mean(x)),
        "中央値": lambda x: float(np.median(x)),
        "標準偏差": lambda x: float(np.std(x)),
        "最大値": lambda x: float(np.max(x)),
        "最小値": lambda x: float(np.min(x)),
        "上四分位値": lambda x: float(np.percentile(x, 75)),
        "下四分位値": lambda x: float(np.percentile(x, 25)),
        "四分位範囲": lambda x: float(np.percentile(x, 75) - np.percentile(x, 25)),
        "平均絶対偏差": lambda x: float(np.mean(np.abs(x - np.mean(x)))),
        "尖度": lambda x: float(kurtosis(x)) if len(x) > 3 else None,
        "歪度": lambda x: float(skew(x)) if len(x) > 2 else None,
        "自己相関": lambda x: float(np.corrcoef(x[:-1], x[1:])[0, 1]) if len(x) > 1 else None,
        "二乗平均平方根": lambda x: float(np.sqrt(np.mean(np.array(x) ** 2))),  # 修正箇所
    }

    # サンプル数とサブキャリア数を取得
    num_samples = len(csi_segments)
    num_subcarriers = len(csi_segments[0]) if num_samples > 0 else 0

    # 特徴量行列と特徴量名リスト
    X = []
    feature_names = []

    # サブキャリアごとに特徴量を計算
    for subcarrier_idx in range(num_subcarriers):
        # サブキャリアごとのデータを抽出
        subcarrier_data = [csi_segments[sample_idx][subcarrier_idx] for sample_idx in range(num_samples)]

        # 特徴量を計算
        features = [func(subcarrier_data) for func in feature_funcs.values()]
        X.extend(features)

        # 特徴量名を作成（最初のサブキャリアのみ）
        if not feature_names:
            for feature_name in feature_funcs.keys():
                feature_names.append(f"subc{subcarrier_idx}_{feature_name}")

    return [X], feature_names


def main():
    # --- 正解ラベル情報 ---
    label_intervals = [
        [156, 356], [372, 551], [598, 770],
        [801, 990], [1013, 1198], [1225, 1406],
        [1451, 1604], [1630, 1803], [1831, 2011],
        [2054, 2248], [2271, 2447], [2503, 2656],
        [2683, 2860], [2883, 3052], [3081, 3261],
        [3290, 3514], [3535, 3741], [3767, 3950],
    ]
    movie_start = 74

    # 動画の読み込み
    video_path = "../resource/デモ用動画.mov"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("動画を開けませんでした。")
        return
    
    
    # 歯のイラスト画像
    tooth_image_path = "../resource/歯（加工）.jpeg"
    tooth_img = cv2.imread(tooth_image_path, cv2.IMREAD_UNCHANGED)  # アルファチャンネルもある場合
    if tooth_img is None:
        print("歯の画像を読み込めませんでした。")
        return
    
    # オーバーレイサイズ調整（1枚あたり）
    overlay_size = (550, 540)  # 適宜調整
    tooth_img = cv2.resize(tooth_img, overlay_size)

    label_num = None
    predicted_label = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 磨いている箇所に色付け（正解）
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        label_num = get_correct_label(current_frame, movie_start, label_intervals)

        if label_num is not None:
            tooth_index_correct = map_label_to_index(label_num)
            tooth_img1 = highlight_tooth_area(tooth_img, tooth_index_correct)
        else:
            tooth_img1 = tooth_img.copy()


        # 磨いている箇所に色付け（推測）
        if current_frame % 30 == 0:
            predicted_label = predict_label(current_frame)

        if predicted_label is not None:
            tooth_img2 = highlight_tooth_area(tooth_img, predicted_label)
        else:
            tooth_img2 = tooth_img.copy()


        # 右上に1枚目
        x_offset = frame.shape[1] - overlay_size[0]
        y_offset = 0
        frame[y_offset:y_offset+overlay_size[1], x_offset:x_offset+overlay_size[0]] = tooth_img1

        # 右上に2枚目（縦に並べる）
        y_offset2 = y_offset + overlay_size[1] # 上の画像の下に20px空ける
        frame[y_offset2:y_offset2+overlay_size[1], x_offset:x_offset+overlay_size[0]] = tooth_img2

        # 左上に日本語ラベル
        frame = put_japanese_text(frame, "正解", (x_offset + 10, y_offset + 10), color=(255,0,0))
        frame = put_japanese_text(frame, "推測", (x_offset + 10, y_offset2 + 10), color=(0,0,255))

        # フレーム表示
        cv2.imshow("でも", frame)

        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()