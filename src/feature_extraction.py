import numpy as np
from scipy.stats import kurtosis, skew
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks


def create_feature_matrix(csi_segments, sampling_rate, use_additional=True):
    """
    CSI分割データから特徴量行列X, ラベルy, 特徴量名feature_namesを作成する関数

    Args:
        csi_segments: [ラベル][分割後データ数][CSI]
        sampling_rate: サンプリングレート
        use_additional: Trueなら追加特徴量も含める

    Returns:
        X: 特徴量行列
        y: ラベルリスト
        feature_names: 特徴量名リスト
    """
    X = []
    y = []
    feature_names = []
    for label_idx, label_csi in enumerate(csi_segments):
        for segment in label_csi:
            features = []
            if use_additional:
                stat_features_list = extract_features_per_subcarrier(segment)
                add_features_list = extract_additional_features_per_subcarrier(segment, sampling_rate)
                if not feature_names:
                    for subc_idx, (stat_features, add_features) in enumerate(zip(stat_features_list, add_features_list)):
                        for key in stat_features.keys():
                            feature_names.append(f"subc{subc_idx}_{key}")
                        for key in add_features.keys():
                            feature_names.append(f"subc{subc_idx}_{key}")
                for stat_features, add_features in zip(stat_features_list, add_features_list):
                    features.extend(stat_features.values())
                    features.extend(add_features.values())
            else:
                stat_features_list = extract_features_per_subcarrier(segment)
                if not feature_names:
                    for subc_idx, stat_features in enumerate(stat_features_list):
                        for key in stat_features.keys():
                            feature_names.append(f"subc{subc_idx}_{key}")
                for stat_features in stat_features_list:
                    features.extend(stat_features.values())
            X.append(features)
            y.append(label_idx)
    return X, y, feature_names


def extract_features_per_subcarrier(segment_csi):
    """
    各サブキャリアごとに統計的特徴量（13種類）を抽出する関数

    Args:
        segment_csi: [サンプル][サブキャリア] の2次元配列（1分割区間のCSIデータ）

    Returns:
        features_per_subc: [サブキャリア][特徴量dict]
            各サブキャリアごとに特徴量を含むdictのリスト
    """
    segment_csi = np.array(segment_csi)
    features_per_subc = []
    for subc in segment_csi.T:
        features = {
            "平均": float(np.mean(subc)),
            "中央値": float(np.median(subc)),
            "標準偏差": float(np.std(subc)),
            "最大値": float(np.max(subc)),
            "最小値": float(np.min(subc)),
            "上四分位値": float(np.percentile(subc, 75)),
            "下四分位値": float(np.percentile(subc, 25)),
            "四分位範囲": float(np.percentile(subc, 75) - np.percentile(subc, 25)),
            "平均絶対偏差": float(np.mean(np.abs(subc - np.mean(subc)))),
            "尖度": float(kurtosis(subc)) if len(subc) > 3 else None,
            "歪度": float(skew(subc)) if len(subc) > 2 else None,
            "自己相関": float(np.corrcoef(subc[:-1], subc[1:])[0, 1]) if len(subc) > 1 else None,
            "二乗平均平方根": float(np.sqrt(np.mean(subc ** 2)))
        }
        features_per_subc.append(features)
    return features_per_subc


def extract_additional_features_per_subcarrier(segment_csi, sampling_rate):
    """
    各サブキャリアごとに特徴量（周波数・時間変化・周期性）を抽出する関数

    Args:
        segment_csi: [サンプル][サブキャリア] の2次元配列（1分割区間のCSIデータ）
        sampling_rate: サンプリングレート（デフォルト: 70）

    Returns:
        features_per_subc: [サブキャリア][特徴量dict]
            各サブキャリアごとに以下の特徴量を含むdictのリスト
            - 主成分周波数
            - パワースペクトル合計
            - バンドパワー_1-10Hz
            - 差分標準偏差
            - 移動平均_5点
            - ピーク数
            - ピーク間隔平均
    """
    segment_csi = np.array(segment_csi)
    features_per_subc = []
    for subc in segment_csi.T:
        features = {}
        # DC成分除去
        subc_detrended = subc - np.mean(subc)
        fft_vals = np.abs(rfft(subc_detrended))
        fft_freqs = rfftfreq(len(subc), d=1/sampling_rate)
        # 0Hz以外の最大成分
        if len(fft_vals) > 1:
            main_freq_idx = np.argmax(fft_vals[1:]) + 1
            features['主成分周波数'] = fft_freqs[main_freq_idx]
        else:
            features['主成分周波数'] = 0
        # パワースペクトルの合計
        features['パワースペクトル合計'] = np.sum(fft_vals)
        # バンドパワー（例: 1-10Hz）
        features['バンドパワー_1-10Hz'] = np.sum(fft_vals[(fft_freqs >= 1) & (fft_freqs <= 10)])
        # 差分の標準偏差
        features['差分標準偏差'] = np.std(np.diff(subc))
        # 移動平均（5点）
        features['移動平均_5点'] = np.mean(np.convolve(subc, np.ones(5)/5, mode='valid'))
        # ピーク検出（周期性の指標）
        peaks, _ = find_peaks(subc, distance=5)
        features['ピーク数'] = len(peaks)
        # ピーク間隔の平均
        if len(peaks) > 1:
            peak_intervals = np.diff(peaks)
            features['ピーク間隔平均'] = np.mean(peak_intervals)
        else:
            features['ピーク間隔平均'] = None
        features_per_subc.append(features)
    return features_per_subc