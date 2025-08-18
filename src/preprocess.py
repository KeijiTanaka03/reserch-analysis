from scipy.signal import butter, filtfilt
from scipy.signal import medfilt
import matplotlib.pyplot as plt


def butter_lowpass_filter_csi(csi_data, sampling_rate, cutoff=10, order=4,
                              label_idx=0, file_idx=0, subc_idx=0):
    """
    CSIデータ全体にバターワースローパスフィルタを適用し、フィルタ前後のプロットも返す
    
    Args:
        csi_data: [ラベル][ファイル][行][CSI]
        sampling_rate: サンプリングレート（Hz）
        cutoff: カットオフ周波数（Hz, デフォルト10Hz）
        order: フィルタ次数（デフォルト4）
        label_idx, file_idx, subc_idx: プロット対象インデックス

    Returns:
        filtered_csi_data: [ラベル][ファイル][行][CSI]（フィルタ済み）
        plt: matplotlib.pyplotオブジェクト（フィルタ前後の波形）
    """
    filtered_csi_data = []
    nyq = 0.5 * sampling_rate
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    for label in csi_data:
        filtered_label = []
        for file in label:
            if len(file) == 0:
                filtered_file = []
            else:
                # CSIデータを転置（[行][サブキャリア]→[サブキャリア][行]）
                csi_array = list(zip(*file))
                # 各サブキャリアごとにバターワースローパスフィルタを適用
                filtered_csi_array = [filtfilt(b, a, subc) for subc in csi_array]
                # 元の形に戻す（[サブキャリア][行]→[行][サブキャリア]）
                filtered_file = [list(x) for x in zip(*filtered_csi_array)]
            filtered_label.append(filtered_file)
        filtered_csi_data.append(filtered_label)
        # フィルタ前後の波形をプロット
    plt_obj = plot_csi_filter_result_csi(
        csi_data, filtered_csi_data, label_idx, file_idx, subc_idx
    )
    return filtered_csi_data, plt_obj



def median_filter_csi(csi_data, kernel_size=3, 
                      label_idx=0, file_idx=0, subc_idx=0):
    """
    CSIデータ全体にメディアンフィルタを適用し、フィルタ前後のプロットも返す

    Args:
        csi_data: [ラベル][ファイル][行][CSI]
        kernel_size: メディアンフィルタの窓幅（デフォルト3）
        label_idx, file_idx, subc_idx: プロット対象インデックス

    Returns:
        filtered_csi_data: [ラベル][ファイル][行][CSI]（フィルタ済み）
        plt: matplotlib.pyplotオブジェクト（フィルタ前後の波形）
    """
    filtered_csi_data = []
    for label in csi_data:
        filtered_label = []
        for file in label:
            if len(file) == 0:
                filtered_file = []
            else:
                # CSIデータを転置（[行][サブキャリア]→[サブキャリア][行]）
                csi_array = list(zip(*file))
                # 各サブキャリアごとにメディアンフィルタを適用
                filtered_csi_array = [medfilt(list(subc), kernel_size=kernel_size) for subc in csi_array]
                # 元の形に戻す（[サブキャリア][行]→[行][サブキャリア]）
                filtered_file = [list(x) for x in zip(*filtered_csi_array)]
            filtered_label.append(filtered_file)
        filtered_csi_data.append(filtered_label)
    # フィルタ前後の波形をプロット
    plt_obj = plot_csi_filter_result_csi(
        csi_data, filtered_csi_data, label_idx, file_idx, subc_idx
    )
    return filtered_csi_data, plt_obj


def plot_csi_filter_result_csi(csi_before, csi_after, label_idx=0, file_idx=0, subc_idx=0):
    """
    csi([ラベル][ファイル][行][CSI])から指定インデックスのフィルタ前後CSI波形をプロット

    Args:
        csi_before: フィルタ前のCSIデータ [ラベル][ファイル][行][CSI]
        csi_after: フィルタ後のCSIデータ [ラベル][ファイル][行][CSI]
        label_idx: ラベル番号
        file_idx: ファイル番号
        subc_idx: サブキャリア番号

    Returns:
        plt（matplotlib.pyplotオブジェクト）
    """
    original_csi = [row[subc_idx] for row in csi_before[label_idx][file_idx]]
    filtered_csi = [row[subc_idx] for row in csi_after[label_idx][file_idx]]
    plt.figure(figsize=(10, 5))
    plt.plot(original_csi, label="Before Filter", alpha=0.6)
    plt.plot(filtered_csi, label="After Filter", linewidth=2)
    plt.title(f"Subcarrier {subc_idx} (Label {label_idx}, File {file_idx})")
    plt.xlabel("Time index (sample)")
    plt.ylabel("CSI Amplitude")
    plt.legend()
    plt.grid(True)
    return plt

