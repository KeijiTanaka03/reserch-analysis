import numpy as np
import matplotlib.pyplot as plt

def load_csi_file(file_path):
    """
    ファイルからCSIデータを読み込む（ラベル分けなし）
    
    ファイル内の1レコードは「index, ms, CSIデータ」の形式。
    
    Args:
        file_path (str): 読み込むファイルのパス

    Returns:
        row_data: [[no, ms], ...]
        row_csi: [[サブキャリア値, ...], ...]
    """
    def parse_csi_line(line):
        parts = line.strip().split(',')
        no = int(parts[0])
        ms = int(parts[1])
        csi_hex = parts[2]
        
        def hex_to_signed_int(h):
            val = int(h, 16)
            if val >= 128:
                val -= 256
            return val
        
        csi = [hex_to_signed_int(csi_hex[i:i+2]) for i in range(0, len(csi_hex), 2)]
        return [no, ms], csi

    row_data = []
    row_csi = []

    with open(file_path, 'r') as f:
        for line in f:
            data, csi = parse_csi_line(line)
            row_data.append(data)
            row_csi.append(csi)

    return row_data, row_csi


"""
    row_ampとrow_phaseから振幅と位相を計算する。

    Args:
        row_amp: [ラベル][ファイル][行][サブキャリア値(実部)]
        row_phase: [ラベル][ファイル][行][サブキャリア値(虚部)]

    Returns:
        amplitude: [ラベル][ファイル][行][サブキャリア値(振幅)]
        phase: [ラベル][ファイル][行][サブキャリア値(位相)]

    使用例：
        amplitude, phase = calculate_amplitude_and_phase(row_amp, row_phase)
"""
def calculate_amplitude_and_phase(row_amp, row_phase):
    
    amplitude = []
    phase = []
    for label_amp, label_phase in zip(row_amp, row_phase):
        label_amplitude = []
        label_phase_result = []
        for file_amp, file_phase in zip(label_amp, label_phase):
            file_amplitude = []
            file_phase_result = []
            for amp, phase_val in zip(file_amp, file_phase):
                # 振幅を計算
                amp_val = np.sqrt(np.array(amp)**2 + np.array(phase_val)**2)
                file_amplitude.append(amp_val)
                # 位相を計算
                phase_angle = np.arctan2(phase_val, amp)
                file_phase_result.append(phase_angle)
            label_amplitude.append(file_amplitude)
            label_phase_result.append(file_phase_result)
        amplitude.append(label_amplitude)
        phase.append(label_phase_result)
    return amplitude, phase


"""
	CSIデータをラベルごと・ファイルごとに読み込み
    
	Args:
		label_count (int): ラベル数
		data_folder (str): データフォルダのパス
		file_paths (list): ファイル名リスト（例: ['xxx.txt', ...]）

	Returns:
		row_data, row_csi
 		row_data: [ラベル][ファイル][[no, ms], ...]
		row_csi:  [ラベル][ファイル][[サブキャリア値, ...], ...]
"""
def load_csi_data(label_count, data_folder, file_paths):
    def parse_csi_line(line):
        parts = line.strip().split(',')
        no = int(parts[0])
        ms = int(parts[1])
        csi_hex = parts[2]
        def hex_to_signed_int(h):
            val = int(h, 16)
            if val >= 128:
                val -= 256
            return val
        csi = [hex_to_signed_int(csi_hex[i:i+2]) for i in range(0, len(csi_hex), 2)]
        return {'no': no, 'ms': ms, 'csi': csi}

    row_data = []
    row_csi = []
    for i in range(1, label_count + 1):
        folder_data = []
        folder_csi = []
        folder = f"{data_folder}/{i}"
        for fname in file_paths:
            file_data = []
            fpath = f"{folder}/{fname}"
            with open(fpath) as f:
                for line in f:
                    file_data.append(parse_csi_line(line))
            folder_data.append([[row['no'], row['ms']] for row in file_data])
            folder_csi.append([row['csi'] for row in file_data])
        row_data.append(folder_data)
        row_csi.append(folder_csi)
    return row_data, row_csi

"""
    CSIデータから振幅(row_amp)と位相(row_phase)を分割する

	Args:
		row_csi: [ラベル][ファイル][行][サブキャリア値1,サブキャリア値2,...]

	Returns:
		row_amp: [ラベル][ファイル][行][サブキャリア値1(実部),サブキャリア値2(実部),...]
		row_phase: [ラベル][ファイル][行][サブキャリア値1(虚数部),サブキャリア値2(虚数部),...]
"""
def split_amp_phase(row_csi):

    row_amp = []
    row_phase = []
    for label in row_csi:
        label_amp = []
        label_phase = []
        for file in label:
            file_amp = []
            file_phase = []
            for row in file:
                amp = row[::2]    # 偶数インデックスが振幅
                phase = row[1::2] # 奇数インデックスが位相
                file_amp.append(amp)
                file_phase.append(phase)
            label_amp.append(file_amp)
            label_phase.append(file_phase)
        row_amp.append(label_amp)
        row_phase.append(label_phase)
    return row_amp, row_phase

"""
	csi_dataから全サンプルで0のサブキャリアを削除する

	Args:
		csi_data: [ラベル][ファイル][行][CSI]

	Returns:
		new_csi_data: [ラベル][ファイル][行][CSI]（0サブキャリア除去済み）
"""
def remove_zero_subcarriers(csi_data):

    # すべてのCSIデータを集めてサブキャリアごとにチェック
    all_csi = []
    for label in csi_data:
        for file in label:
            for row in file:
                all_csi.append(row)
    all_csi = np.array(all_csi)  # [全サンプル数, サブキャリア数]

    # サブキャリアごとに全サンプルが0か判定
    zero_mask = np.all(all_csi == 0, axis=0)
    # 0でないサブキャリアのインデックス
    valid_indices = [i for i, is_zero in enumerate(zero_mask) if not is_zero]
    # csi_dataから該当サブキャリアのみ残す
    new_csi_data = []
    for label in csi_data:
        new_label = []
        for file in label:
            new_file = []
            for row in file:
                new_row = [row[i] for i in valid_indices]
                new_file.append(new_row)
            new_label.append(new_file)
        new_csi_data.append(new_label)

    original_subcarriers = all_csi.shape[1] 
    new_subcarriers = len(valid_indices)
    return new_csi_data


"""
	500msごとに分割する関数

	Args:
		row_data: [ラベル][ファイル][[no, ms], ...]
		row_csi:  [ラベル][ファイル][CSI, ...]
		interval_ms: 分割間隔（ミリ秒）

	Returns:
		split_data: [ラベル][ファイル][分割後データ数][データ]
		split_csi:  [ラベル][ファイル][分割後データ数][CSI]
"""
def split_by_interval(row_data, row_csi, interval_ms=500):
    split_data = []
    split_csi = []
    for label_idx, label in enumerate(row_data):
        label_split_data = []
        label_split_csi = []
        for file_idx, file in enumerate(label):
            if not file:
                label_split_data.append([])
                label_split_csi.append([])
                continue
            base_ms = file[0][1]  # 各ファイルの先頭msを基準
            seg_data = []
            seg_csi = []
            segments_data = []
            segments_csi = []
            for idx, (no, ms) in enumerate(file):
                rel_ms = ms - base_ms
                if rel_ms // interval_ms == len(segments_data):
                    seg_data.append([no, ms])
                    seg_csi.append(row_csi[label_idx][file_idx][idx])
                else:
                    if len(seg_data) > 0:
                        segments_data.append(seg_data)
                        segments_csi.append(seg_csi)
                    seg_data = [[no, ms]]
                    seg_csi = [row_csi[label_idx][file_idx][idx]]
            # 最後の端数は追加しない（端数削除）
            label_split_data.append(segments_data)
            label_split_csi.append(segments_csi)
        split_data.append(label_split_data)
        split_csi.append(label_split_csi)
    return split_data, split_csi

"""
データをスライディングウィンドウで切り出す関数

Args:
    row_data: [ラベル][ファイル][[no, ms], ...]
    row_csi:  [ラベル][ファイル][CSI, ...]
    window_ms: ウィンドウサイズ（ミリ秒）
    step_ms: スライド間隔（ミリ秒）

Returns:
    split_data: [ラベル][ファイル][分割後データ数][データ]
    split_csi:  [ラベル][ファイル][分割後データ数][CSI]
"""
def split_by_sliding_window(row_data, row_csi, window_ms=500, step_ms=100):
    split_data = []
    split_csi = []
    for label_idx, label in enumerate(row_data):
        label_split_data = []
        label_split_csi = []
        for file_idx, file in enumerate(label):
            if not file:
                label_split_data.append([])
                label_split_csi.append([])
                continue
            base_ms = file[0][1]  # 各ファイルの先頭msを基準
            segments_data = []
            segments_csi = []
            start_idx = 0
            while start_idx < len(file):
                # 現在のウィンドウの開始時間と終了時間
                start_time = file[start_idx][1]
                end_time = start_time + window_ms

                # ウィンドウ内のデータを収集
                seg_data = []
                seg_csi = []
                for idx in range(start_idx, len(file)):
                    no, ms = file[idx]
                    if ms < end_time:
                        seg_data.append([no, ms])
                        seg_csi.append(row_csi[label_idx][file_idx][idx])
                    else:
                        break

                # ウィンドウが空でなければ追加
                if len(seg_data) > 0:
                    segments_data.append(seg_data)
                    segments_csi.append(seg_csi)

                # 次のウィンドウの開始インデックスを計算
                while start_idx < len(file) and file[start_idx][1] < start_time + step_ms:
                    start_idx += 1

            label_split_data.append(segments_data)
            label_split_csi.append(segments_csi)
        split_data.append(label_split_data)
        split_csi.append(label_split_csi)
    return split_data, split_csi

"""
	各ラベルごとに全ファイルの分割データを連結する

	Args:
		split_data: [ラベル][ファイル][分割後データ数][データ]
		split_csi:  [ラベル][ファイル][分割後データ数][CSI]

	Returns:
		merged_split_data: [ラベル][分割後データ数][データ]
		merged_split_csi:  [ラベル][分割後データ数][CSI]
"""
def merge_split_data(split_data, split_csi):
    
    merged_split_data = []
    merged_split_csi = []
    for label_data, label_csi in zip(split_data, split_csi):
        merged_data = []
        merged_csi = []
        for file_data, file_csi in zip(label_data, label_csi):
            merged_data.extend(file_data)
            merged_csi.extend(file_csi)
        merged_split_data.append(merged_data)
        merged_split_csi.append(merged_csi)
    return merged_split_data, merged_split_csi


"""
	指定したラベルグループごとにCSI分割データをまとめる関数

	Args:
		csi_segments: [ラベル][分割後データ数][CSI]
		label_groups: [[ラベルインデックス群], ...] 例: [[0,1],[3],[5,6],...]
			例: [[0,1],[3],[5,6],...]
			- 各サブリストが新しいラベル（グループ）を表す
			- サブリスト内のインデックスは元データのラベル番号
			- 例えば [[0,1],[3]] なら、元の0番と1番ラベルのデータをまとめて新ラベル0に、
			  3番ラベルのデータを新ラベル1にする

	Returns:
		grouped_csi_segments: [新ラベル][分割後データ数][CSI]
"""
def group_labels_by_indices(csi_segments, label_groups):
    grouped_csi_segments = []
    for group in label_groups:
        group_segments = []
        for label_idx in group:
            group_segments.extend(csi_segments[label_idx])  # 各ラベルの分割データを連結
        grouped_csi_segments.append(group_segments)
    return grouped_csi_segments


"""
split_csi のクラスを指定したラベルグループごとに削減する。

Args:
    split_csi: [ラベル][ファイル][分割後データ数][CSI]
    label_groups: [[ラベルインデックス群], ...]
        - 各サブリストが新しいラベル（グループ）を表す
        - サブリスト内のインデックスは元データのラベル番号
        - 例: [[0,1],[3],[5,6]] の場合、
            元の0番と1番ラベルのデータをまとめて新ラベル0に、
            3番ラベルのデータを新ラベル1にする

Returns:
    reduced_split_csi: [新ラベル][ファイル][分割後データ数][CSI]
"""
def reduce_classes_in_split_csi(split_csi, label_groups):
    reduced_split_csi = []
    for group in label_groups:
        group_csi = []
        for label_idx in group:
            group_csi.extend(split_csi[label_idx])  # 各ラベルのファイルデータを連結
        reduced_split_csi.append(group_csi)
    return reduced_split_csi


"""
    row_ampの特定のサブキャリアをプロットする。

    Args:
        row_amp: [ラベル][ファイル][行][サブキャリア値]
        label_idx: プロットするラベルのインデックス
        file_idx: プロットするファイルのインデックス
        subcarrier_idx: プロットするサブキャリアのインデックス

    使用例：plot_row_amp(row_amp, label_idx=10, file_idx=0, subcarrier_idx=111)
"""
def plot_row_amp(row_amp, label_idx, file_idx, subcarrier_idx):
    # 指定したラベルとファイルのデータを取得
    data = row_amp[label_idx][file_idx]
    
    # 指定したサブキャリアの値を抽出
    subcarrier_values = [row[subcarrier_idx] for row in data]
    
    # プロット
    plt.figure(figsize=(10, 6))
    plt.plot(subcarrier_values, label=f"Label {label_idx}, File {file_idx}, Subcarrier {subcarrier_idx}")
    plt.title(f"Subcarrier {subcarrier_idx} for Label {label_idx}, File {file_idx}")
    plt.xlabel("Sample Index")
    plt.legend()
    plt.grid()
    plt.show()