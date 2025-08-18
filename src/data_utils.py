import numpy as np

def load_csi_data(label_count, data_folder, file_paths):
    """
    CSIデータをラベルごと・ファイルごとに読み込み
    
    Args:
        label_count (int): ラベル数
        data_folder (str): データフォルダのパス
        file_paths (list): ファイル名リスト（例: ['xxx.txt', ...]）

    Returns:
        row_data, row_csi
        row_data: [ラベル][ファイル][[no, ms], ...]
        row_csi:  [ラベル][ファイル][CSI, ...]
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


def split_amp_phase(row_csi):
    """
    CSIデータから振幅(row_amp)と位相(row_phase)を分割する

    Args:
        row_csi: [ラベル][ファイル][行][CSI]

    Returns:
        row_amp: [ラベル][ファイル][行][振幅]
        row_phase: [ラベル][ファイル][行][位相]
    """
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


def remove_zero_subcarriers(csi_data):
    """
    csi_dataから全サンプルで0のサブキャリアを削除する

    Args:
        csi_data: [ラベル][ファイル][行][CSI]

    Returns:
        new_csi_data: [ラベル][ファイル][行][CSI]（0サブキャリア除去済み）
    """
    
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
    return new_csi_data, original_subcarriers, new_subcarriers


def split_by_interval(row_data, row_csi, interval_ms=500):
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


def merge_split_data(split_data, split_csi):
    """
    各ラベルごとに全ファイルの分割データを連結する

    Args:
        split_data: [ラベル][ファイル][分割後データ数][データ]
        split_csi:  [ラベル][ファイル][分割後データ数][CSI]

    Returns:
        merged_split_data: [ラベル][分割後データ数][データ]
        merged_split_csi:  [ラベル][分割後データ数][CSI]
    """
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


def group_labels_by_indices(csi_segments, label_groups):
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
    grouped_csi_segments = []
    for group in label_groups:
        group_segments = []
        for label_idx in group:
            group_segments.extend(csi_segments[label_idx])  # 各ラベルの分割データを連結
        grouped_csi_segments.append(group_segments)
    return grouped_csi_segments
