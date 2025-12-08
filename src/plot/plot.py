import numpy as np
import matplotlib.pyplot as plt

# --- サブキャリア選択設定 ---
CSI_VAID_SUBCARRIER_INTERVAL = 1
# 0~65(LLTF), 66~122(HT-LTF), 123~191(STBC-HT-LTF)
# valid_indices = list(range(66, 94, CSI_VAID_SUBCARRIER_INTERVAL)) + \
#                 list(range(95, 123, CSI_VAID_SUBCARRIER_INTERVAL))
valid_indices = list(range(0, 66, CSI_VAID_SUBCARRIER_INTERVAL))
# valid_indices = list(range(66, 123, CSI_VAID_SUBCARRIER_INTERVAL))
# valid_indices = list(range(123, 192, CSI_VAID_SUBCARRIER_INTERVAL))


# --- データ読み込み関数（修正版） ---
def load_csi_from_txt_corrected(file_path, max_records=None):
    amp_all = []
    phase_all = []

    with open(file_path, "r") as f:
        for idx, line in enumerate(f):
            if max_records is not None and idx >= max_records:
                break

            parts = line.strip().split(",")
            if len(parts) < 3:
                continue

            hex_data = parts[2]
            values = [int(hex_data[i:i+2], 16) for i in range(0, len(hex_data), 2)]
            
            # ESP-CSIの仕様に基づき、偶数番目をQ、奇数番目をIとする
            # ただし、8bit符号なし整数(0-255)から符号付き(-128 to 127)に変換する必要がある
            
            # 1. 0-255 の値を -128 to 127 の符号付き整数に変換
            signed_values = [(v - 256) if v > 127 else v for v in values]

            # 2. QとIに分離
            q_values = np.array(signed_values[::2])
            i_values = np.array(signed_values[1::2])

            # 3. 複素数配列を作成
            # np.complex(I, Q) の順であることに注意
            complex_csi = np.array([complex(i, q) for i, q in zip(i_values, q_values)])
            
            # 4. 振幅と位相を計算
            amplitude = np.abs(complex_csi)
            phase = np.angle(complex_csi) # -pi から +pi の範囲で位相を計算

            # 5. 指定サブキャリアだけ抽出
            amp_sel = [amplitude[i] for i in valid_indices if i < len(amplitude)]
            phase_sel = [phase[i] for i in valid_indices if i < len(phase)]
            
            if amp_sel: # データが正しく取れた場合のみ追加
                amp_all.append(amp_sel)
                phase_all.append(phase_sel)

    return np.array(amp_all), np.array(phase_all)

# --- メイン ---
if __name__ == "__main__":
    # 歯磨き(0630)
    file_path = "../../data/20250630/1/20250630_213940_data.txt"
    file_path = "../../data/20250630/7/20250630_213940_data.txt"

    # 歯磨き(0809)
    # file_path = "../../data/20250809/1/20250809_160221_data.txt"
    # file_path = "../../data/20250809/3/20250809_160221_data.txt"

    # 体固定(1007)
    # file_path = "../../data/20251007/1/20251007_083701_data.txt"
    # file_path = "../../data/20251007/3/20251007_083701_data.txt"

    # 体の位置
    # file_path = "../../data/20250922/1/20250922_121949_data.txt"
    # file_path = "../../data/20250922/3/20250922_121949_data.txt"
    
    # 修正版の関数を呼び出す
    csi_amplitude, csi_phase = load_csi_from_txt_corrected(file_path, max_records=150)

    print(f"振幅データ 読み込み完了: shape = {csi_amplitude.shape}")
    print(f"位相データ 読み込み完了: shape = {csi_phase.shape}")

    # 線の色をサブキャリアごとに変えるためのカラーマップ設定
    num_lines = csi_amplitude.shape[1]
    # 'viridis', 'plasma', 'jet' など好きなカラーマップを選ぶ
    cmap = plt.get_cmap('plasma')

    # --- 振幅のプロット ---
    plt.figure(figsize=(12, 6))
    for i in range(num_lines):
        color = cmap(i / (num_lines - 1) if num_lines > 1 else 0.5)
        plt.plot(csi_amplitude[:, i], alpha=0.7, label=f"Sub {valid_indices[i]}", color=color)

    plt.title(f"CSI Amplitude (Corrected) - First {csi_amplitude.shape[0]} Records")
    plt.xlabel("Sample index (time)")
    plt.ylabel("Amplitude")
    plt.ylim(0, 20)
    # plt.ylim(0, 120)
    plt.legend(fontsize=8, ncol=3, loc='upper right')
    plt.tight_layout()
    plt.show()

    # --- 位相のプロット ---
    plt.figure(figsize=(12, 6))
    for i in range(num_lines):
        # 位相はラップ（-piと+piを行き来）するのでギザギザになる
        color = cmap(i / (num_lines - 1) if num_lines > 1 else 0.5)
        plt.plot(csi_phase[:, i], 'o', markersize=2, alpha=0.7, label=f"Sub {valid_indices[i]}", color=color)

    plt.title(f"CSI Phase (Raw) - First {csi_phase.shape[0]} Records")
    plt.xlabel("Sample index (time)")
    plt.ylabel("Phase (radians)")
    plt.ylim(-(np.pi+0.2), (np.pi+0.2)) # Y軸を -pi から +pi に固定
    plt.legend(fontsize=8, ncol=3, loc='upper right')
    plt.tight_layout()
    plt.show()