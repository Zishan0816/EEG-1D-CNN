import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import mne
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import butter, filtfilt
from sklearn.metrics import roc_auc_score

# 配置
DATASET_DIR = '/Users/unionstrong/Desktop/神经网络训练/task_dataset_20250519/dataset'
EXCEL_PATH = '/Users/unionstrong/Desktop/神经网络训练/train_set_sorted_by_date_20250116.xlsx'
MODEL_PATH = '/Users/unionstrong/Desktop/神经网络训练/best_model_seed105.pth'
SAVE_DIR = './channelwise_topomap_outputs'
MAX_EPOCHS = 40
os.makedirs(SAVE_DIR, exist_ok=True)

# 自动读取训练时通道顺序
ALL_CHANNELS_TXT = '/Users/unionstrong/Desktop/神经网络训练/saved_models/all_channels.txt'
if os.path.exists(ALL_CHANNELS_TXT):
    with open(ALL_CHANNELS_TXT, 'r') as f:
        channels = [line.strip() for line in f if line.strip()]
else:
    # 兜底：用当前目录下顺序
    channels = sorted([c for c in os.listdir(DATASET_DIR) if c.startswith('E')])
num_channels = len(channels)
print(f"推理时通道数: {num_channels}")

# 模型结构（与训练保持一致）
class ConsciousnessClassifier(nn.Module):
    def __init__(self, input_dim=400, num_channels=173):
        super().__init__()
        self.channel_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 16, 9, padding=4),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Conv1d(16, 32, 7, padding=3),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(10))
            for _ in range(num_channels)])
        self.combiner = nn.Sequential(
            nn.Linear(32*num_channels*10, 128),
            nn.ReLU(),
            nn.Linear(128, 1))
    def forward(self, x, return_channel_features=False):
        # x: [B, C, L]
        features = [encoder(x[:,i:i+1]) for i, encoder in enumerate(self.channel_encoders)]
        # features: list of [B, 32, 10]
        combined = torch.cat([f.flatten(1) for f in features], dim=1)
        out = self.combiner(combined)
        if return_channel_features:
            # 返回每个通道编码器输出的均值 [B, C]
            channel_means = torch.stack([f.mean(dim=(1,2)) for f in features], dim=1)  # [B, C]
            return out, channel_means
        return out

# 预处理
def filter_signal(data, fs=1000, lowcut=0.5, highcut=45):
    nyq = 0.5 * fs
    b, a = butter(4, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data, axis=1)
def preprocess_signal(data):
    data = filter_signal(data)
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    return (data - mean) / (std + 1e-6)

# 读取fold1验证集
excel = pd.read_excel(EXCEL_PATH)
val_df = excel[excel['Dataset'] == 'fold1'][['ID', 'CRS-R']]

# GSN256标准通道坐标
montage = mne.channels.make_standard_montage("GSN-HydroCel-256")
ch_names = montage.ch_names
pos = np.array([montage.dig[i + 3]['r'][:2] for i in range(len(ch_names))])
name_to_pos = dict(zip(ch_names, pos))

# 加载模型
model = ConsciousnessClassifier(num_channels=num_channels)
ckpt = torch.load(MODEL_PATH, map_location='cpu')
if 'model' in ckpt:
    model.load_state_dict(ckpt['model'])
else:
    model.load_state_dict(ckpt)
model.eval()

# 校验集推理与AUC评估
all_probs, all_labels = [], []
for _, row in tqdm(val_df.iterrows(), total=len(val_df)):
    status = row['CRS-R']
    pid = str(row['ID']).zfill(3)
    tag = f"{status}{pid}"
    label = 0 if status == 'VS' else 1
    # 收集所有通道Binaural数据
    patient_data = []
    for ch in channels:
        ch_path = os.path.join(DATASET_DIR, ch)
        files = [f for f in os.listdir(ch_path) if f.startswith(tag) and f.endswith('_Binaural.npy')]
        if files:
            arr = np.load(os.path.join(ch_path, files[0]))
            arr = preprocess_signal(arr)
            if arr.shape[0] > MAX_EPOCHS:
                arr = arr[:MAX_EPOCHS]
            # 取所有epoch的均值
            mean_epoch = arr.mean(axis=0)  # [400]
            patient_data.append(mean_epoch)
        else:
            # 缺失通道补零
            patient_data.append(np.zeros(400))
    patient_data = np.stack(patient_data)  # [通道数, 400]
    x = torch.tensor(patient_data[None, :, :], dtype=torch.float32)  # [1, C, 400]
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).item()
    all_probs.append(prob)
    all_labels.append(label)

auc = roc_auc_score(all_labels, all_probs)
print(f"[RESULT] fold1 校验集AUC: {auc:.4f}")

# ========== Topomap 可视化部分暂时注释 ==========
for _, row in tqdm(val_df.iterrows(), total=len(val_df)):
    status = row['CRS-R']
    pid = str(row['ID']).zfill(3)
    tag = f"{status}{pid}"
    # 收集所有通道Binaural数据
    patient_data = []
    for ch in channels:
        ch_path = os.path.join(DATASET_DIR, ch)
        files = [f for f in os.listdir(ch_path) if f.startswith(tag) and f.endswith('_Binaural.npy')]
        if files:
            arr = np.load(os.path.join(ch_path, files[0]))
            arr = preprocess_signal(arr)
            if arr.shape[0] > MAX_EPOCHS:
                arr = arr[:MAX_EPOCHS]
            # 取所有epoch的均值
            mean_epoch = arr.mean(axis=0)  # [400]
            patient_data.append(mean_epoch)
        else:
            # 缺失通道补零
            patient_data.append(np.zeros(400))
    patient_data = np.stack(patient_data)  # [通道数, 400]
    x = torch.tensor(patient_data[None, :, :], dtype=torch.float32)  # [1, C, 400]
    with torch.no_grad():
        _, channel_means = model(x, return_channel_features=True)  # [1, C]
    channel_means = channel_means.squeeze(0).numpy()
    # 匹配通道坐标
    values, positions, names = [], [], []
    for ch, val in zip(channels, channel_means):
        if ch in name_to_pos:
            values.append(val)
            positions.append(name_to_pos[ch])
            names.append(ch)
    if not values:
        continue
    values = np.array(values)
    positions = np.array(positions)
    fig, ax = plt.subplots(figsize=(6, 5))
    im, cn = mne.viz.plot_topomap(
        values, positions, axes=ax, show=False,
        contours=0,
        extrapolate='local',
        sphere=None,
        cmap='coolwarm')
    plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.7, pad=0.05, label='Channel Encoder Mean')
    ax.set_title(f"{tag} - {'MCS' if status == 'MCS' else 'VS'}")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{tag}.png"))
    plt.close()
print(f"[INFO] Topomap 已保存至 {SAVE_DIR}") 