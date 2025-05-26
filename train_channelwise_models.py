import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
from scipy.signal import butter, filtfilt, iirnotch
import matplotlib.pyplot as plt
from tqdm import tqdm

# 配置参数
class Config:
    data_dir = '/Users/unionstrong/Desktop/神经网络训练/task_dataset_20250519/dataset'
    excel_path = '/Users/unionstrong/Desktop/神经网络训练/train_set_sorted_by_date_20250116.xlsx'
    save_dir = './saved_models'
    batch_size = 32
    epochs = 40
    lr = 3e-4
    seed = 105
    test_size = 0.2  # 验证集比例
    max_epochs_per_patient = 45
    sample_epochs_per_patient = 30

# 数据加载器（修复数据泄露）
class EEGDataLoader:
    def __init__(self, patient_keys):  # patient_keys: List[(status, id)]
        self.patient_keys = patient_keys
        self.all_channels = sorted([d for d in os.listdir(Config.data_dir) if d.startswith('E')])
        self.num_channels = len(self.all_channels)

    def load_data(self):
        df = pd.read_excel(Config.excel_path)
        patient_data = []
        patient_labels = []
        for status, pid in self.patient_keys:
            row = df[(df['ID'] == pid) & (df['CRS-R'] == status)].iloc[0]
            status_pid = f"{status}{int(pid):03d}"
            label = 0 if status == 'VS' else 1
            patient_epochs = []
            for channel in self.all_channels:
                try:
                    files = [f for f in os.listdir(f"{Config.data_dir}/{channel}") 
                            if f.startswith(status_pid + '_') and f.endswith('_Binaural.npy')]
                    if files:
                        data = np.load(f"{Config.data_dir}/{channel}/{files[0]}")
                        data = self.preprocess_signal(data)
                        if data.shape[0] > 40:
                            idx = np.random.choice(data.shape[0], 40, replace=False)
                            data = data[idx]
                        elif data.shape[0] < 40:
                            pad = np.zeros((40 - data.shape[0], 400))
                            data = np.concatenate([data, pad], axis=0)
                        patient_epochs.append(data)
                    else:
                        print(f"警告: {channel}下未找到{status_pid}相关文件，补零")
                        patient_epochs.append(np.zeros((40, 400)))
                except Exception as e:
                    print(f"警告: {channel}下{status_pid}加载异常: {e}，补零")
                    patient_epochs.append(np.zeros((40, 400)))
            try:
                patient_epochs = np.stack(patient_epochs)  # [C, 40, 400]
                patient_epochs = patient_epochs.transpose(1, 0, 2)  # [40, C, 400]
                np.random.seed(Config.seed)
                selected = np.random.choice(patient_epochs.shape[0], Config.sample_epochs_per_patient, replace=False)
                patient_data.append(patient_epochs[selected])
                patient_labels.extend([label] * Config.sample_epochs_per_patient)
                print(f"患者{status_pid}：采样{Config.sample_epochs_per_patient}个epoch，标签{label}")
            except Exception as e:
                print(f"患者{status_pid}数据堆叠异常: {e}，补零{Config.sample_epochs_per_patient}个epoch")
                patient_data.append(np.zeros((Config.sample_epochs_per_patient, self.num_channels, 400)))
                patient_labels.extend([label] * Config.sample_epochs_per_patient)
        if not patient_data:
            raise ValueError("No valid data loaded!")
        all_data = np.concatenate(patient_data, axis=0)
        all_labels = np.array(patient_labels)
        indices = np.arange(len(all_data))
        np.random.shuffle(indices)
        # 数据分布统计
        print("\n数据分布统计:")
        print(f"总样本数: {len(all_labels)}")
        print(f"VS (0): {sum(all_labels==0)} | MCS (1): {sum(all_labels==1)}")
        if sum(all_labels==0) > 0:
            print(f"正负样本比例: {sum(all_labels==1)/sum(all_labels==0):.2f}")
        else:
            print("正负样本比例: Inf")
        return all_data[indices], all_labels[indices]

    def preprocess_signal(self, data):
        # 带通滤波 + 50Hz陷波
        b, a = butter(2, [0.5/250, 45/250], btype='band')
        filtered = filtfilt(b, a, data, axis=1)
        b_notch, a_notch = iirnotch(50, 30, fs=1000)
        filtered = filtfilt(b_notch, a_notch, filtered, axis=1)
        # 逐样本标准化
        normed = (filtered - np.mean(filtered, axis=1, keepdims=True)) / (np.std(filtered, axis=1, keepdims=True) + 1e-6)

        return normed

# 共享权重的轻量模型
class EEGClassifier(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.net(x).squeeze(1)

# 训练函数
def train():
    # 初始化
    torch.manual_seed(Config.seed)
    np.random.seed(Config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(Config.save_dir, exist_ok=True)
    
    # 数据加载
    df = pd.read_excel(Config.excel_path)
    # 训练集和验证集用(CRS-R, ID)唯一标识
    train_rows = df[df['Dataset'].isin(['fold0','fold2','fold3','fold4'])][['CRS-R','ID']]
    val_rows = df[df['Dataset']=='fold1'][['CRS-R','ID']]
    train_keys = [(row['CRS-R'], int(row['ID'])) for _, row in train_rows.iterrows()]
    val_keys = [(row['CRS-R'], int(row['ID'])) for _, row in val_rows.iterrows()]
    print(f"训练样本数: {len(train_keys)}, 验证样本数: {len(val_keys)}")
    print("训练集样本示例:", train_keys[:5])
    print("验证集样本示例:", val_keys[:5])
    print("是否有重叠:", set(train_keys) & set(val_keys))
    train_loader = EEGDataLoader(train_keys)
    val_loader = EEGDataLoader(val_keys)
    num_channels = train_loader.num_channels
    X_train, y_train = train_loader.load_data()
    X_val, y_val = val_loader.load_data()
    
    # 输出验证集ID与标签分布，辅助排查
    print('\n验证集fold1详细分布:')
    val_df = df[df['Dataset']=='fold1'][['CRS-R','ID']]
    for _, row in val_df.iterrows():
        status = row['CRS-R']
        pid = int(row['ID'])
        status_pid = f"{status}{pid:03d}"
        print(f"{status_pid}")
    print('统计:', val_df['CRS-R'].value_counts())
    
    # 处理类别不平衡
    pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # 模型和优化器
    model = EEGClassifier(num_channels=train_loader.num_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)
    
    # 数据加载器
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size)
    
    # 训练循环
    best_val_auc = 0
    for epoch in range(1, Config.epochs + 1):
        model.train()
        train_preds, train_trues = [], []
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}"):
            xb, yb = xb.to(device), yb.to(device).float()
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_preds.extend(torch.sigmoid(out).detach().cpu().numpy())
            train_trues.extend(yb.cpu().numpy())
        
        # 验证集评估
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device).float()
                out = model(xb)
                val_preds.extend(torch.sigmoid(out).cpu().numpy())
                val_trues.extend(yb.cpu().numpy())
        
        # 计算指标
        train_auc = roc_auc_score(train_trues, train_preds)
        val_auc = roc_auc_score(val_trues, val_preds)
        print(f"Epoch {epoch} | Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}")
        
        # 保存最佳模型
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_auc': val_auc,
                'num_channels': num_channels
            }, os.path.join(Config.save_dir, 'best_model.pth'))
            print(f"Saved new best model with Val AUC: {val_auc:.4f}")

if __name__ == "__main__":
    train()