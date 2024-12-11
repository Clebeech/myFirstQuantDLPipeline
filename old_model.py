import torch
import torch.nn as nn
import numpy as np
import h5py
import json
import pandas as pd
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr

# 1. 数据集类
class AlphaNetDataset(Dataset):
    def __init__(self, h5_path, start_date='2013-01-01', end_date='2017-01-01'):
        """
        Args:
            h5_path: h5文件路径
            start_date: 训练起始日期
            end_date: 训练结束日期
        """
        with h5py.File(h5_path, 'r') as f:
            # 读取日期信息
            dates = pd.to_datetime([date.decode() for date in f['sample_infos']['end_date']])
            
            # 根据日期筛选数据
            date_mask = (dates >= start_date) & (dates < end_date)
            valid_indices = np.where(date_mask)[0]
            
            # 加载数据
            features = torch.FloatTensor(f['features'][valid_indices])
            labels = torch.FloatTensor(f['labels'][valid_indices])
            
            # 特征标准化：按batch进行标准化
            batch_means = features.mean(dim=(0, 2), keepdim=True)
            batch_stds = features.std(dim=(0, 2), keepdim=True)
            self.features = (features - batch_means) / (batch_stds + 1e-7)
            
            # 标签处理：去极值
            labels_np = labels.numpy()
            q1, q99 = np.percentile(labels_np, [1, 99])
            labels_np = np.clip(labels_np, q1, q99)
            self.labels = torch.FloatTensor(labels_np)
            
            self.dates = dates[valid_indices]
            
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 2. IC损失函数
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.7, l1_lambda=0.01):
        super().__init__()
        self.alpha = alpha
        self.l1_lambda = l1_lambda
        self.mse = nn.MSELoss()
        
    def forward(self, predictions, targets, epoch=None):
        # 动态调整alpha
        if epoch is not None:
            self.alpha = min(0.9, 0.7 + epoch * 0.01)
            
        # 计算IC损失
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        ic = spearmanr(pred_np, target_np)[0]
        ic_loss = -torch.tensor(ic, device=predictions.device, requires_grad=True)
        
        # 计算MSE损失
        mse_loss = self.mse(predictions, targets)
        
        # 添加L1正则化
        l1_loss = torch.mean(torch.abs(predictions))
        
        # 组合损失
        return self.alpha * ic_loss + (1 - self.alpha) * mse_loss + self.l1_lambda * l1_loss

# 3. 多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        batch_size, seq_len, hidden_size = x.size()
        
        # 线性变换并分头
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 注意力计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        
        # 合并多头
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        output = self.out(context)
        return output

# 4. 增强版GRU模型
class DailyGRUBranch(nn.Module):
    def __init__(self, input_size=6, seq_length=40, hidden_sizes=[64, 128, 256]):
        super().__init__()
        
        # 特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.LayerNorm(hidden_sizes[0]),
            nn.Dropout(0.2),
            nn.Linear(hidden_sizes[0], hidden_sizes[0]),
            nn.ReLU(),
            nn.LayerNorm(hidden_sizes[0]),
            nn.Dropout(0.2)
        )
        
        # 双向GRU层
        self.gru1 = nn.GRU(
            input_size=hidden_sizes[0],
            hidden_size=hidden_sizes[1],
            num_layers=2,
            dropout=0.2,
            bidirectional=True,
            batch_first=True
        )
        
        # 多头注意力层
        self.attention = MultiHeadAttention(hidden_sizes[1] * 2)
        
        # 残差连接的LayerNorm (注意这里的维度是[hidden_sizes[1] * 2])
        self.layer_norm1 = nn.LayerNorm([seq_length, hidden_sizes[1] * 2])
        self.layer_norm2 = nn.LayerNorm([seq_length, hidden_sizes[1] * 4])  # *4是因为我们拼接了global_info
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_sizes[1] * 4, hidden_sizes[2]),  # 输入维度改为*4
            nn.ReLU(),
            nn.LayerNorm(hidden_sizes[2]),
            nn.Dropout(0.2),
            nn.Linear(hidden_sizes[2], hidden_sizes[1]),
            nn.ReLU(),
            nn.LayerNorm(hidden_sizes[1]),
            nn.Dropout(0.1),
            nn.Linear(hidden_sizes[1], 1)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        # x shape: (batch_size, 6, 40)
        x = x.transpose(1, 2)  # -> (batch_size, 40, 6)
        
        # 特征提取
        x = self.feature_extractor(x)  # -> (batch_size, 40, 64)
        
        # GRU处理
        gru_out, _ = self.gru1(x)  # -> (batch_size, 40, 256)
        
        # 残差连接 + 多头注意力
        attended = self.attention(gru_out)
        attended = self.layer_norm1(attended + gru_out)  # 第一个残差连接
        
        # 全局信息
        global_info = torch.mean(attended, dim=1, keepdim=True).expand(-1, attended.size(1), -1)
        attended = torch.cat([attended, global_info], dim=-1)
        attended = self.layer_norm2(attended)  # 第二个残差连接
        
        # 最后一个时间步的输出
        final_state = attended[:, -1, :]
        
        # 全连接层
        return self.fc_layers(final_state)

# 5. 训练函数
def train_epoch(model, dataloader, criterion, optimizer, device, epoch=None):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)
        
        predictions = model(features)
        loss = criterion(predictions.squeeze(), labels, epoch)
        
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
    return total_loss / num_batches

# 6. 评估函数
def evaluate(model, dataloader, device, ic_mode='rank'):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            predictions = model(features)
            all_preds.extend(predictions.cpu().squeeze().numpy())
            all_labels.extend(labels.numpy())
    
    if ic_mode == 'rank':
        ic = spearmanr(all_preds, all_labels)[0]
    else:
        ic = np.corrcoef(all_preds, all_labels)[0, 1]
    return ic