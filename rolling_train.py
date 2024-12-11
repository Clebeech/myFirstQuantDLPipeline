import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from data import get_market_data, prepare_dl_dataset_fast
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
import os
import pickle

class StockDataset(Dataset):
    def __init__(self, X, y):
        # 转置最后两个维度，从 (N, 6, 40) 变为 (N, 40, 6)
        self.X = torch.FloatTensor(X).transpose(1, 2)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, 
                           hidden_size=hidden_size,
                           num_layers=num_layers, 
                           batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output).squeeze()

def prepare_rolling_data(start_date, end_date, train_days=252, test_days=63, batch_size=256):
    """
    准备滚动窗口的训练和测试数据
    train_days: 训练期长度（交易日数量，约一年）
    test_days: 测试期长度（交易日数量，约一季度）
    """
    # 获取原始数据
    data, stock_ids = get_market_data(start_date=start_date, end_date=end_date)
    
    # 获取所有交易日期
    dates = data.index
    
    # 生成滚动窗口的起止日期
    windows = []
    for i in range(0, len(dates) - train_days - test_days + 1, test_days):
        train_start_idx = i
        train_end_idx = i + train_days
        test_end_idx = min(train_end_idx + test_days, len(dates))
        
        # 确保测试期至少有window_size+1天的数据
        if test_end_idx - train_end_idx <= 40:  # 40是window_size
            break
            
        windows.append({
            'train_start': dates[train_start_idx].strftime('%Y-%m-%d'),
            'train_end': dates[train_end_idx-1].strftime('%Y-%m-%d'),
            'test_start': dates[train_end_idx].strftime('%Y-%m-%d'),
            'test_end': dates[test_end_idx-1].strftime('%Y-%m-%d')
        })
    
    return data, stock_ids, windows

def prepare_window_data(data, window, batch_size=256):
    """为单个窗口准备数据"""
    # 分割训练集和测试集
    train_data = data[window['train_start']:window['train_end']]
    test_data = data[window['test_start']:window['test_end']]
    
    # 检查数据长度是否足够
    if len(test_data) <= 40:  # 40是window_size
        raise ValueError(f"测试集长度（{len(test_data)}）小于窗口大小（40），无法生成样本")
    
    # 准备训练集
    X_train, y_train = prepare_dl_dataset_fast(train_data, window_size=40)
    print(f"训练集样本数量: {len(X_train)}")
    print(f"训练集特征维度: X.shape = {X_train.shape}, y.shape = {y_train.shape}")
    
    # 准备测试集
    X_test, y_test = prepare_dl_dataset_fast(test_data, window_size=40)
    print(f"测试集样本数量: {len(X_test)}")
    print(f"测试集特征维度: X.shape = {X_test.shape}, y.shape = {y_test.shape}")
    
    # 移除nan值
    train_mask = ~np.isnan(y_train)
    test_mask = ~np.isnan(y_test)
    
    X_train, y_train = X_train[train_mask], y_train[train_mask]
    X_test, y_test = X_test[test_mask], y_test[test_mask]
    
    # 创建数据加载器
    train_dataset = StockDataset(X_train, y_train)
    test_dataset = StockDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, test_loader

def train_window(model, train_loader, test_loader, window, epochs=10, learning_rate=0.001):
    """训练单个窗口的模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    best_test_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        # 测试阶段
        model.eval()
        total_test_loss = 0
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                total_test_loss += loss.item()
                
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(y_batch.cpu().numpy())
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_test_loss = total_test_loss / len(test_loader)
        
        print(f'Window {window["train_start"]} to {window["test_end"]}, Epoch {epoch+1}/{epochs}:')
        print(f'Training Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')
        
        # 保存最佳模型
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_model_state = model.state_dict().copy()
    
    # 返回最佳模型状态和预测结果
    return best_model_state, best_test_loss, predictions, actuals

def rolling_train(start_date,end_date,train_days=252,test_days=63,batch_size=256,epochs=1):
            
    # 创建保存模型的目录
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('results'):
        os.makedirs('results')
        
    # 准备滚动窗口数据
    data, stock_ids, windows = prepare_rolling_data(start_date, end_date)
        
    # 存储每个窗口的结果
    window_results = []
        
    # 对每个窗口进行训练和测试
    for i, window in enumerate(windows):
        print(f"\nProcessing window: {window['train_start']} to {window['test_end']}")
            
        # 准备当前窗口的数据
        train_loader, test_loader = prepare_window_data(data, window, batch_size)
            
        # 创建新的模型实例
        model = LSTM()
            
        # 训练模型
        best_model_state, test_loss, predictions, actuals = train_window(
            model, train_loader, test_loader, window, epochs)
            
        # 保存模型
        model_path = f"models/model_window_{i}_{window['train_start']}_{window['test_end']}.pth"
        torch.save(best_model_state, model_path)
            
        # 保存预测结果
        result_dict = {
            'window': window,
            'test_loss': test_loss,
            'model_path': model_path,
            'predictions': predictions,
            'actuals': actuals
        }
        window_results.append(result_dict)

    return window_results
    