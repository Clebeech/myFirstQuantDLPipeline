from qdb.factor.factor_mgr import FactorMgr
from ml_lib import get_stk_pool
from tqdm import tqdm #不要直接导入
import pandas as pd
import numpy as np

fm = FactorMgr()

def get_market_data(start_date='2020-01-01', end_date='2020-12-31', stk_pool='all'):
    """获取量价数据并进行后复权处理"""
    fm = FactorMgr()
    stock_ids = get_stk_pool(stk_pool)
    
    # 获取复权因子
    cumadj = fm.get('cumadj')
    cumadj = cumadj.loc[start_date:end_date]
    cumadj = cumadj[stock_ids]
    
    # 获取需要复权的价格类数据
    fields = ['open', 'high', 'low', 'close', 'vwap', 'vol']
    price_fields = ['open', 'high', 'low', 'close', 'vwap']
    
    data = {}
    for field in tqdm(fields, desc="获取字段数据"):
        data[field] = fm.get(field)
        data[field] = data[field].loc[start_date:end_date]
        data[field] = data[field][stock_ids]
        
        # 对价格类数据进行后复权处理
        if field in price_fields:
            data[field] = data[field].multiply(cumadj)
            
        # 前向填充后再后向填充
        data[field] = data[field].fillna(method='ffill').fillna(method='bfill')
        
    #label
    data['label'] = (data['vwap'].shift(-2)-data['vwap'].shift(-1))/data['vwap'].shift(-1)
    
    # 找出所有字段中完全为空的股票，这些是补充也补不上的。
    valid_stocks = []
    for field in data:
        valid_stocks.append(~data[field].isna().all())
    
    # 取所有字段的交集，得到至少有一个非空值的股票
    valid_mask = pd.concat(valid_stocks, axis=1).all(axis=1)
    valid_stock_ids = valid_mask[valid_mask].index.tolist()
    
    # 更新stock_ids和数据
    stock_ids = valid_stock_ids
    for field in data:
        data[field] = data[field][stock_ids]
    
    df = pd.concat(data, axis=1)
    return df, stock_ids

def prepare_dl_dataset_fast(data, window_size=40):
    feature_cols = ['open', 'high', 'low', 'close', 'vwap', 'vol']
    features = np.stack([data[col].values for col in feature_cols], axis=1)
    labels = data['label'].values
    
    # 使用滚动窗口的最后一个值进行标准化
    n_samples = len(features) - window_size
    n_stocks = features.shape[2]
    n_features = len(feature_cols)
    
    X = np.zeros((n_samples * n_stocks, n_features, window_size))
    y = np.zeros(n_samples * n_stocks)
    
    for i in range(n_samples):
        window_data = features[i:i+window_size]
        # 使用窗口内最后一个有效值进行标准化
        last_valid_values = window_data[-1]
        last_valid_values[last_valid_values == 0] = 1
        normalized_window = window_data / last_valid_values[None, :, :]
        
        # 截面标准化（保持不变）
        valid_mask = ~np.isnan(normalized_window)
        means = np.nanmean(normalized_window, axis=2, keepdims=True)
        stds = np.nanstd(normalized_window, axis=2, keepdims=True)
        stds[stds == 0] = 1
        normalized_window = np.where(valid_mask, (normalized_window - means) / stds, normalized_window)
        
        start_idx = i * n_stocks
        end_idx = (i + 1) * n_stocks
        X[start_idx:end_idx] = normalized_window.transpose(2, 1, 0)
        y[start_idx:end_idx] = labels[i+window_size]
    
    return X, y