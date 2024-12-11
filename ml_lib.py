import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm as original_tqdm
from glob import glob
import importlib
from multiprocessing import Pool
from functools import partial
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import seaborn as sns
from scipy import stats
from numpy.lib.stride_tricks import as_strided
from numpy.lib.stride_tricks import sliding_window_view
import joblib
from joblib import Parallel, delayed
import pickle
import yaml
from openpyxl import Workbook
from openpyxl.drawing.image import Image
from openpyxl.utils.dataframe import dataframe_to_rows
from io import BytesIO
import copy
import gc
import logging
from datetime import datetime
from sklearn.model_selection import KFold
import optuna
from optuna.storages.journal import JournalFileBackend
from qdb import QdbApi
q = QdbApi(use_timestamp=True)
from qdb.factor.factor_mgr import FactorMgr
fm = FactorMgr()
data_library = 'BACKTEST'
fm.set_lib(data_library)

# ============================================
#               log_setting.py
# ============================================
class StreamToLogger:
    """Fake file-like stream object that redirects writes to a logger instance."""
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

logger = logging.getLogger()

tqdm_out = StreamToLogger(logger, log_level=logging.INFO)


# Define a custom tqdm wrapper
def tqdm(*args, **kwargs):
    kwargs.setdefault('file', tqdm_out)
    kwargs.setdefault('mininterval', 60)
    kwargs.setdefault('leave', True)
    return original_tqdm(*args, **kwargs)

def logging_config(output_folder):
    
    # 默认在已存在的model_log上添加
    logging.basicConfig(
        filename = output_folder + "model_log.log",
        level = logging.INFO,
        format = "%(asctime)s %(levelname)s %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
        force = True
    )
    # Redirect stdout and stderr to logging
    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)
    logging.info("日志初始化成功！")

# ============================================
#             parameter_search.py
# ============================================
class abstract_optuna_search:
    def __init__(self, output_folder, temp_folder, optuna_setting, model_name, t):
        for key, value in optuna_setting.items():
            setattr(self, key, value)
        self.study_name = model_name+t.strftime('%Y-%m-%d')
        self.output_folder = output_folder
        self.temp_folder = temp_folder
        
        self.sample_parameter_space = {}       
    
    def _create_study(self):
        """设置optuna trials的存储日志路径 self.storage 并创建study"""
        journal_file_path = self.output_folder+'study_journal.log'
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        self.storage = optuna.storages.JournalStorage(JournalFileBackend(journal_file_path))
        self.study = optuna.create_study(storage=self.storage, study_name=self.study_name, direction='minimize', load_if_exists=True, pruner=self.pruner)
    
    def _set_sample_parameter_space(self, trial):
        """帮助trial从超参搜索空间中进行超参采样"""

        for key, (param_type, *values) in self.parameter_space.items():
            if param_type == 'categorical':
                self.sample_parameter_space[key] = trial.suggest_categorical(key, values)
            elif param_type == 'int':
                if values[2] is not None:
                    self.sample_parameter_space[key] = trial.suggest_int(key, values[0], values[1], step=values[2])
                elif values[3]:
                    self.sample_parameter_space[key] = trial.suggest_int(key, values[0], values[1], log=values[3])
                else:
                    self.sample_parameter_space[key] = trial.suggest_int(key, values[0], values[1])
            elif param_type == 'float':
                if values[2] is not None:
                    self.sample_parameter_space[key] = trial.suggest_float(key, values[0], values[1], step=values[2])
                elif values[3]:
                    self.sample_parameter_space[key] = trial.suggest_float(key, values[0], values[1], log=values[3])
                else:
                    self.sample_parameter_space[key] = trial.suggest_float(key, values[0], values[1])
            else:
                raise ValueError(f"Unsupported parameter type: {param_type}!")
        
        # 更新在现有超参空间之外的固定参数
        for key, value in self.fixed_parameters.items():
            if key not in self.sample_parameter_space:
                self.sample_parameter_space[key] = value

        if self.device == 'gpu':
            self.sample_parameter_space['device'] = 'gpu'
            self.sample_parameter_space['gpu_platform_id'] = 0
            self.sample_parameter_space['gpu_device_id'] = trial.number%2 
        # print(self.sample_parameter_space)
        
    def __objective__(self, trial):
        # print("__objective__ called!")
        return self.objective(trial)

    def run_study_trials(self, n_trials, study_name, objective_function):
        self.study = optuna.load_study(study_name=study_name, storage=self.storage)
        self.study.optimize(objective_function, n_trials=n_trials, gc_after_trial=True)
        
        return 0

    def optuna_training(self, t):
        self._create_study()

        # 计算每个进程需要执行的 trial 数量
        trials_per_job = self.n_trials // self.n_jobs_trial
        extra_trials = self.n_trials % self.n_jobs_trial  # 处理不能被整除的部分

        with joblib.parallel_backend('loky', temp_folder=self.temp_folder):
            Parallel(n_jobs=self.n_jobs_trial)(delayed(self.run_study_trials)(
                trials_per_job + (1 if i < extra_trials else 0),
                self.study_name,
                self.__objective__) for i in range(self.n_jobs_trial))
                
class PurgedKFold(KFold):
    def __init__(self, n_splits=5, gap=1, shuffle=False, random_state=None):
        super().__init__(n_splits, shuffle=shuffle, random_state=random_state)
        self.gap = gap

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        indices = np.arange(n_samples)
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_idx = indices[start:stop]
            if start > 0:
                train_idx = indices[:start-self.gap].tolist() + indices[stop+self.gap:].tolist()
            else:
                train_idx = indices[stop+self.gap:].tolist()
            current = stop
            if len(train_idx) > 0 and len(test_idx) > 0:
                yield np.array(train_idx), np.array(test_idx)

# ============================================
#                  util.py
# ============================================

def ensure_tuple(obj):
    """ 将对象转换为元组
    参数
    ----------
        obj : object
            待转换的对象
    """
    assert obj is not None
    if isinstance(obj, tuple):
        return obj
    elif isinstance(obj, list):
        return tuple(obj)
    else:
        return (obj,)

def ensure_list(param):
    """ 将对象转换为列表
    参数
    ----------
        obj : object
            待转换的对象
    """
    if isinstance(param, list):
        return param
    else:
        # 如果不是列表，将其转换为包含该元素的列表
        return [param]

def ensure_dict(param):
    """ 将对象转换为字典
    参数
    ----------
        obj : object
            待转换的对象
    """
    if isinstance(param, dict):
        return param
    else:
        # 如果不是列表，将其转换为包含该元素的列表
        return [param]


def handle_na(data):
    """处理NA值，目前仅将NA值填为0"""
    data = data.fillna(0)
    return data

# 新版get_stk_pool 2024-12-2
def get_stk_pool(stk_pool='all', filter_by_status=True):
    """ 获取股票池
    参数
    ----------
        stk_pool : str | optional
            股票池名称, 可选值为 ['all', 'hs300', 'csi500', 'csi800', 'csi1000', 'top1500', 'top2000', 'top2500'], 
            默认值为 'all', 即全A股票（剔除状态的）
    """
    # 检查输入
    if type(stk_pool) is pd.DataFrame:
        return stk_pool
    assert stk_pool in ['all', 'hs300', 'csi500', 'csi800', 'csi1000', 'top1500', 'top2000', 'top2500']
    assert isinstance(filter_by_status, bool), "filter_by_status should be a boolean"
    # 读取股票池
    if stk_pool == 'all':
        stk_pool_tn = FactorMgr().get("close").notnull()
    elif stk_pool in ['hs300', 'csi500', 'csi800', 'csi1000']:
        wgts = FactorMgr().get(f'wgts_{stk_pool}').copy()
        wgts_sum0 = (wgts.sum(axis=1) == 0)
        for i, date_i in enumerate(wgts.index):
            if i == 0:
                continue
            if wgts_sum0.iloc[i]: 
                wgts.loc[date_i] = wgts.iloc[i-1]
        stk_pool_tn = wgts.notnull()
    elif stk_pool in ['top1500', 'top2000', 'top2500']:
        stk_pool_tn = FactorMgr().get(f'univ_{stk_pool}_new').copy()
    else:
        print(f"unknown stk_pool = {stk_pool}")
        return
    if filter_by_status:
        # 剔除掉状态异常的股票
        eval_status = FactorMgr().get('eval_status') 
        stk_pool_tn = stk_pool_tn & (eval_status != 1) & (eval_status != 2) & (eval_status != 3) & (eval_status != 6)
        # 剔除近20日成交额小于5%分位数的股票
        amt = FactorMgr().get('amt')
        amt_quantile = amt.apply(lambda row: row.quantile(0.05), axis=1)
        mask_amt = (amt.rolling(20).mean().T - amt_quantile).T > 0
        # 剔除市值小于1%分位数的股票
        cap = FactorMgr().get('cap')
        cap_quantile = cap.apply(lambda row: row.quantile(0.01), axis=1)
        mask_cap = (cap.T - cap_quantile).T > 0

        stk_pool_tn = stk_pool_tn[mask_amt]
        stk_pool_tn = stk_pool_tn[mask_cap]
        
    return stk_pool_tn


def filter_factors(factors_dict, stk_pool):
    """
    本函数根据股票池过滤因子矩阵。

    Input:
    - factors_dict: 一个字典，键为日期，值为DataFrame，每个DataFrame是N x K矩阵，表示因子数据。
    - stk_pool: str 表示选择的股票池名称，如 'hs300'

    Output:
    - 一个字典，键为日期，值为过滤后的DataFrame，尺寸为N x K。

    Author: Yusen Su 20240618
    """
    stock_pool = get_stk_pool(stk_pool)
    filtered_factors_dict = {}
    for date, factor_data in tqdm(factors_dict.items(), desc = '逐日筛选指定股票池内股票:', mininterval=60):
        # 获取当天的股票池，True表示选中的股票
        selected_stocks = stock_pool.loc[date]
        
        # 复制因子数据以避免修改原数据
        filtered_factor_data = factor_data.copy()
        
        # 未被选中的股票设置为NaN
        filtered_factor_data = filtered_factor_data.loc[filtered_factor_data.index.get_level_values(0)[selected_stocks],:]
        
        # 将处理后的数据存入字典
        filtered_factors_dict[date] = filtered_factor_data
        
    return filtered_factors_dict


def save_dict_to_yaml(d, yaml_file_name):
    """
    # 将字典写入YAML文件
    author: Teng Li, 20231231
    """
    with open(yaml_file_name, 'w') as f:  
        yaml.dump(d, f)


def neutralize(f0, ind=None, factors={}, stk_pool=None, min_sample_num=60):
    """
    Func: 中性处理
    Input:
        ind为行业TN矩阵，如果为None则不进行行业中性化
        factors为其他需要中性化的因子，传递为字典，默认为空字典
        stk_pool为回归时判断是否参与拟合的TN大小的0-1矩阵，默认为None
        当某天非空值小于min_sample_num时，该天就全部设为空，默认为150
    Output:
        中性化后的TN矩阵
    author: Mingyang Zhang & Teng Li, 20231231
    """
    
    # 如果ind和fators均为空，不做任何处理
    if (ind is None) and len(factors)==0:
        print('Warning：没有中性化参数！')
        return f0
    
    # 用f1来处理股票池问题
    if stk_pool is not None:
        f1 = f0[stk_pool]
    else:
        f1 = f0.copy()

    # 中性化：逐日线性回归
    dates = f0.index
    stk_ids = f0.columns
    f = pd.DataFrame(index=dates, columns=stk_ids, dtype=np.float64) # 初始化返回变量
    for t in tqdm(dates, desc='中性化', mininterval=60):
        
        # 生成回归中的y
        y = f1.loc[t]
        y1 = f0.loc[t]
        
        # 生成回归中的X
        X = pd.DataFrame(index=f0.columns, dtype=np.float64)
        if ind is None:
            X = sm.add_constant(X) # 添加常数项
        else:
            ind_expo = pd.get_dummies(ind.loc[t]) + 0 # 生成t日的行业0-1因子
            # ind_expo.columns = [str(e) for e in ind_expo.columns] # 列名换为str
            X = X.join(ind_expo)
        for fac_name, fac_values in factors.items():
            X[fac_name] = fac_values.loc[t]
        
        # 回归计算
        if y1.notnull().sum() > min_sample_num and X.notnull().all(axis=1).sum() > min_sample_num: 
            if X.shape[1]==0:
                print(f"中性化: {t.date()}, 回归中X的列数是0，返回nan值")
                f.loc[t] = np.nan # 处理最新一天数据可能完全缺失的情形
            else:
                model = sm.OLS(y1, X, missing='drop') # 生成模型
                result = model.fit() # 模型拟合
                f.loc[t] = y - X @ result.params
    
    return f


def normalize_by_row(f):
    """
    逐行标准化
    author: Teng Li, 20231231
    """
    f1 = f.sub(f.mean(axis=1), axis=0).divide(f.std(axis=1), axis=0) # 标准化
    return f1




# ============================================
#             prepare_data.py
# ============================================
def calc_y_rolling_mean(df, average_type, window_size):
    """ 计算滑动平均
    参数:
    ----------
        df: dataframe
            t x n 矩阵
        average_type: str 
            权重定义: 'mean'等权平均, 'exp'权重为[exp(-1),exp(-2),...,exp(-k)]归一化
        window_size: int
            滑动平均窗口长度
    输出:
    ----------
        滑动平均后的 t x n 矩阵
    修改日期：20241204
    """
    if average_type == 'mean':
        res = df.rolling(window = window_size, min_periods=1).mean()
    elif average_type == 'exp':
        alpha = 1 - np.exp(-1)
        alpha_rev = 1 - alpha
        pows = alpha_rev**(np.arange(window_size)) # Exponential权重array
        moving_weights = pows/pows.sum() # 归一化
        res = moving_weights[0]*df
        for w in np.arange(1, window_size, 1):
            res += moving_weights[w] * df.shift(w)
    return res

def calculate_X_rolling_mean(dates, factor_nk, average_type, window_size):
    """
    计算X的移动平均，两种方式同上
    Input:
    - dates: dataframe | t x n 矩阵
    - facotr_nk: dict | 一个字典，键为日期，值为DataFrame，每个DataFrame是N x K矩阵，表示因子数据
    - average_type: str | 权重定义: 'mean'等权平均, 'exp'权重为[exp(-1),exp(-2),...,exp(-k)]归一化
    - window_size: int | 滑动平均窗口长度
    Output:
    - 滑动平均后的dict
    """
    rolling_factor_nk = {}
    
    if average_type == 'mean':
        for t in tqdm(dates, desc="计算X移动平均"):
            if t >= dates[window_size - 1]:
                rolling_window = pd.concat(
                    [factor_nk[dates[i]] for i in range(dates.get_loc(t) - window_size + 1, dates.get_loc(t) + 1)]
                )
                rolling_factor_nk[t] = rolling_window.groupby(level=0).mean()
            else:
                rolling_factor_nk[t] = factor_nk[t]
                
    elif average_type == 'exp':
        for t in tqdm(dates, desc="计算X移动平均"):
            df = pd.DataFrame(np.zeros_like(factor_nk[t]), columns = factor_nk[t].columns, index = factor_nk[t].index)
            if t >= dates[window_size - 1]:
                exp_values = np.exp(-np.arange(1, window_size + 1))
                X_weights = (exp_values / np.sum(exp_values))[::-1]
                df1 = df.copy()
                for i in range(dates.get_loc(t) - window_size + 1, dates.get_loc(t) + 1):
                    df1 += factor_nk[dates[i]] * X_weights[i - (dates.get_loc(t) - window_size + 1)]
                rolling_factor_nk[t] = df1
            else:
                rolling_factor_nk[t] = factor_nk[t]
    return rolling_factor_nk

def merge_returns_dict(df_x_dict, df_ret):
    """
    本函数将来自 df_ret 的收益率数据合并到 df_x_dict 中的每个 DataFrame。

    Input:
    - df_x_dict: dictionary of dataframes，每个key表示一个时间 ID ，value是含有股票特征 X 的 DataFrame。
    - df_ret: DataFrame，其中每列是日期，每行表示对应列日期的收益率。

    Output:
    - 更新后的dict，其中每个 DataFrame 都已合并了收益率数据。

    Author: Yusen Su 20240605
    """
    df_y = df_ret.copy().T
    # 更新 data_x_dict 中的每个 DataFrame 添加到新的dict中
    df_x_dict_local = {}
    for time_id, df_x in tqdm(df_x_dict.items(), desc = "合并X和y", mininterval=60):
        # 如果 time_id 还不是 pd.Timestamp，将其转换为 pd.Timestamp
        if not isinstance(time_id, pd.Timestamp):
            if re.match(r'^\d{8}$', time_id):
                time_id_new = pd.to_datetime(time_id, format='%Y%m%d')
            elif re.match(r'^\d{4}-\d{2}-\d{2}$', time_id):
                time_id_new = pd.to_datetime(time_id, format='%Y-%m-%d')
        else:
            time_id_new = time_id.strftime('%Y-%m-%d')
            
        # 提取对应时间 ID 的 Y 列
        if time_id_new in df_y.columns:
            y_series = df_y[time_id_new]
            if 'index' in df_x.columns:
                y_series.index = df_x['index']  # 确保股票索引与 Y 的索引匹配
                # 添加收益率为新列
                df_x['ret'] = df_x['index'].map(y_series)
            else:
                # y_series.index = df_x.index  # 确保股票索引与 Y 的索引匹配
                # # 添加收益率为新列
                df_x['ret'] = df_x.index.map(y_series)
        else:
            print(f"该时间点 {time_id} 没有可用数据")
        # 更新原数据列表中的 DataFrame
        df_x_dict_local[time_id] = df_x
    return df_x_dict_local


def merge_weights_dict(df_x_dict, df_weights):
    """
    本函数将给每个样本点赋的权重合并到 df_x_dict 中的每个 DataFrame

    Input:
    - df_x_dict: dictionary of dataframes, 每个key表示一个时间 ID , value是含有股票特征 X 的 DataFrame。
    - df_weights: Dataframe, 每列是日期, 每行表示不同股票, 每个元素表示对应日期和股票的样本权重

    Output:
    - 更新后的dict, 其中每个DataFrame都已合并了样本权重数据
    """
    # df_y is of shape (num_stocks, num_dates)
    df_y = df_weights.copy().T
    # 更新后的data_x_dict_new中的每个dataframe都合并了weights数据
    df_x_dict_new = {}
    for time_id, df_x in tqdm(df_x_dict.items(), desc = 'Merging weights:'):
        # 如果 time_id 还不是 pd.Timestamp，将其转换为 pd.Timestamp
        if not isinstance(time_id, pd.Timestamp):
            if re.match(r'^\d{8}$', time_id):
                time_id_new = pd.to_datetime(time_id, format='%Y%m%d')
            elif re.match(r'^\d{4}-\d{2}-\d{2}$', time_id):
                time_id_new = pd.to_datetime(time_id, format='%Y-%m-%d')
        else:
            time_id_new = time_id.strftime('%Y-%m-%d')

        if time_id_new in df_y.columns:
            y_series = df_y[time_id_new]
            if 'index' in df_x.columns:
                y_series.index = df_x['index']  # 确保股票索引与 Y 的索引匹配
                # 添加样本权重为新列
                df_x['weights'] = df_x['index'].map(y_series)
            else:
                # y_series.index = df_y.index  # 确保股票索引与 Y 的索引匹配
                # # 添加样本权重为新列
                df_x['weights'] = df_x.index.map(y_series)       
        else:
            print(f"该时间点 {time_id} 没有可用数据")
        # 更新原数据列表中的 DataFrame
        df_x_dict_new[time_id] = df_x
    return df_x_dict_new  

def drop_out_by_row(data, method='pull_back', threshold=3):
    """
    Func: 逐行去极值函数：用中位数加减指定倍数的平均绝对偏离度来计算上下界
    Input:
        data：dataframe of size TxN，可以有nan
        method：'pull_back'（拉回到边界） or 'drop'（丢弃样本）
        treshold：阈值，计算上下界时mad的倍数
    author: Teng Li, 20231231
    """

    # 转为np.array --> data_np
    data_np = data.values.copy()

    # 计算每行的中位数median和平均绝对偏离度mad，tile成同尺寸矩阵
    (T,N) = data_np.shape
    median = np.tile(np.nanmedian(data_np, axis=1).reshape(-1,1), [1,N])
    mad = np.tile(np.nanmedian(np.abs(data_np - median), axis=1).reshape(-1,1), [1,N])

    # 计算上下界
    lower_bound = median - threshold * mad
    upper_bound = median + threshold * mad

    # 替换大于上界的样本
    if method == 'pull_back':
        data_np[data_np > upper_bound] = upper_bound[data_np > upper_bound]
    elif method == 'drop':
        data_np[data_np > upper_bound] = np.nan

    # 替换小于下界的样本
    if method == 'pull_back':
        data_np[data_np < lower_bound] = lower_bound[data_np < lower_bound]
    elif method == 'drop':
        data_np[data_np < lower_bound] = np.nan

    # 复原为dataframe
    data1 = pd.DataFrame(data_np, index=data.index, columns=data.columns)

    return data1