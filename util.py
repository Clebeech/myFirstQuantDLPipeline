import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import ScalarFormatter
import itertools
import time
from tqdm import tqdm
from glob import glob
from multiprocessing import Pool
from functools import partial
import statsmodels.api as sm
import seaborn as sns
from scipy import stats
from numpy.lib.stride_tricks import as_strided
from numpy.lib.stride_tricks import sliding_window_view
from joblib import Parallel, delayed
import pickle
import yaml
from openpyxl import Workbook
from openpyxl.drawing.image import Image
from openpyxl.utils.dataframe import dataframe_to_rows
from io import BytesIO

from qdb.qdb_api import QdbApi
import qdb.util.research_util as ru
from qdb.factor.factor_mgr import FactorMgr
from qdb.factor.evaluate.plot_format import (
    basic_style,
    SLICE_EX_RET,
    SLICE_CUM_RET,
    CUM_IC,
)
# from qdb.factor.utils import get_stk_pool


q = QdbApi(use_timestamp=True)
data_library = 'BACKTEST'
fm = FactorMgr()
fm.set_lib(library=data_library)




def my_nanmean(arr, axis=0):
    """
    自定义的np矩阵nanmean：整列/行缺失时不会warning
    author: Teng Li, 20231231
    """
    arr_m = arr.copy() # 复制一个，防止改动了输入的矩阵
    is_nan = np.isnan(arr_m).all(axis=axis)
    if axis == 0:
        arr_m.T[is_nan] = 0
    else :
        arr_m[is_nan] = 0
    mean = np.nanmean(arr_m, axis=axis)
    mean[is_nan] = np.nan
    return mean


def normalize_by_row(f):
    """
    逐行标准化
    author: Teng Li, 20231231
    """
    f1 = f.sub(f.mean(axis=1), axis=0).divide(f.std(axis=1), axis=0) # 标准化
    return f1


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



def roll(df, w):
    '''
    Func:生成滑动窗口样本
    Input:
        df：dataframe类型，需要进行取移动滑窗操作的TxN矩阵
        w：窗口大小
    Output:
        长度为T的list, 每个位置是截止当日的滑动窗口面板样本(w*N)，前w-1行的元素为缺失
    author: Mingyang Zhang & Teng Li, 20231031
    '''
    data = df.values
    T, N = data.shape
    s0, s1 = data.strides # 每个数据的步长
    a = as_strided(data, shape=(T-(w-1), w, N), strides=(s0, s0, s1))
    rolling_df = [pd.DataFrame(i, columns=df.columns.tolist()) for i in a]
    return rolling_df


def roll1(df, w):
    """
    author: Teng Li, 20231231
    """
    data = df.values
    T, N = data.shape
    v = sliding_window_view(data, (w,1))
    rolling_df = [pd.DataFrame(i.reshape(N,w).T, columns=df.columns.tolist()) for i in v]
    return rolling_df


def neutralize_old(f0, ind, factors={}, stk_pool=None, min_sample_num=150):
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
    
    # 初始化返回变量f
    dates = f0.index
    stk_ids = f0.columns
    f = pd.DataFrame(index=dates, columns=stk_ids, dtype='float64') # 强制设置为float64类型
    
    # 用f1来处理股票池问题
    if stk_pool is not None:
        f1 = f0[stk_pool]
    else:
        f1 = f0.copy()

    # 中性化：逐日线性回归
    for t in tqdm(dates, desc='中性化'):
        
        # 生成回归中的y
        y = f1.loc[t]
        y1 = f0.loc[t]
        
        # 生成回归中的X
        if ind is None:
            X = pd.DataFrame(index=f0.columns, dtype='float64')
            for fac_name, fac_values in factors.items():
                X[fac_name] = fac_values.loc[t]
            X = sm.add_constant(X)
        else:
            X = pd.get_dummies(ind.loc[t]) + 0 # 生成t日的行业0-1因子
            X = X.rename(columns={e: str(e) for e in X.columns}) # 列名换为str
            # 市值和行业拼接
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



def calc_one_day(t, f0, factors, R_sqr):
    """
    Func: 中性处理（无行业）
    Input:
        t｜str：为当日日期
        f0｜dataframe：为原始因子值矩阵
        factors｜dict：为其他需要中性化的因子，传递为字典，默认为空字典
        R_sqr｜bool：是否计算回归R方
    Output:
        [t日因子值回归预测值,回归R方，回归系数theta]
    author: Lijianan, 20241022
    """
    # 获取当日因子值
    y = f0.loc[t]
    # 设定最小回归数据量
    min_sample_num = 150
    # 生成X
    X = sm.add_constant(pd.DataFrame({fac_name: fac_values.loc[t] for fac_name, fac_values in factors.items()}, dtype='float64'))
    X.index = y.index
    # 找到x、y的非nan交集
    index_X = X.dropna().index
    index_y = y.dropna().index
    index_both = index_X.intersection(index_y)
    
    if len(index_both) > min_sample_num:
        # 使用最小二乘法计算回归系数
        X_b = X.reindex(index=index_both).values
        y_b = y.reindex(index=index_both).values
        theta = np.linalg.solve(X_b.T.dot(X_b),X_b.T.dot(y_b))
        try:
            # 计算预测值
            y_pred = X_b.dot(theta)
            if R_sqr:
                # 计算总平方和 (SST)
                y_mean = np.mean(y_b)
                SST = np.sum((y_b - np.mean(y_b)) ** 2)
                # 计算残差平方和 (SSE)
                SSE = np.sum((y_b - y_pred) ** 2)
                # 计算R方
                R_squared = 1 - (SSE / SST)
            else:
                R_squared = np.nan
        except:
            R_squared = np.nan

        # 得到预测值
        f = pd.Series(y_pred, index=index_both, name=t)
    else:
        f = None
        R_squared = np.nan
        theta = np.full(len(factors)+1, np.nan)
    
    return [f,R_squared,theta]

def calc_one_day_ind(t, f0, factors, ind_t, R_sqr):
    """
    Func: 中性处理（无行业）
    Input:
        t｜str：为当日日期
        f0｜dataframe：为原始因子值矩阵
        factors｜dict：为其他需要中性化的因子，传递为字典，默认为空字典
        ind_t｜series：t日行业Series
        R_sqr｜bool：是否计算回归R方
    Output:
        [t日因子值回归预测值,回归R方，回归系数theta]
    author: Lijianan, 20241022
    """
    # 获取当日因子值
    y = f0.loc[t]
    # 设定最小回归数据量
    min_sample_num = 150
    # 生成回归中的X
    X = pd.get_dummies(ind_t) 
    X = X.replace(False, 0).replace(True, 1) # 生成t日的行业0-1因子
    X = X.rename(columns={e: str(e) for e in X.columns}) # 列名换为str
    # 市值和行业拼接
    for fac_name, fac_values in factors.items():
        X[fac_name] = fac_values.loc[t]
    
    # 找到x、y的非nan交集
    index_X = X.dropna().index
    index_y = y.dropna().index
    index_both = index_X.intersection(index_y)
    
    if len(index_both) > min_sample_num:
        # 使用最小二乘法计算回归系数
        X_b = X.reindex(index=index_both).values
        # 判断是否存在没有覆盖到的行业
        if len(pd.DataFrame(X_b).replace(0,np.nan).dropna(how='all',axis=1).columns) != X_b.shape[1]:
            # 如果存在，删除该列行业
            drop_name = pd.DataFrame(X_b).replace(0,2).replace(1,np.nan).dropna(axis=1).columns[:-1]
            X_b = np.delete(X_b, drop_name, axis=1)
        
        y_b = y.reindex(index=index_both).values
        theta = np.linalg.solve(X_b.T.dot(X_b),X_b.T.dot(y_b))
        try:
            # 计算预测值
            y_pred = X_b.dot(theta)
            if R_sqr:
                # 计算总平方和 (SST)
                y_mean = np.mean(y_b)
                SST = np.sum((y_b - np.mean(y_b)) ** 2)
                # 计算残差平方和 (SSE)
                SSE = np.sum((y_b - y_pred) ** 2)
                # 计算R方
                R_squared = 1 - (SSE / SST)
            else:
                R_squared = np.nan
        except:
            R_squared = np.nan

        # 得到预测值
        f = pd.Series(y_pred, index=index_both, name=t)
    else:
        f = None
        R_squared = np.nan
        theta = np.full(len(factors)+1, np.nan)
    
    return [f,R_squared,theta]

def neutralize(f0, ind=None, factors={}, n_jobs=6, backend='multiprocessing', R_sqr=False, theta_auto=False):
    """
    Func: 中性处理
    Input:
        f0｜dataframe：为原始因子值矩阵
        ind｜dataframe：为标准区间内的行业矩阵
        factors｜dict：为其他需要中性化的因子，传递为字典，默认为空字典
        R_sqr｜bool：为是否计算回归R方
        theta_auto｜bool：为是否计算回归系数的一阶自相关系数
    Output:
        中性化后的TN矩阵 or [中性化后的TN矩阵,R方均值,theta自相关系数均值]
    author: Lijianan, 20241022
    """
    # 如果ind和fators均为空，不做任何处理
    if (ind is None) and len(factors)==0:
        print('Warning：没有中性化参数！')
        return f0
    
    # 获得dates和stk_ids
    dates = f0.index
    stk_ids = f0.columns

    # 根据是否进行行业中性化选择不同的函数多进程并行处理
    if (ind is None):
        # 逐日进行回归（多进程并行）
        results = Parallel(n_jobs=n_jobs, backend=backend)(delayed(calc_one_day)(t, f0, factors, R_sqr) for t in tqdm(dates, desc="逐日进行回归中性化"))
        f = pd.concat([row[0] for row in results], axis=1).T
        f_pre = f
        f = f0 - f.reindex(index=dates,columns=stk_ids)
        # 如果需要计算R方均值
        if R_sqr:
            R = [row[1] for row in results]
            print('R方均值：' + str(pd.Series(R).dropna().mean()))
        else:
            R = np.nan
        # 如果需要计算theta_auto
        if theta_auto:
            theta = [row[2] for row in results]
            theta_df = pd.DataFrame(theta)
            theta_autocorr = theta_df.corrwith(theta_df.shift(-1),axis=1).mean()
            print('theta相关系数均值：' + str(theta_autocorr))
        else:
            theta_autocorr = np.nan
        
    else:
        # 逐日进行回归（多线程并行）
        results = Parallel(n_jobs=n_jobs, backend=backend)(delayed(calc_one_day_ind)(t, f0, factors, ind.loc[t], R_sqr) for t in tqdm(dates, desc="逐日进行回归中性化(带行业)"))
        f = pd.concat([row[0] for row in results], axis=1).T
        f_pre = f
        f = f0 - f.reindex(index=dates,columns=stk_ids)
        # 如果需要计算R方均值
        if R_sqr:
            R = [row[1] for row in results]
            print('R方均值：' + str(pd.Series(R).dropna().mean()))
        else:
            R = np.nan
        # 如果需要计算theta_auto
        if theta_auto:
            theta = [row[2] for row in results]
            theta_df = pd.DataFrame(theta)
            theta_autocorr = theta_df.corrwith(theta_df.shift(-1),axis=1).mean()
            print('theta相关系数均值：' + str(theta_autocorr))
        else:
            theta_autocorr = np.nan

    # 根据参数选择输出的结果
    if (R_sqr+theta_auto) == 2:
        res = [f,R,theta_autocorr]
    elif R_sqr == True:
        res = [f,R]
    elif theta_auto == True:
        res = [f,theta_autocorr]
    else:
        res = f
        
    return res


def count_inversions(series):  
    """
    计算逆序的数量
    author: Teng Li, 20231031
    """
    inversions = 0  
    for i in range(len(series) - 1):  
        for j in range(i + 1, len(series)):  
            if series[i] > series[j]:  
                inversions += 1  
    return inversions  

def get_forward_returns(period=1, trade_mode='trade_close'):
    """ 获取未来收益率 2024.6.11 ljn
    参数
    ----------
        period : int
            未来收益率的期数
        mode : str | optional
            收益率计算方式, 可选值为 ['trade_close', 'trade_open'], 默认值为 'trade_close'
            trade_close 模式下:
                未来 period 期收益率 = (t + period) 期的收盘价 / t 期的收盘价 - 1
            trade_open 模式下: 
                未来 period 期收益率 = (t + period) 期的收盘价 / (t + 1) 期的开盘价 - 1
    """
    assert trade_mode in ['trade_close', 'trade_open']
    cumadj = FactorMgr().get('cumadj')
    adj_close = FactorMgr().get('close') * cumadj
    if trade_mode == 'trade_close':
        target = adj_close.shift(-period) / adj_close - 1
    elif trade_mode == 'trade_open':
        adj_open = FactorMgr().get('open') * cumadj
        target = adj_close.shift(-period) / adj_open.shift(-1) - 1
    return target

def calculate_mutual_information(X, Y, bins=10):
    """
    计算X与Y的互信息
    
    Parameters:
    - X: panel series data of variable X
    - Y: panel series data of variable Y
    - bins: number of bins for discretization
    
    Returns:
    - mutual information of X and Y
    """
    
    # 计算联合熵函数，输入为任意数量的数组
    def joint_entropy(*args):
        # 形成一个二维数组，再计算多维直方图
        hist, edges = np.histogramdd(np.column_stack(args), bins=bins)
        if np.any(np.sum(hist)==0):
            return np.nan
        # 将直方图计数归一化为概率
        p = hist / np.sum(hist)
        # 过滤掉概率为零的值，方便对数计算
        p = p[np.nonzero(p)]
        # 计算联合熵
        joint_entropy = -np.sum(p * np.log2(p))
        return joint_entropy
    
    # 计算条件熵函数
    def conditional_entropy(Y, *Z):
        # 条件熵 = YZ联合熵 - Z信息熵（当Z为多维时，此处为Z的联合熵）
        conditional_entropy = joint_entropy(Y, *Z) - joint_entropy(*Z)
        return conditional_entropy

    entropy_X = joint_entropy(X)
    entropy_Y = joint_entropy(Y)
    joint_entropy_XY = joint_entropy(X,Y)
    
    mutual_information = entropy_X + entropy_Y - joint_entropy_XY
    
    return mutual_information




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

def row_correlation_matrix(X, Y):
    # 确保X和Y是NumPy数组
    X = np.array(X)
    Y = np.array(Y)
    
    # 初始化相关系数数组
    correlation = np.zeros(X.shape[0])

    for i in range(X.shape[0]):
        # 提取每一行并移除NaN
        xi = X[i, :]
        yi = Y[i, :]
        valid_mask = ~np.isnan(xi) & ~np.isnan(yi)
        xi = xi[valid_mask]
        yi = yi[valid_mask]
        
        # 如果所有元素都为NaN，跳过计算
        if len(xi) == 0 or len(yi) == 0:
            correlation[i] = np.nan
        else:
            # 计算均值
            mean_xi = np.mean(xi)
            mean_yi = np.mean(yi)
            # 计算偏差平方和
            ss_xi = np.sum((xi - mean_xi)**2)
            ss_yi = np.sum((yi - mean_yi)**2)
            # 计算协方差
            cov_xy = np.sum((xi - mean_xi) * (yi - mean_yi))
            # 计算相关系数
            correlation[i] = cov_xy / np.sqrt(ss_xi * ss_yi)
    
    return correlation

def first_n_non_na_bool(df, n):
    """
    找到一个dataframe每列的前n个非nan值的位置

    参数:
    p - dataframe
    n - int

    返回:
    dataframe : bool类型的dataframe，True值为每列前n个非nan值的位置
    """
    def check_first_n(col):
        bool_array = np.zeros_like(col, dtype=bool)
        non_na_indices = np.where(~col.isna())[0]  # 获取非NaN的索引
        bool_array[non_na_indices[:n]] = True      # 前n个非NaN值设置为True
        return bool_array
    
    # 对df的每一列应用check_first_n函数
    return df.apply(check_first_n, axis=0)

def llt(df, a):
    """
    计算时间序列数据的低延迟滤波器（Low Lag Tracking, LLT）值。

    参数:
    p - dataframe
    a - 滤波器系数，介于0到1之间

    返回:
    dataframe - 经过llt滤波后的序列dataframe
    """
    # 获取非nan的mask
    df_mask = ~df.isna()
    # llt计算需要保证从有数据开始没有nan值
    df = df.fillna(method='ffill')
    
    # 找到每列的前2个非NaN值的布尔矩阵
    bool_matrix = first_n_non_na_bool(df, 2)
    df_llt = df[bool_matrix].copy()
    # 循环每日计算所有截面
    n = len(df)
    for t in tqdm(range(n)):
        if t == 0 or t == 1:
            continue
        else:
            mask = df_llt.iloc[t].isna()
            df_llt.iloc[t][mask] = (
                (a - 1/4 * a**2) * df.iloc[t] +
                (1/2 * a**2) * df.iloc[t-1] -
                (a - 3/4 * a**2) * df.iloc[t-2] +
                2 * (1 - a) * df_llt.iloc[t-1] -
                (1 - a)**2 * df_llt.iloc[t-2]
            )[mask]

    return df_llt[df_mask]

def row_corr(df1, df2):
    """计算两个TxN矩阵的逐行相关性，允许有nan"""
    df1 = df1.values
    df2 = df2.values
    T, N = df1.shape
    rc = np.ones(T) * np.nan
    for t in range(T):
        x = df1[t,:]
        y = df2[t,:]
        valid_mask = ~np.isnan(x) & ~np.isnan(y)
        x = x[valid_mask]
        y = y[valid_mask]
        if len(x) == 0 or len(y) == 0:
            rc[t] = np.nan
        else:
            x = x - np.mean(x)
            y = y - np.mean(y)
            var_x = np.sum(x**2)
            var_y = np.sum(y**2)
            if var_x * var_y == 0:
                rc[t] = np.nan
            else:
                rc[t] = np.sum(x*y) / np.sqrt(var_x * var_y)
    return rc