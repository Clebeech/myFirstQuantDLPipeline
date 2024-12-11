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

from util import *


q = QdbApi(use_timestamp=True)
data_library = 'BACKTEST'
fm = FactorMgr()
fm.set_lib(library=data_library)


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
        stk_pool_tn = stk_pool_tn & (eval_status != 1) & (eval_status != 2) & (eval_status != 3)
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


# 检查因子分布
# util.normality_test(f)
def normality_test(f):
    """
    Func: 正态性测试
    author: Teng Li, 20231231
    """
    f1 = normalize_by_row(f) # 按行标准化
    # 进行Kolmogorov-Smirnov检验
    data = f1.values.reshape(-1)
    data = data[~np.isnan(data)]
    statistic, p_value = stats.kstest(data, 'norm')
    print('Statistic (KS Test):', statistic)  
    print('P-value (KS Test):', p_value)
    # 画概率密度函数
    data = f1.sample(20) # 样本量太大，随机抽取10行作为样本
    data = data.values.reshape(-1) # 面板样本混为一行
    bmk_data = np.random.randn(1000)
    plt.figure(figsize=(10,4))
    sns.kdeplot(data)
    sns.kdeplot(bmk_data)
    plt.legend(['Sample', 'Standard normal N(0,1)'])
    plt.grid(True)
    plt.title('Probability density distribution')
    # 正态性：截面偏度、截面丰度
    hdl = plt.figure()
    tmp = pd.DataFrame(index=f.index)
    tmp['skew'] = f1.skew(axis=1)
    tmp['kurtosis'] = f1.kurtosis(axis=1)
    tmp.plot(grid=True, subplots=True, figsize=(12,6))
    print('偏度时序均值：'+str(tmp.mean()['skew']))
    print('峰度时序均值：'+str(tmp.mean()['kurtosis']))

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
    
# 计算股票池覆盖度
# coverage = util.calc_stk_pool_coverage(f)
def calc_stk_pool_coverage(f):
    """
    计算股票池覆盖率
    author: Teng Li, 20231130
    """
    fm = FactorMgr()
    pool_300 = get_stk_pool('hs300')
    pool_500 = get_stk_pool('csi500')
    pool_1000 = get_stk_pool('csi1000')
    pool_2500 = get_stk_pool('top2500')
    # list_time = fm.get('list_time')
    pool_all = get_stk_pool('all')

    coverage = pd.DataFrame()
    coverage['300'] = f[pool_300].count(axis=1) / pool_300.sum(axis=1)
    coverage['500'] = f[pool_500].count(axis=1) / pool_500.sum(axis=1)
    coverage['1000'] = f[pool_1000].count(axis=1) / pool_1000.sum(axis=1)
    coverage['top2500'] = f[pool_2500].count(axis=1) / pool_2500.sum(axis=1)
    # coverage['all'] = f.count(axis=1) / (list_time>0).sum(axis=1)
    coverage['all'] = f[pool_all].count(axis=1) / pool_all.sum(axis=1)

    coverage.plot(figsize=(16,8), grid=True, title='stk pool coverage')
    print("股票池覆盖率:")
    display(coverage.mean())

    return coverage


# 因子评估：slice分析 --> res_slice
# res_slice = util.slice_analysis_new(fe, f, stk_pool='top2500', reb_freq=1, slice_num=10, trade_mode='trade_vwap_allday') 
def slice_analysis_new(f, stk_pool='all', reb_freq=1, slice_num=10, trade_mode='trade_open', display=True):
        """ 分层分析
        参数
        ----------
            reb_freq : int | optional
                调仓周期，默认为1
            slice_num : int | optional
                分层数, 默认为 5
            trade_mode : str | optional
                交易模式，可选值为 ['trade_close', 'trade_open'], 默认为 'trade_open'
        """
        stk_pool_tn = get_stk_pool(stk_pool) # 股票池tn矩阵
        # 剔除掉状态异常的股票  2024-12-2
        eval_status = FactorMgr().get('eval_status') 
        stk_pool_tn = stk_pool_tn & (eval_status != 6)
    
        f1 = f[stk_pool_tn] # 屏蔽股票池外的因子值
        slice_ret, close_weight_np = _calc_return_by_slice(f1, reb_freq, slice_num, trade_mode) # 计算分层组合收益
        ex_ret = _calc_ex_return_by_slice(slice_ret, slice_num, benchmark="equal_wgt") # 计算各slice的超额收益序列
        perf_ind = ru.cal_period_perf_indicator((1+ex_ret).cumprod(), rf=0) # 计算超额业绩指标
        top_wgt = close_weight_np[:,:,-1] # top组合权重numpy矩阵
        top_turn_over = np.mean(np.sum(np.abs(np.diff(top_wgt, axis=0)), axis=1)) / 2 # 计算单边换手率
        res = {'slice_ret': slice_ret, 'ex_ret':ex_ret, 'close_wgt_np': close_weight_np, 'perf_ind':perf_ind, 
              'top_wgt':top_wgt, 'top_turn_over':top_turn_over}
        # 展示分析图表
        if display is True:
            print_table(ru.indicator_formator(perf_ind)[['年化收益', '年化波动', '夏普率', '最大回撤', '卡玛率', '平均回撤']]) # 超额业绩指标表
            # print(perf_ind)
            fig = plot_cummulative_ex_returns(ex_ret, benchmark="equal_wgt", reb_freq=reb_freq) # 画超额收益走势图
            # ex_ret.cumsum().plot()
            print(f"top portfolio avg daily one-round turn_over is {top_turn_over*100:.2f}%") # print top slice换手率
            res['fig'] = fig


        return res

def _calc_return_by_slice(factor, reb_freq, slice_nums, trade_mode="trade_close"):
    """ 计算因子分层组合收益：按因子值大小对股票排序，分层，计算各分层的等权组合收益
    参数
    ----------
        factor : pd.DataFrame
            因子值
        reb_freq : int
            调仓周期
        slice_nums : int
            分层数
        trade_mode : str | optional
            交易模式, 可选值为 ['trade_close', 'trade_open'], 默认值为 'trade_close'
    """
    assert trade_mode in ['trade_close', 'trade_open', 'trade_vwap_allday']
    T, N = factor.shape
    if trade_mode == "trade_close":        
        interday_ret = _calc_daily_return(mode="interday").fillna(0).values  # 日间收益率
        close_weight_np = np.zeros((T, N, slice_nums + 1))  # 每期 close 时的持仓权重
        port_ret_np = np.zeros((T, slice_nums + 1))        # 每期 close 时的持仓收益率 = 上期 close 时的持仓权重 * 今天的日间收益率 
        for t in tqdm(range(0, T), desc="分层分析"):
            if t % reb_freq == 0:
                close_weight_np[t] = _trade(factor.iloc[t], slice_nums)
            else:
                close_weight_np[t] = close_weight_np[t - 1] * (1 + interday_ret[t][:,np.newaxis])
        close_weight_np = close_weight_np / (np.sum(close_weight_np, axis=1, keepdims=True) + 1e-8)
        port_ret_np[1:] = np.sum(close_weight_np[:-1] * interday_ret[1:,:,np.newaxis], axis=1)
        columns = ['S' + str(e) if e != 0 else 'equal_wgt' for e in range(slice_nums + 1)]
        port_ret = pd.DataFrame(port_ret_np, index=factor.index, columns=columns)
    elif trade_mode == "trade_open":
        intraday_ret = _calc_daily_return(mode="intraday").fillna(0).values  # 日内收益率
        overnight_ret = _calc_daily_return(mode="overnight").fillna(0).values  # 隔夜收益率
        close_weight_np = np.zeros((T, N, slice_nums + 1)) # 每期 close 时的持仓权重
        open_weight_np = np.zeros((T, N, slice_nums + 1))  # 每期 open 时的持仓权重
        # 每期 close 时的持仓收益率 = (上期 close 时的持仓权重 * 隔夜收益率 + 1) * (open 时的持仓权重 * 日内收益率 + 1) - 1
        port_ret_np = np.zeros((T, slice_nums + 1))
        # 第零天没有信号，不交易
        for t in tqdm(range(1, T), desc="分层分析"):
            if t % reb_freq == 0:
                open_weight_np[t] = _trade(factor.iloc[t - 1], slice_nums)
            else:
                open_weight_np[t] = close_weight_np[t - 1] * (1 + overnight_ret[t][:,np.newaxis])
            close_weight_np[t] = open_weight_np[t] * (1 + intraday_ret[t][:,np.newaxis]) 
        open_weight_np = open_weight_np / (np.sum(open_weight_np, axis=1, keepdims=True) + 1e-8)
        close_weight_np = close_weight_np / (np.sum(close_weight_np, axis=1, keepdims=True) + 1e-8)
        ret1 = np.sum(close_weight_np[:-1] * overnight_ret[1:,:,np.newaxis], axis=1)
        ret2 = np.sum(open_weight_np[1:] * intraday_ret[1:,:,np.newaxis], axis=1)
        port_ret_np[1:] = (ret1 + 1) * (ret2 + 1) - 1
        columns = ['S' + str(e) if e != 0 else 'equal_wgt' for e in range(slice_nums + 1)]
        port_ret = pd.DataFrame(port_ret_np, index=factor.index, columns=columns)
    elif trade_mode == "trade_vwap_allday":
        vwap_allday_ret = _calc_daily_return(mode="vwap_allday").fillna(0).values  # vwap_allday收益率
        vwap_weight_np = np.zeros((T, N, slice_nums + 1))  # 每期 vwap 时的持仓权重
        # 每期 vwap 时的持仓收益率 = 
        port_ret_np = np.zeros((T, slice_nums + 1))
        # 第零天没有信号，不交易
        for t in tqdm(range(1, T), desc="分层分析"):
            if t % reb_freq == 0:
                vwap_weight_np[t] = _trade(factor.iloc[t - 1], slice_nums)
            else:
                vwap_weight_np[t] = vwap_weight_np[t - 1] * (1 + vwap_allday_ret[t][:,np.newaxis])
        vwap_weight_np = vwap_weight_np / (np.sum(vwap_weight_np, axis=1, keepdims=True) + 1e-8)
        
        ret1 = np.sum(vwap_weight_np[:-1] * vwap_allday_ret[1:,:,np.newaxis], axis=1)
        # ret1 = np.sum(vwap_weight_np[1:] * vwap_allday_ret[1:,:,np.newaxis], axis=1)
        port_ret_np[1:] = ret1
        columns = ['S' + str(e) if e != 0 else 'equal_wgt' for e in range(slice_nums + 1)]
        port_ret = pd.DataFrame(port_ret_np, index=factor.index, columns=columns)  
        close_weight_np = vwap_weight_np
        
    return port_ret, close_weight_np
def _calc_daily_return(mode='interday'):
    cumadj = FactorMgr().get('cumadj')
    adj_close = FactorMgr().get('close') * cumadj
    adj_open = FactorMgr().get('open') * cumadj
    adj_vwap_allday = FactorMgr().get('vwap') * cumadj
    assert mode in ['interday', 'intraday', 'overnight', 'vwap_allday']
    if mode == 'interday':
        returns = adj_close.pct_change()
    elif mode == 'intraday':
        returns = adj_close / adj_open - 1
    elif mode == 'overnight':
        returns = adj_open / adj_close.shift(1) - 1
    elif mode == 'vwap_allday':
        returns = adj_vwap_allday.pct_change()
    return returns
def _trade(t_factor, slice_nums):
    """ 分层等权调仓
    参数
    ----------
        t_factor : pd.Series
            第 t 期的因子值
    """
    weight = np.zeros((len(t_factor), slice_nums + 1))  # 每股票每层的持仓权重，第一列为所有股票等权
    rank = t_factor.rank(ascending=True)
    avg_stk_num = (rank > 0).sum() / slice_nums
    weight[rank > 0, 0] = 1 / ((rank > 0).sum() + 1e-8)  # 等权
    for s in range(1, slice_nums + 1):
        mask = (rank > avg_stk_num * (s - 1)) & (rank <= avg_stk_num * s)
        weight[mask, s] = 1 / (mask.sum() + 1e-8)
    return weight
def _calc_ex_return_by_slice(slice_rets, slice_nums, benchmark):
    assert benchmark in slice_rets.columns

    slice_columns = ['S' + str(e) for e in range(1, slice_nums + 1)]
    ex_ret = slice_rets[slice_columns].sub(slice_rets[benchmark], axis=0)
    ex_ret["top_bottom"] = ex_ret[f'S{slice_nums}'] - ex_ret['S1']
    ex_ret[benchmark] = slice_rets[benchmark]

    return ex_ret

def print_table(table, name=None, fmt=None):
    if isinstance(table, pd.Series):
        table = pd.DataFrame(table)
    if isinstance(table, pd.DataFrame):
        table.columns.name = name

    prev_option = pd.get_option('display.float_format')
    if fmt is not None:
        pd.set_option('display.float_format', lambda x: fmt.format(x))

    display(table)

    if fmt is not None:
        pd.set_option('display.float_format', prev_option)

def plot_cummulative_ex_returns(daily_rets, benchmark, reb_freq=1, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(18,9))

    slice_cols = [col for col in daily_rets.columns if col.startswith('S')]
    cum_ret = daily_rets.add(1).cumprod()
    cum_ret['top_bottom'].plot(lw=2.5, ax=ax, color='red')
    cum_ret[benchmark].plot(lw=1.5, ax=ax, color='black', alpha=0.2)
    cum_ret[slice_cols].plot(lw=1.5, ax=ax, cmap=cm.RdYlGn_r)

    ax.legend(loc='upper left')
    ymin, ymax = cum_ret.min().min(), cum_ret.max().max()
    ax.set(
        ylabel=SLICE_EX_RET.get("YLABEL"),
        title=SLICE_EX_RET.get("TITLE").format(reb_freq),
        ylim=(ymin, ymax)
    )

    ax.set_yscale('symlog', linthresh=1)
    ax.set_yticks(np.linspace(ymin, ymax, _y_tick_num(ymax)))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.axhline(1.0, linestyle='-', color='black', lw=1)

    return fig

def _y_tick_num(ymax):
    if ymax > 1e2:
        return 4
    elif ymax > 50:
        return 5
    elif ymax > 30:
        return 6
    elif ymax > 5:
        return 7
    else:
        return 8



# 拥挤度分析
# cro = util.crowd_analize(f,stk_pool='top2500')
def crowd_analize(f,stk_pool='top2500'):
    # 2024.6.11 ljn
    '''
    计算10档组合拥挤度函数
    f：标准因子值tn矩阵
    stk_pool：股票池
    '''
    
    factor = normalize_by_row(f)
    # 获取“隔夜收益率因子”（t日收盘可以计算得到的最新t-1日隔夜收益率）
    night = pd.read_feather('/home/i_ljn/ljn/feather_fold/隔夜收益率.feather').set_index('date')
    
    # 获取对应股票池的股票，由于目前股票池TN矩阵只更新到2023-10-31，所以需要进行ffill操作（样本外测试）
    stk_pool_tn = get_stk_pool(stk_pool).reindex(index=f.index).fillna(method='ffill')
    factor = factor[stk_pool_tn]
    night_ret = night[stk_pool_tn]
    # 适应样本内和样本外的因子值矩阵
    night_ret = night_ret.reindex(index=f.index)
    # 对因子值按行排序
    f_rank = factor.rank(axis=1)
    
    # 逐10档组合计算拥挤度
    def calc_crowd(s):
        ind = f_rank.apply(lambda x: (x>(len(x.dropna())*(s-1)/10)) & (x<(len(x.dropna())*s/10)),axis=1)
        ind_short = ind.shift(1)
        # t-1日隔夜收益率转换为t日隔夜收益率，再进行计算
        Si_long = night_ret.shift(-1)[ind].mean(axis=1)
        Si_short = night_ret.shift(-1)[ind_short].mean(axis=1)
        Si_crowd = (Si_long - Si_short)*242
        
        return Si_crowd.rename(f'S{s}')
    
    results = Parallel(n_jobs=32)(delayed(calc_crowd)(s) for s in tqdm(range(1,11), desc="逐档组合计算因子拥挤度"))
    df = pd.concat(results, axis=1)
    df = drop_out_by_row(df.T, method='drop', threshold=4).T.fillna(method='ffill')
    print('因子值拥挤度(回测区间内最近半年)：')
    display(df.tail(120).mean().rename('因子拥挤度'))
    
    # 画图
    plt.figure(figsize=(10, 5))
    res_list = []
    for i,s in enumerate(df.columns):
        c_ret = df[s].rolling(120).mean()
        plt.plot(factor.index, c_ret,color=(0.1*i, 1-0.1*i, 0), label=f'S{i+1}') 
        s = pd.Series(c_ret,index=factor.index).rename(f'S{i+1}')
        res_list.append(s)
    
    plt.legend()
    plt.title('Factor crowding degree')
    plt.xlabel('dates')
    plt.ylabel('Order execution efficiency')
    
    # 显示图形
    plt.show()

    return df



# ic分析
# ic_res = util.ic_analysis_(f, ic_modes=['rank'],ret_modes=['stage'], 
#                  trade_modes=['trade_vwap_allday'], stk_pools=['all', 'hs300', 'csi500', 'csi800', 'csi1000', 'top2500'],
#                   sides=['both'], horizons=[1, 2, 5, 10, 20, 40, 60], ind = None, factors={})

def ic_analysis_(f, n=32, ic_modes=['rank', 'linear'],ret_modes=['stage', 'cum'], 
                 trade_modes=['trade_vwap_allday', 'trade_vwap_30min', 'trade_open'], stk_pools=['all', 'hs300', 'csi500', 'csi800', 'csi1000'],
                  sides=['both', 'up', 'down'], horizons=[1, 2, 5, 10, 20, 40, 60], ind = 'ind1', factors={'size': 0}):
    """ IC 分析
    参数
    ----------
        n : int | optional
            并行使用cpu数量，默认为32
        ic_mode : str | optional
            ic类型，可选值为['rank', 'linear']，默认为['rank', 'linear']
        ret_mdoe : str | optional
            未来收益率计算方式，可选值为['stage', 'cum']，默认为['stage', 'cum']
        trade_mode : str | optional
            收益率计算方式，可选值为['trade_vwap_allday', 'trade_vwap_30min', 'trade_open', 'trade_close']，默认为['trade_vwap_allday', 'trade_vwap_30min', 'trade_open']
        stk_pools : str | optional
            股票池范围，可选值为['all', 'hs300', 'csi500', 'csi800', 'csi1000']，默认为['all', 'hs300', 'csi500', 'csi800', 'csi1000']
        horizons : tuple | optional
            目标收益的计算周期, 默认为 [1, 2, 5, 10, 20, 40, 60]
        sides : str | optional
            ic的多空范围，默认为['both', 'up', 'down']
        ind : str | optional
            行业中性化基于的行业分类，默认为'ind1'
        factors ： dict | optional
            选择进行的其他中性化因子，默认为{'size': 0}，如果不做市值中性化请传入factor={}
    """

    # 将参数转换为列表
    ic_modes = ensure_list(ic_modes)
    ret_modes = ensure_list(ret_modes)
    trade_modes = ensure_list(trade_modes)
    stk_pools = ensure_list(stk_pools)
    sides = ensure_list(sides)
    # 确保factors类型为字典
    factors = ensure_dict(factors)
    # 确保horizons类型为元组
    horizons = ensure_tuple(horizons)
    # 获取所需要的股票池tn矩阵
    stk_pool_tns = {}
    for stk_pool in stk_pools:
        stk_pool_tns[stk_pool] = get_stk_pool(stk_pool)

    res_rets_cum = {}
    res_rets_stage = {}
    if ind == 'ind1' and list(factors.keys()) == ['size']:
        for t_m in trade_modes:
            for period in horizons:  
                for ret_mode in ret_modes:
                    if ret_mode == 'stage':
                        target = fm.get(f'res_ret_{t_m[6:]}_{period}')
                        res_rets_stage[t_m + '_' + str(period)] = target
                    elif ret_mode == 'cum':
                        target = fm.get(f'res_ret_{t_m[6:]}_{period}_cum')
                        res_rets_cum[t_m + '_' + str(period)] = target
    # 生成所有参数的组合
    param_combinations = list(itertools.product(ic_modes, ret_modes, trade_modes, stk_pools, sides))
    
    # 使用 joblib 并行运行
    # return calc_ic(f,'rank','stage','trade_vwap_allday','all','both','5')
    results = Parallel(n_jobs=n, backend='multiprocessing')(delayed(calc_ic)(f, stk_pool_tns, horizons,ind, factors,res_rets_cum,res_rets_stage, p1, p2, p3, p4, p5) for p1, p2, p3, p4, p5 in tqdm(param_combinations, desc="计算ic"))

    ic_res = pd.concat([sublist[0] for sublist in results])
    ic_res.columns = f.index
    ic_res = ic_res.sort_index()
    ic_res.columns = ic_res.columns.astype(str)

    turnover_res = pd.concat([sublist[1] for sublist in results])
    turnover_res.columns = f.index
    turnover_res = turnover_res.sort_index()
    turnover_res.columns = turnover_res.columns.astype(str)
    
    return [ic_res, turnover_res]

def calc_ic(f, stk_pool_tns, horizons,ind, factors,res_rets_cum,res_rets_stage, ic_mode, ret_mode, trade_mode, stk_pool, side):
    """ IC 分析
    参数
    ----------
        f : dataframe
            需要进行ic分析的矩阵
        stk_pool_tns : list 
            不同股票池tn矩阵的列表
        horizons : tuple | optional
            目标收益的计算周期, 默认为 [1, 2, 5, 10, 20, 40, 60]
        ind : str | optional
            行业中性化基于的行业分类，默认为'ind1'
        factors ： dict | optional
            选择进行的其他中性化因子，默认为{'size': 0}，如果不做市值中性化请传入factor={}   
        res_rets : list 
            不同参数的残差收益率tn矩阵的列表
        ic_mode : str | optional
            ic类型，不能为列表，需要为单个类型的字符串
        ret_mdoe : str | optional
            未来收益率计算方式，不能为列表，需要为单个类型的字符串
        trade_mode : str | optional
            收益率计算方式，不能为列表，需要为单个类型的字符串
        stk_pool : str | optional
            股票池范围，不能为列表，需要为单个类型的字符串
        side : str | optional
            ic的多空范围，不能为列表，需要为单个类型的字符串
    """
    # 判断单边&双边
    if side == 'up':
        f_n = normalize_by_row(f)
        fac = f[f_n>0]
    elif side == 'down':
        f_n = normalize_by_row(f)
        fac = f[f_n<0]
    else:
        fac = f
    # 进行ic_mode判断
    fac = fac.rank(axis=1) if ic_mode == 'rank' else fac
    # 获取股票池tn矩阵
    stk_pool_tn = stk_pool_tns[stk_pool]
    fac = fac[stk_pool_tn]
    # 获取未来不同周期的收益率，并计算对应horizon的ic
    ic_table = pd.DataFrame(index=horizons,columns=fac.index)
    turnover_table = pd.DataFrame(index=horizons,columns=fac.index)
    for i,horizon in enumerate(horizons):
        # 获得未来收益率的起始位置和终止位置，进行ret_mode判断
        start = 0 if i == 0 else (horizons[i-1] if ret_mode == 'stage' else 0)
        end = horizon
        # 如果进行行业、市值中性化则直接从factorbase中提取数据，否则需要现场计算
        if ind == 'ind1' and list(factors.keys()) == ['size']:
            if ret_mode == 'stage':
                target = res_rets_stage[trade_mode + '_' + str(horizon)][stk_pool_tn]
            elif ret_mode == 'cum':
                target = res_rets_cum[trade_mode + '_' + str(horizon)][stk_pool_tn]
        else:
            target = get_forward_returns(start, end, trade_mode, ind, factors)[stk_pool_tn]
        # 如果计算rankic，则需要将target转换为截面排名
        target = target.rank(axis=1) if ic_mode=='rank' else target
        # 计算ic
        ic_table.loc[horizon] = row_correlation_matrix(fac.values,target.values)
        turnover_table.loc[horizon] = calc_turnover_das(fac, horizon)

    
    # 创建 MultiIndex
    index = pd.MultiIndex.from_product(
        [[ic_mode], [ret_mode], [trade_mode], [stk_pool], [side], horizons],
        names=['ic_mode', 'ret_mode', 'trade_mode', 'stk_pool', 'side', 'horizon']
    )
    # 生成multiindex_dataframe
    ic_df = pd.DataFrame(ic_table.values, index=index, columns=fac.index)
    turnover_df = pd.DataFrame(turnover_table.values, index=index, columns=fac.index)
    
    return [ic_df, turnover_df]

def get_res_ret(trade_mode='trade_vwap_allday', periods = 1):
    
    target = fm.get(f'res_ret_{trade_mode[6:]}_{periods}')
    
    return target    

def get_forward_returns(start = 0, end = 1, trade_mode='trade_vwap_allday', ind = None, factors={}):
    """ 获取未来收益率
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
    assert trade_mode in ['trade_close', 'trade_open', 'trade_vwap_allday', 'trade_vwap_30min']

    cumadj = FactorMgr().get('cumadj')
    adj_close = FactorMgr().get('close') * cumadj
    if trade_mode == 'trade_close':
        target = adj_close.shift(-end) / adj_close.shift(-start) - 1
    elif trade_mode == 'trade_open':
        adj_open = FactorMgr().get('open') * cumadj
        target = adj_open.shift(-end-1) / adj_open.shift(-start-1) - 1
    elif trade_mode == 'trade_vwap_allday':
        vwap_allday = FactorMgr().get('vwap') * cumadj
        target = vwap_allday.shift(-end-1) / vwap_allday.shift(-start-1) - 1
    elif trade_mode == 'trade_vwap_30min':
        vwap_allday = FactorMgr().get('vwap_1000') * cumadj
        target = vwap_allday.shift(-end-1) / vwap_allday.shift(-start-1) - 1
        
    # 如果ind和fators均为空，不做任何处理
    if (ind is None) and len(factors)==0:
        return target
    else:
        ind_df = FactorMgr().get(ind)
        target = neutralize(target, ind=ind_df, factors=factors)
        return target


def calc_turnover_das(f, horizons):
    """
    计算因子矩阵的das换手率，das指diff_abs_sum
    """
    # 权重化：先截面标准化，然后除以截面绝对值之和
    # stk_pool_tn = get_stk_pool(stk_pool)
    # f1 = f[stk_pool_tn] # 过滤股票池
    f1 = normalize_by_row(f) # 截面标准化
    f1 = f1.divide(f1.abs().sum(axis=1), axis=0) # 截面归一化
    # res = pd.DataFrame(index=f1.index, columns=horizons, dtype=np.float64)
    # for h in tqdm(horizons, desc="计算各horizon上的res"):
    turnover = f1.diff(periods=horizons).abs().sum(axis=1) / 2# das换手率
    return turnover


def double_sort(f, n, factor = {}):
    """
    Func: 因子双重排序分析
    Input:
        factors为需要进行双重排序分析的风格特征因子，传递为字典，默认为空字典
        n为进行双重排序的n分位数
    Output:
        展示用的dataframe
    """
    cap = pd.read_feather('/home/i_ljn/ljn/feather_fold/cap.feather').set_index('date')
    factor['ln_cap'] = np.log(cap)
    # 排序，获取风格因子名字列表
    f_rank = f.rank(axis=1)
    factor_names = list(factor.keys())
    
    # 逐n档组合计算风格特征均值
    def calc_crowd(s):
        # 获取该档组合示性矩阵
        ind = f_rank.apply(lambda x: (x>(len(x.dropna())*(s-1)/n)) & (x<(len(x.dropna())*s/n)),axis=1)
        # 对每个风格特征先计算截面均值，再计算时序均值
        factor_list = []
        for name in factor_names:
            series_s = factor[name][ind].mean(axis=1)
            factor_list.append(series_s)
        
        return factor_list
    
    results = Parallel(n_jobs=10)(delayed(calc_crowd)(s) for s in tqdm(range(1,n+1), desc="逐档组合计算风格特征均值"))
    # 得到双重排序1至n档特征风格均值的dataframe
    values_s = [[df.mean() for df in results[i]] for i in range(len(results))]
    df = pd.DataFrame(values_s,index=range(1,n+1),columns=factor_names).T
    # 计算high-low、t统计量、p值
    hl = []
    t_list = []
    p_list = []
    # 循环每个特征风格因子
    for i,name in enumerate(factor_names):
        # 得到最高档和最低档的序列
        high = results[n-1][i].dropna()
        low = results[0][i].dropna()
        # 计算high-low、t统计量、p值
        t_stat, p_val = stats.ttest_ind(high, low)
        hl.append(high.mean() - low.mean())
        t_list.append(round(t_stat,2))
        p_list.append(round(p_val,2))
    df['high-low'] = hl
    df['t_stat'] = t_list
    df['p_value'] = p_list
    display(df)

    return df

def gen_monthly_ret(res_slice):
    tmp = pd.DataFrame(index=res_slice['ex_ret'].index, data=(1 + res_slice['ex_ret']['S10']).cumprod())
    mth_ret = ru.get_monthly_ret(tmp)
    mth_rpt = mth_ret.pivot_table(index='year', columns='mth', values='S10')
    mth_rpt['年度'] = (1+mth_rpt).prod(axis=1) - 1
    display(mth_rpt.applymap(lambda x:(f"{x*100:.2f}%")))

def gen_summary(f, coverage, res_slice):
    valid_dates = coverage.replace(0,np.nan).dropna().index
    
    # 评估摘要
    dates = res_slice['ex_ret'].index
    print(f"评估期: {dates[0].date()} - {dates[-1].date()}")
    # 逆序数: 至多1对
    AnnRet_slice = (res_slice['ex_ret']+1).loc[valid_dates].rolling(242).apply(np.prod, raw=True).iloc[::-242] - 1
    inverse_num_list = []
    for i in range(len(AnnRet_slice)):
        inverse_num = count_inversions(AnnRet_slice.iloc[i].iloc[:-2])
        inverse_num_list.append(inverse_num)
    inverse_num_mean = round(np.mean(inverse_num_list),1)
    print('slice逆序数:      ' + str(inverse_num_mean))
    # 1日换手：低于35%
    top_wgt = pd.DataFrame(res_slice['top_wgt'],index=f.index)
    top_wgt_array = top_wgt.loc[valid_dates].values
    top_turn_over = np.mean(np.sum(np.abs(np.diff(top_wgt[800:], axis=0)), axis=1)) / 2
    print('S10的1日换手:      ' + f'{round(top_turn_over*100,1)}%')
    # S10费后年化超额：不低于6%
    S10_ex_ret_net = AnnRet_slice.mean()['S10'] - top_turn_over*0.24
    print('S10费后年化收益:   ' + f'{round(S10_ex_ret_net*100,2)}%')
    # S10费前超额夏普：不低于0.7
    S10_ex_sr = res_slice['perf_ind'].loc['S10', 'SR'] 
    print('S10费前超额夏普:   ' + f'{round(S10_ex_sr,2)}')
    # S10近年表现：不低于5%
    S10_ex_ret_recent = AnnRet_slice.iloc[:4].mean()['S10'] - top_turn_over*0.24
    print('S10近年费后超额:   ' + f'{round(S10_ex_ret_recent*100,2)}%')
    # 评估摘要
    dates = res_slice['ex_ret'].index
    print(f"评估期: {dates[0].date()} - {dates[-1].date()}")

def gen_report(ic_df, turnover_df, horizons=[1, 2, 5, 10, 20, 40, 60], start_date='2010-01-04', end_date='2022-12-30', is_ewm = True):
    """ 生成 IC 均值、标准差和ICIR
    参数：
    ----------
        ic_df: dataframe
            Multi-index IC 数值
        horizons: list | optional
            评估时的目标收益计算周期, 默认为 [1, 2, 5, 10, 20, 40, 60]
        start_date: str | optional
            开始日期，选择回测周期内的日期
        end_date: str | optional
            结束日期，选择回测周期内的晚于start_date的日期
    输出：
    ----------
        ic_stats: dataframe
    """
    # 确保输入的日期在合理区间
    first_date = ic_df.columns[0]
    last_date = ic_df.columns[-1]
    if end_date > last_date:
        end_date = last_date
    if start_date < first_date:
        start_date = first_date


    if is_ewm:
        # 计算ewm版本的均值、标准差和ICIR
        ic_ewm_mean = ic_df.T.ewm(halflife=242*4).mean().iloc[-1]
        ic_ewm_stdev = ic_df.T.ewm(halflife=242*4).std().iloc[-1]
        ewm_mean_std = ic_ewm_mean / ic_ewm_stdev
        ewm_ic_ir = ewm_mean_std * np.sqrt(242/np.tile(np.array(horizons), int(len(ic_df)/len(horizons))))
        s = int(ic_df.shape[0] / 7)
        turnover_ewm_mean = turnover_df.T.ewm(halflife=242*4).mean().iloc[-1] / (s*[1, 2, 5, 10, 20, 40, 60])
        score = (4.8*ic_ewm_mean - 0.34*turnover_ewm_mean)
        # 生成包含所有统计指标的dataframe
        report = pd.DataFrame({
            'ewm_mean': round(ic_ewm_mean /  (int(ic_df.shape[0] / len(horizons))*[1, 1, np.sqrt(3), np.sqrt(5), np.sqrt(10), np.sqrt(20), np.sqrt(20)]),3),
            'ewm_stdev': round(ic_ewm_stdev,3),
            'mean/stdev': round(ewm_mean_std,2),
            'ewm_ICIR': round(ewm_ic_ir,2),
            'ewm_turnover_mean': round(turnover_ewm_mean, 3),
            'ewm_score': round(score,3)
        })
    else:
        # # 计算均值、标准差和ICIR
        ic_mean = ic_df.loc[:, start_date:end_date].mean(axis=1)
        ic_stdev = ic_df.loc[:, start_date:end_date].std(axis=1)
        mean_std = ic_mean / ic_stdev
        ic_ir = mean_std * np.sqrt(242/np.tile(np.array(horizons), int(len(ic_df)/len(horizons))))

        turnover_mean = turnover_df.loc[:, start_date:end_date].mean(axis=1)/ (s*[1, 2, 5, 10, 20, 40, 60])
        score = (4.8*ic_mean - 0.34*turnover_mean)

        # 生成包含所有统计指标的dataframe
        report = pd.DataFrame({
            'mean': ic_mean /  (int(ic_df.shape[0] / len(horizons))*[1, 1, np.sqrt(3), np.sqrt(5), np.sqrt(10), np.sqrt(20), np.sqrt(20)]),
            'stdev': ic_stdev,
            'mean/stdev': mean_std,
            'ICIR': ic_ir,
            'turnover_mean': turnover_mean,
            'score': score
        })
    return report