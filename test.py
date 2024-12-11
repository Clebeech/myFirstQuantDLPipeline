import torch
import os
import pickle
from data import get_market_data, prepare_dl_dataset_fast
from rolling_train import rolling_train, LSTM, prepare_rolling_data
from evaluate import FcstEvaluator
import pandas as pd

def predict(data, stock_ids, window_results, device='cpu'):
    all_predictions = []  # 用于存储所有窗口的预测结果
    
    for result in window_results:
        window = result['window']
        model_path = result['model_path']
        
        # 加载模型
        model = LSTM()
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()
        
        # 获取测试数据，并包含向前40天的数据
        test_start_idx = data.index.get_loc(window['test_start'])
        history_start_idx = max(0, test_start_idx - 40)  # 确保不会超出数据范围
        test_end_idx = data.index.get_loc(window['test_end'])
        
        # 获取完整的数据片段
        full_data = data.iloc[history_start_idx:test_end_idx + 1]
        
        # 准备预测数据
        X_test, _ = prepare_dl_dataset_fast(full_data, window_size=40)
        
        # 进行预测
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).transpose(1, 2).to(device)
            predictions_window = model(X_test_tensor).cpu().numpy()
        
        # 只保留测试期的预测结果
        test_dates = data.index[test_start_idx:test_end_idx + 1]
        predictions_reshaped = predictions_window.reshape(-1, len(stock_ids))
        
        # 创建当前窗口的预测DataFrame
        window_df = pd.DataFrame(
            predictions_reshaped,
            index=test_dates,
            columns=stock_ids
        )
        
        all_predictions.append(window_df)
    
    # 合并所有窗口的预测结果
    predictions_df = pd.concat(all_predictions)
    predictions_df = predictions_df.sort_index()
    
    return predictions_df

def evaluate_predictions(predictions_df, output_folder='./evaluation', 
                       stk_pools=['all', 'hs300', 'csi500'],
                       ic_modes=['rank', 'linear'],
                       ret_modes=['stage', 'cum'],
                       horizons=[1, 2, 5, 10, 20, 40, 60],
                       trade_modes=['open'],
                       sides=['both'],
                       pcts=[1.0],  # 添加默认的pct值
                       start_date=None,
                       end_date=None):
    """
    评估模型预测结果
    
    参数:
    predictions_df: pd.DataFrame
        预测结果DataFrame，index为日期，columns为股票代码
    output_folder: str
        结果输出文件夹路径
    stk_pools: list
        要评估的股票池列表，可选值为['all', 'hs300', 'csi500', 'csi800', 'csi1000']
    ic_modes: list
        IC计算模式，可选值为['linear', 'rank']
    ret_modes: list
        收益率计算方式，可选值为['stage', 'cum']
    horizons: list
        收益率计算的时间跨度
    trade_modes: list
        交易模式，可选值为['close', 'open', 'vwap', 'vwap_1000']
    sides: list
        多空方向，可选值为['up', 'down', 'both']
    pcts: list
        百分位数列表，当sides不为'both'时使用
    start_date: str
        评估起始日期，格式为'YYYY-MM-DD'
    end_date: str
        评估结束日期，格式为'YYYY-MM-DD'
    
    返回:
    dict: 包含评估结果的字典，包括：
        - ic_stats: IC统计信息
        - combined_stats: 合并后的统计信息
    """
    # 如果未指定日期范围，使用预测数据的日期范围
    if start_date is None:
        start_date = predictions_df.index.min().strftime('%Y-%m-%d')
    if end_date is None:
        end_date = predictions_df.index.max().strftime('%Y-%m-%d')
    
    # 创建评估器实例
    evaluator = FcstEvaluator(
        output_folder=output_folder,
        pred_horizons=[1],  # 这里设置为1因为我们只有一个预测周期
        n_jobs=20
    )
    
    # 设置评估参数
    evaluator.set_up(
        h=1,  # 预测周期为1
        preds=predictions_df,
        stk_pools=stk_pools,
        ic_modes=ic_modes,
        ret_modes=ret_modes,
        horizons=horizons,
        trade_modes=trade_modes,
        sides=sides,
        pcts=pcts  # 使用指定的百分位数
    )
    
    # 进行IC分析
    ic_results = evaluator.ic_analysis_()
    
    # 生成IC统计信息
    ic_stats = evaluator.gen_ic_stats(start_date=start_date, end_date=end_date)
    
    # 处理所有组合的结果
    combined_stats = evaluator.process_all_combinations(start_date=start_date, end_date=end_date)
    
    return {
        'ic_stats': ic_stats,
        'combined_stats': combined_stats
    }