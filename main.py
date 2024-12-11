import torch
import os
import pickle
from data import get_market_data, prepare_dl_dataset_fast
from rolling_train import rolling_train, LSTM
from test import predict, evaluate_predictions

def main():
    # 1. 设置参数
    start_date = '2012-01-01'
    end_date = '2022-12-31'
    train_days = 252  # 约一年的交易日
    test_days = 63   # 约一季度的交易日
    batch_size = 256
    epochs = 10
    
    # 2. 创建必要的目录
    for directory in ['models', 'results', 'evaluation']:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # 3. 执行滚动训练
    print("开始滚动训练...")
    window_results = rolling_train(
        start_date=start_date,
        end_date=end_date,
        train_days=train_days,
        test_days=test_days,
        batch_size=batch_size,
        epochs=epochs
    )
    
    # 4. 保存训练结果
    with open('results/window_results.pkl', 'wb') as f:
        pickle.dump(window_results, f)
    
    # 5. 加载数据用于预测
    print("准备预测数据...")
    data, stock_ids = get_market_data(start_date=start_date, end_date=end_date)
    
    # 6. 生成预测矩阵
    print("生成预测矩阵...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictions_df = predict(data, stock_ids, window_results, device=device)
    
    # 7. 保存预测结果
    predictions_df.to_csv('results/predictions.csv')
    
    # 8. 评估预测结果
    print("评估预测结果...")
    evaluation_results = evaluate_predictions(
        predictions_df,
        output_folder='./evaluation',
        stk_pools=['all', 'hs300', 'csi500'],
        ic_modes=['rank', 'linear'],
        ret_modes=['stage', 'cum'],
        horizons=[1, 2, 5, 10, 20, 40, 60],
        trade_modes=['open'],
        sides=['both'],
        pcts=[1.0]
    )
    
    # 9. 保存评估结果
    with open('evaluation/evaluation_results.pkl', 'wb') as f:
        pickle.dump(evaluation_results, f)
    
    print("训练和评估完成！")

if __name__ == "__main__":
    main()