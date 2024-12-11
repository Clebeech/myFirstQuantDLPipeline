import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import numpy as np

# 读取数据
df = pd.read_csv('/Users/tigerxu/Desktop/mirror/re/evaluation/results/h1_ic.csv')
baseline_df = pd.read_csv('/Users/tigerxu/Desktop/mirror/re/evaluation/results/baseline.csv')

# 获取日期列（除去前7列非日期数据）
date_columns = df.columns[7:]
dates = pd.to_datetime(date_columns)

# 创建图形
plt.figure(figsize=(15, 8))

# 设置样式
sns.set_style("whitegrid")

# 颜色和标签设置
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# 绘制所有组合的累积IC
combinations = [
    ('linear', 'all'),
    ('linear', 'csi500'),
    ('rank', 'all'),
    ('rank', 'csi500')
]

for i, (ic_mode, stk_pool) in enumerate(combinations):
    data = df[(df['ic_mode'] == ic_mode) & 
              (df['ret_mode'] == 'cum') & 
              (df['stk_pool'] == stk_pool)].iloc[:, 7:].values
    if len(data) > 0:
        cum_ic = np.cumsum(data[0])
        plt.plot(dates, cum_ic, label=f'{ic_mode}-{stk_pool}', 
                color=colors[i], linewidth=2)

# 添加baseline（调整到相同的时间范围）
baseline_dates = pd.to_datetime(baseline_df.columns[7:])
baseline_data = baseline_df[(baseline_df['ic_mode'] == 'linear') & 
                          (baseline_df['ret_mode'] == 'cum') & 
                          (baseline_df['stk_pool'] == 'all')].iloc[:, 7:].values
if len(baseline_data) > 0:
    # 找到与我们数据时间范围对应的baseline数据点
    start_idx = np.where(baseline_dates >= dates[0])[0][0]
    end_idx = np.where(baseline_dates <= dates[-1])[0][-1] + 1
    baseline_dates = baseline_dates[start_idx:end_idx]
    baseline_data = baseline_data[0][start_idx:end_idx]
    # 重新计算累积IC
    baseline_cum_ic = np.cumsum(baseline_data)
    plt.plot(baseline_dates, baseline_cum_ic, label='baseline', 
            color=colors[-1], linewidth=2, linestyle='--')

# 设置图表格式
plt.title('Cumulative IC Comparison', fontsize=14, pad=15)
plt.xlabel('date', fontsize=12)
plt.ylabel('cum_ic', fontsize=12)
plt.legend(fontsize=10, loc='best')  # 让matplotlib自动选择最佳图例位置
plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('/Users/tigerxu/Desktop/mirror/re/evaluation/results/cumulative_ic_comparison.png', 
            dpi=300, bbox_inches='tight')
