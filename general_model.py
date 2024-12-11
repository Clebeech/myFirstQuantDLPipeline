import sys
import os
import gc
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.model_selection import GridSearchCV

from ml_lib import *
import matplotlib.pyplot as plt


# ============================================
#                general_model
# ============================================
class general_model:
    
    def __init__(self, general_setting, output_folder, input_folder, start_date=None, end_date=None):
        """初始化 general_model 实例，包含基本的模型配置和设置。
        general_setting 所需要的参数没有完全统计，需要根据具体的模型需求进行调整。
        参数:
            general_setting (dict): 模型的配置参数。
            output_folder (str): 存储模型输出的文件夹路径。
            input_folder (str): 输入数据的文件夹路径。
            start_date (str): 训练数据的开始日期，格式为 "YYYY-MM-DD"。
            end_date (str): 训练数据的结束日期，格式为 "YYYY-MM-DD"。
        返回:
            None
        """
        # 将 general_setting 中的每个键值对作为类属性
        for key, value in general_setting.items():
            setattr(self, key, value)
        self.model_params = {}  # 模型各期参数
        self.checkpoint_file = None  # 用于保存检查点文件的路径
        self.output_folder = output_folder
        self.input_folder = input_folder

        self.start_date = start_date
        self.end_date = end_date

        # 设置模型输出路径
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        # 保存模型参数
        save_dict_to_yaml(general_setting, self.output_folder + 'general_setting.yml')
        
        # 从FactorMgr的close dataframe获取TN矩阵框架
        close_ = fm.get('close')
        self.dates = close_.index
        self.stk_ids = close_.columns
        
        # 日期选择功能
        if start_date is not None and end_date is not None:
            dates_df = pd.DataFrame(self.dates, columns=['date'])
            filtered_dates = dates_df[(dates_df['date'] >= start_date) & (dates_df['date'] <= end_date)]
            filtered_dates_index = pd.DatetimeIndex(fildered_dates['date'])
            self.dates = filtered_dates_index

        # 获取训练日列表 self.train_dates: list
        self.train_dates = q.get_eoq_tdates(self.dates[0], self.dates[-1])
        
        # 预测值容器 self.fcst: TN dataframe
        self.fcst = pd.DataFrame(index=self.dates, columns=self.stk_ids, dtype=np.float64)

        # 设置训练日志
        logging_config(self.output_folder)
    
    def return_calculation(self):
        """
        计算未来 h 天的复权后加权平均价格 (VWAP) 收益率，
        并按指定方法对收益率数据进行平滑化、行业中性化及截面标准化处理。

        参数:
            无。

        使用的内部变量:
            self.ret_type (str): 收益率计算类型，默认值为 "vwap"。
            self.h (int): 收益率的未来天数窗口。
            self.ret_preprocess_method (str): 行业中性化的选项，'ind' 代表仅行业中性化，'ind+size' 为行业及市值中性化。
            self.normalize_by_row (bool): 是否对收益率数据进行截面标准化。

        处理步骤:
            1. 获取复权调整后的 VWAP 数据并计算未来 h 天的收益率。
            2. 使用平方根去量纲操作处理收益率。
            3. 根据配置对收益率数据进行平滑处理（如指定了移动平均参数）。
            4. 应用行业中性化（根据配置）。
            5. 对收益率数据进行截面标准化（如启用该选项）。

        输出:
            更新类属性:
            self.ret (pd.DataFrame): 计算后的 t x n 矩阵，包含处理后的未来 h 天收益率数据。
            self.smooth_ret (pd.DataFrame): 平滑化后的收益率数据，根据配置而定。
        """
        # 从 FactorMgr 获取复权调整因子、市值和行业数据
        cumadj = fm.get('cumadj')
        cap = fm.get('cap')
        ind1 = fm.get('ind1')
        
        # 从FactorMgr中获取复权调整后的vwap
        ret_type = self.ret_type
        target = fm.get(ret_type)
        target_adj = target * cumadj
        
        # 计算未来 h 天的收益率，shift 操作为未来天数的收益率计算
        self.ret = target_adj.shift(-(self.h+1)) / target_adj.shift(-1) - 1

        # 对y做平滑
        def y_smoothing():
            if self.Y_moving_average and self.Y_moving_average[0]>1:
                self.ret = calculate_y_rolling_mean(ret=self.ret,
                                                    average_type=self.Y_moving_average[1],
                                                    window_size=self.Y_moving_average[0])
        # 中性化
        def y_neutralize():
            if self.ret_preprocess_method == 'ind':
                self.ret = neutralize(self.ret, ind=ind1)
            elif self.ret_preprocess_method == 'ind+size':
                self.ret = neutralize(self.ret, ind=ind1, factors={'size': np.log(cap)})
        
        # 截面标准化
        def y_normalize():
            if self.normalize_by_row:
                self.ret = normalize_by_row(self.ret)

        # 根据需求调整平滑、中性化和截面标准化的顺序
        y_preprocess_map = {'smoothing': y_smoothing,
                            'neutralize': y_neutralize,
                            'normalize': y_normalize,
                           }
        for operation in self.y_preprocessing_sequence:
            y_preprocess_map.get(operation, lambda: None)()

    def reduce(self):
        """使用降维算法对筛选后的因子数据进行降维处理。

        参数:
            无。

        使用的内部变量:
            self.dimensionality_reduction_params (dict): 降维参数，包括：
                - 'method' (str): 降维方法名称。
                - 'target_dim' (int): 目标维度数。
                - 'fillna_strategy' (str): 缺失值填充策略。
            self.filtered_train_nk (dict): 筛选后的因子数据字典，每个键为日期，值为包含因子和收益率的 DataFrame。

        处理步骤:
            1. 检查 `dimensionality_reduction_params` 是否已设置，如果未启用降维功能则跳过降维。
            2. 对每个交易日的数据应用降维处理：
               - 使用指定方法对数据进行降维（如 PCA）。
               - 填充缺失值（根据 `fillna_strategy`）。
               - 保留收益率列，将降维后的因子数据与收益率列重新组合。
            3. 更新 `self.filtered_train_nk`，将降维处理后的数据替换原数据。

        输出:
            更新类属性:
            self.filtered_train_nk (dict): 每个日期的数据都降维后的因子数据字典，保留收益率列。
        """

        if self.dimensionality_reduction_params is None:
            logging.info("降维功能未启用，跳过降维步骤")
            return

        self.reduced_data = {}  # 存储降维后的数据
        for key, data in self.filtered_train_nk.items():
            reducer = DimensionalityReducer(
                method=self.dimensionality_reduction_params.get("method", "pca"),
                target_dim=self.dimensionality_reduction_params.get("target_dim", 10),
                fillna_strategy=self.dimensionality_reduction_params.get("fillna_strategy", "zero")
            )
            ret = data['ret']
            data = data.drop('ret', axis=1)
            result = reducer.fit_transform(data)
            self.reduced_data[key] = pd.concat([result, ret], axis=1)
        logging.info("降维完成")
        self.filtered_train_nk = self.reduced_data
    
    def data_loading(self):
        """
        逐日加载输入的因子数据并构建 股票 x 因子 (n x k) 矩阵 (self.train_nk)。 
        对加载的数据进行平滑化处理（若指定了平滑参数），并最终将收益率数据合并到因子矩阵中。

        参数:
            无。

        使用的内部变量:
            self.input_folder (str): 因子数据文件的输入路径。
            self.dates (list of datetime): 每个交易日的日期列表。
            self.X_moving_average (list or None): 对因子数据 (X) 进行平滑处理的参数，包括窗口大小和平均类型。
            self.ret (pd.DataFrame): 计算的未来收益率数据，将合并到因子数据中。

        处理步骤:
            1. 逐日读取因子数据文件，生成包含每个日期的因子矩阵。
            2. 根据配置（如指定了平滑参数）对因子数据进行平滑处理。
            3. 合并收益率数据 (ret) 和因子矩阵，生成带收益率标签的训练数据。
            4. 释放冗余内存并将日期和股票代码设置为多重索引。
        
        输出:
            更新类属性:
            self.train_nk (dict): 包含逐日加载的 n x k 矩阵，每个键为日期，值为 pd.DataFrame。
            每个 DataFrame 的列为因子和收益率，索引为股票代码和日期。
        """
        
        # 初始化空字典，用于存储每个交易日的因子矩阵
        self.factor_nk = {}
        
        # 遍历每个日期，从文件中加载该日期的因子数据
        for t in tqdm(self.dates, file=tqdm_out, desc="逐日加载因子NxK矩阵", mininterval=60):
            self.factor_nk[t] = pd.read_feather(self.input_folder + f'{t.date()}.feather').set_index('stk_id') 

        # 如果指定了 X 的平滑参数，应用移动平均平滑
        if self.X_moving_average and self.X_moving_average[0]>1:
            self.factor_nk = calculate_X_rolling_mean(dates=self.dates,
                                                      factor_nk=self.factor_nk,
                                                      average_type=self.X_moving_average[1],
                                                      window_size=self.X_moving_average[0])
        
        # 对于train_nk dict中的每一个t, 合并y和X到同一个DataFrame
        self.train_nk = merge_returns_dict(self.factor_nk, self.ret)
        
        #删除冗余的变量释放内存
        del self.factor_nk
        gc.collect()

        # 为训练数据设置多重索引，索引为股票代码和日期
        # 2-level indexed dataframe: index(stk_id, date) | factors ... | ret
        for date in tqdm(self.dates[:], desc = '设置stk_id和date双索引: '):
            self.train_nk[date] = self.train_nk[date].assign(date=date).set_index('date', append=True)

    def select_target_pool(self):
        """
        根据指定的股票池筛选每个交易日的数据，去除其他股票的信息。
        在筛选后的数据上进行缺失值处理，并根据降维参数对数据进行降维处理。

        参数:
            无。

        使用的内部变量:
            self.target_stock_pool (str): 指定的股票池名称，可选值为 ['all', 'hs300', 'csi500', 'csi800', 'csi1000', 'top1500', 'top2000', 'top2500']。
            self.train_nk (dict): 每日的 n x k 因子数据字典，键为日期，值为 pd.DataFrame。
            
        处理步骤:
            1. 使用指定的股票池名称对因子数据进行筛选，仅保留目标股票池中的股票。
            2. 对筛选后的数据逐日删除完全为空的行和列，以及缺少收益率的行。
            3. 调用 `handle_na` 函数处理数据中的缺失值。
            4. 如果启用了降维功能，则对筛选后的数据进行降维处理。

        输出:
            更新类属性:
            self.filtered_train_nk (dict): 筛选后的因子数据字典，包含降维处理后的每个交易日数据。字典的键为日期，值为 pd.DataFrame。
        """
        # 筛选目标股票池的因子数据
        self.filtered_train_nk = filter_factors(self.train_nk, self.target_stock_pool)
        
        # 删除未筛选过的因子数据
        del self.train_nk
        gc.collect()

        # 对每个交易日的数据进行缺失值处理
        for t, data in tqdm(self.filtered_train_nk.items(), desc='逐日删除非指定股票池内股票并处理NA值'):
            data = data.dropna(subset=data.columns.difference(['ret']), how='all')
            data = handle_na(data)
            self.filtered_train_nk[t] = data

        # 筛选数据后,调用降维函数，对self.filtered_train_nk中的每个日期的DataFrame进行降维
        self.reduce()
    
    def train_data_preparation(self,i):
        """从train_nk字典中获取截止训练日t的前（t-train_size）天的dataframe, 去除h+1天未来信息后拼接在一起"""
        self.curr_train_dates = self.dates[i+1-self.train_size:i+1]
        
        # 去除h+1天的未来信息
        self.train_data = pd.concat([self.filtered_train_nk[date] 
                                     for date in self.curr_train_dates[:-(self.h+1)]]) 
        self.train_data = self.train_data.dropna(subset=['ret'])
        
        # 分割X和y
        self.X = self.train_data.drop(['ret'], axis=1)
        self.y = self.train_data[['ret']]

    
    def load_checkpoint(self):
        """
        加载之前的训练检查点。

        参数:
            无。

        使用的内部变量:
            self.checkpoint_file (str): 检查点文件的路径，格式为 "output_folder/h{h}_checkpoint.pkl"。
            self.model_params (dict): 每期训练的模型参数字典。
            self.fcst (pd.DataFrame): 预测结果的 DataFrame，索引为日期和股票代码。
            self.train_dates (list): 更新模型的日期列表。
            self.h (int): 模型未来收益率的预测天数。

        处理步骤:
            1. 检查 checkpoint 文件是否存在，且文件不为空。
            2. 如果文件存在且有效，使用 pickle 加载文件内容。
            3. 恢复模型参数 (model_params)、预测结果 (fcst)、训练日期 (train_dates)，并设置最后训练日期的最佳模型参数。
            4. 如果加载成功，返回最后一次训练的日期；否则返回 None。

        输出:
            str 或 None: 返回最后一次训练的日期。如果文件损坏或为空，返回 None。
        """
        self.checkpoint_file = self.output_folder + f"h{self.h}_checkpoint.pkl"
        if os.path.exists(self.checkpoint_file):
            if os.path.getsize(self.checkpoint_file) > 0:  # 检查文件是否不为空
                try:
                    with open(self.checkpoint_file, 'rb') as file:
                        checkpoint = pickle.load(file)
                    self.model_params = checkpoint['model_params']
                    self.fcst = checkpoint['fcst']
                    self.train_dates = checkpoint['train_dates']
                    self.best_model = self.model_params[checkpoint['last_date']]['best_trained_model'] # 意为：加载存档点最后一期模型参数，作为断点重训的初始参数
                    logging.info(f"恢复检查点，最后训练日期为 {checkpoint['last_date']}")
                    return checkpoint['last_date']
                except EOFError:
                    logging.error("检查点文件读取失败，文件可能已损坏。")
                    return None
            else:
                logging.error("检查点文件为空，无法加载。")
                return None
        else:
            logging.info("检查点文件不存在，跳过加载。")
            return None
        
    
    def save_checkpoint(self, t, model_params):
        """
        保存当前的训练进度和模型参数。

        参数:
            t (datetime): 当前训练日期，用于记录断点位置。

        使用的内部变量:
            self.checkpoint_file (str): 检查点文件的路径，格式为 "output_folder/h{h}_checkpoint.pkl"。
            self.model_params (dict): 每期训练的模型参数字典。
            self.fcst (pd.DataFrame): 预测结果的 DataFrame，索引为日期和股票代码。
            self.train_dates (list): 更新模型的日期列表。

        处理步骤:
            1. 将当前模型参数、预测结果、训练日期以及当前日期保存为字典格式。
            2. 使用 pickle 将字典保存到指定的 checkpoint 文件路径中。

        输出:
            无，保存操作仅在本地生成检查点文件。
        """
        checkpoint = {
            'model_params': model_params, # model_params是一个dict, 存储最优超参数下，每一期训练的模型参数
            'fcst': self.fcst,
            'train_dates': self.train_dates,
            'last_date': t
        }
        with open(self.checkpoint_file, 'wb') as file:
            pickle.dump(checkpoint, file)
        logging.info(f"保存检查点，当前日期为 {t.date()}")     

    
    def general_main_init(self):
        """
        初始化数据加载和整理的主流程。
        该方法包括计算收益率、加载因子数据、筛选目标股票池并处理缺失值，最终返回已筛选和处理的数据字典。

        参数:
            无。

        使用的内部变量:
            self.ret (pd.DataFrame or None): 如果已计算收益率，跳过收益率计算。
            self.train_nk (dict or None): 如果因子数据已加载，跳过因子数据加载。
            self.filtered_train_nk (dict or None): 如果已筛选股票池，跳过筛选步骤。

        处理步骤:
            1. 检查是否已经计算过收益率数据 (`ret`)。
               - 如果未计算过，调用 `return_calculation` 方法计算未来 h 天的收益率。
               - 如果已存在收益率数据，则跳过计算步骤。
            2. 检查是否已经加载过因子数据 (`train_nk`)。
               - 如果未加载，调用 `data_loading` 方法逐日加载因子数据。
               - 如果已存在因子数据，则跳过加载步骤。
            3. 检查是否已经筛选过股票池 (`filtered_train_nk`)。
               - 如果未筛选，调用 `select_target_pool` 方法筛选指定股票池中的股票并处理缺失值。
               - 如果已存在筛选后的数据，则跳过筛选步骤。
            4. 将最终的筛选结果保存为文件，以便后续使用。

        输出:
            已筛选和整理后的因子数据字典 (self.filtered_train_nk)，用于训练和预测。
        """

        # 检查是否已经计算过ret
        if not hasattr(self, 'ret') or self.ret is None:
            self.return_calculation()
        else:
            logging.info("收益率数据已加载，跳过return_calculation!")

        # 检查是否已经加载过 factor_nk
        if not hasattr(self, 'train_nk') or self.train_nk is None:
            self.data_loading()
        else:
            logging.info("训练数据已加载，跳过data_loading!")

        # 检查是否已经筛选过股票池
        if not hasattr(self, 'filtered_train_nk') or self.filtered_train_nk is None:
            self.select_target_pool()
        else:
            logging.info("股票池筛选已完成，跳过select_target_pool!")
        logging.info("数据加载和筛选已全部完成，返回filtered_train_nk: dict!")
        with open(f"{self.output_folder}h{self.h}_filtered_train_nk_dict.pkl", "wb") as f:
            pickle.dump(self.filtered_train_nk, f)
            
    def clear_data_loading(self):
        del self.filtered_train_nk
        gc.collect()
    
    def main(self):
        print("WARNING: This is a dummy main! Please write yours.")
        if os.path.exists(f"{output_folder}filtered_train_nk_dict.pkl"):
            with open("filtered_train_nk_dict.pkl", "rb") as f:
                self.filtered_train_nk = pickle.load(f) 
        else: 
            self.general_main_init()
        # xgboost = xgb.XGBRegressor()

        # 遍历每一天，训练和预测
        for i, t in enumerate(tqdm(self.dates, desc="逐日滑动训练和预测")):
            if i < self.train_size:  # 如果历史数据不足，跳过！
                continue
            elif t in self.train_dates:  # 如果是模型更新日，则更新模型
                tqdm.write(f"训练日{t.date()}：更新模型")  # 显示进度
                
                self.train_data_preparation(i)
                
                if len(self.train_data) == 0:
                    tqdm.write(f"训练日{t.date()}: 无有效X输入，特征不足，跳过训练")
                    continue
                    
                self.feature_label_split()

                self.grid_search_training(xgboost,t)
                
                logging.info(f"{self.model_params[t]}")

            else:  # 如果不是模型更新日
                if len(self.model_params.keys()) == 0:
                    continue  # 如果尚无训练好的模型，则跳过！
                    
            self.model_predict(t)
                
        # 保存模型参数与预测TxN矩阵
        with open(self.output_folder + 'model_params.pkl', 'wb') as file:
            pickle.dump(self.model_params, file)
        self.fcst.reset_index().to_feather(self.output_folder + 'score.feather')