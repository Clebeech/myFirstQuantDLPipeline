import itertools
from scipy.stats import norm
from ml_lib import *

class FcstEvaluator:
    def __init__(self, output_folder, pred_horizons, n_jobs=20):
        """初始化，设置并行计算的线程数量
        参数
        ----------
            output_folder: str
                结果储存路径，最终输出路径为output_folder下的results文件夹
            pred_horizons: list
                预测时的h_label列表
            n_jobs: int
                并行设置的线程数量，默认值为20
        """
        self.n_jobs = n_jobs
        self.pred_horizons = pred_horizons
        self.all_preds = {}
        self.results_path = os.path.join(output_folder, 'results/')
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)
        print(f"FcstEvaluator实例创建成功！")

    def set_up(self, h, preds, stk_pools, ic_modes, ret_modes, ind=None, factors={}, horizons=[1, 2, 5, 10, 20, 40, 60], trade_modes=['open'], sides=['both'], pcts=[]):
        """设置IC计算模式
        参数
        ----------
            h: int
                预测值tn矩阵的h_label
            preds: dataframe
                需要进行ic分析的预测值tn矩阵
            stk_pools: list
                股票池范围，可选值为['all', 'hs300', 'csi500', 'csi800', 'csi1000']
            ind: str | optional
                行业中性化基于的行业分类，默认为None，即不进行行业中性化
            factors: dict | optional
                选择进行的其他中性化因子，默认为{}，即不进行其他中性化
            ic_mode: str | optional
                ic类型，可选值为['linear', 'rank']，默认为'rank'
            ret_mode: str | optional
                收益率计算方式，可选值为['stage', 'cum']，默认为'stage'
            horizons: list | optional
                收益率计算的h_label，默认值为[1, 2, 5, 10, 20, 40, 60]
            trade_mode: str | optional
                收益率计算所使用的price种类，可选值为 ['close', 'open', 'vwap', 'vwap_1000'], 默认值为 'open'
            side: list
                ic的多空（选择百分位点前/后），可选值为['up', 'down', 'both']，默认值为'both'
            pct: float | optional
                ic的多空百分位点，可选范围为(0,1]，默认值为1
        """
        self.pred_h = h
        self.preds = preds
        # 保存不同预测周期的预测值
        self.all_preds[h] = preds
        self.stk_pools = ensure_list(stk_pools)
        self.ind = ind
        self.factors = ensure_dict(factors)
        self.ic_modes = ensure_list(ic_modes)
        self.ret_modes = ensure_list(ret_modes)
        self.trade_modes = ensure_list(trade_modes)
        self.sides = ensure_list(sides)
        if self.sides != ['both']:
            self.pcts = ensure_list(pcts)
            if len(self.pcts)==0:
                raise ValueError("请设置百分位点！")
        self.horizons = ensure_tuple(horizons)
        # 获取所需要的股票池tn矩阵
        self.stk_pool_dict = {}
        for stk_pool in self.stk_pools:
            self.stk_pool_dict[stk_pool] = get_stk_pool(stk_pool)
        # 获取fm中的中性化后的残差收益率
        self.res_rets_cum = {}
        self.res_rets_stage = {}
        if ind=='ind1' and list(factors.keys())==['size']:
            for mode in self.trade_modes:
                for h in self.horizons:
                    for ret_mode in self.ret_modes:
                        if ret_mode == 'stage':
                            target = fm.get(f'res_ret_{mode}_{h}')
                            self.res_rets_stage[f"{mode}_{str(h)}"] = target
                        elif ret_mode == 'cum':
                            target = fm.get(f'res_ret_{mode}_{h}_cum')
                            self.res_rets_cum[f"{mode}_{str(h)}"] = target
        # 生成所有参数的组合
        self.param_combinations = list(itertools.product(stk_pools, ic_modes, ret_modes, trade_modes, sides, pcts))
        print(f"h_label为{self.pred_h}的FcstEvaluator设置成功！")
    
    @staticmethod
    def calc_fcst_targets(start=0, end=1, trade_mode='open', ind=None, factors={}):
        """计算未来h天的收益率
        参数
        ----------
            start: int
                标记评估周期开始的位置
            end: int
                标记评估周期结束的位置，在calc_ic中被设置为h_label
            trade_mode: str | optional
                收益率计算所使用的price种类，可选值为 ['close', 'open', 'vwap', 'vwap_1000'], 默认值为 'open'
                close模式下：
                    未来 h 天收益率 = (t + h)收盘价 / t 收盘价
                其他模式下：
                    未来 h 天收益率 = (t + h)收盘价 / (t + 1) 收盘价
            ind: str | optional
                是否对收益率进行行业中性化，默认值为 None， 即不进行行业中性化
            factors: dict | optional
                是否对收益率进行其他中性化，默认值为 {}， 即不进行其他中性化
            ic_mode: str | optional
                ic类型，可选值为['linear', 'rank']，默认为'rank'
        
        """
        assert trade_mode in ['close', 'open', 'vwap', 'vwap_1000']
        
        cumadj = fm.get('cumadj')
        price_ = fm.get(trade_mode)
        adj_price = price_ * cumadj

        if trade_mode == 'close': 
            # 收益率 = 未来第h天的 adj_close / 当天的 adj_close
            target = adj_price.shift(-end) / adj_price.shift(-start) - 1
        else: 
            # 收益率 = 未来第h天的 adj_price / 明天的 adj_price
            target = adj_price.shift(-end-1) / adj_price.shift(-start-1) - 1

        # 如果ind和factors均为空，不做任何处理
        if (ind is None) and (len(factors)==0):
            return target
        else:
            ind_df = fm.get(ind)
            target = neutralize(target, ind=ind_df, factors=factors)
            return target
          
    def calc_ic(self, preds, horizons, ind, factors, stk_pool, ic_mode, ret_mode, trade_mode, side, pct):
        """计算单一模式下的IC值"""
        
        # 判断多头 / 空头及多空百分比
        if side == 'up':
            preds_normalized = normalize_by_row(preds)
            selected_preds = preds[preds_normalized>norm.ppf(pct)]
        elif side == 'down':
            preds_normalized = normalize_by_row(preds)
            selected_preds = preds[preds_normalized<norm.ppf(pct)]
        else:
            selected_preds = preds.copy()
        
        # 进行ic_mode判断
        selected_preds = selected_preds.rank(axis=1) if ic_mode=='rank' else selected_preds
        
        # 获取股票池tn矩阵
        stk_pool_tn = self.stk_pool_dict[stk_pool]
        
        # 获取未来不同周期的收益率，并计算对应周期的IC
        ic_table = pd.DataFrame(index=horizons, columns=selected_preds.index)
        for i, horizon in enumerate(horizons):   
            # 获得未来收益率的起始位置和终止位置，进行ret_mode判断
            start = 0 if i==0 else (horizons[i-1] if ret_mode=='stage' else 0)
            end = horizon
            
            # 如果进行行业、市值中性化则直接从factorbase中提取数据，否则需要现场计算
            if ind=='ind1' and list(factors.keys())==['size']:
                if ret_mode == 'stage':
                    target = self.res_rets_stage[f"{self.trade_mode}_{str(horizon)}"][stk_pool_tn]
                elif ret_mode == 'cum':
                    target = self.res_rets_cum[f"{self.trade_mode}_{str(horizon)}"][stk_pool_tn]
            else:
                target = self.calc_fcst_targets(start, end, trade_mode, ind, factors)[stk_pool_tn]
            
            # 如果计算 RankIC, 则需要将target转换为截面排名
            target = target.rank(axis=1) if ic_mode=='rank' else target
            
            # 计算 IC
            ic_table.loc[horizon] = selected_preds.corrwith(target, axis=1, method='pearson')
        
        # 创建 MultiIndex
        index = pd.MultiIndex.from_product([[ic_mode], [ret_mode], [trade_mode], [stk_pool], [side], [pct], horizons],
                                           names=['ic_mode', 'ret_mode', 'trade_mode', 'stk_pool', 'side', 'pct', 'horizon']
                                          )
        # 生成multiindex_dataframe
        ic_df = pd.DataFrame(ic_table.values, index=index, columns=selected_preds.index)
        
        return ic_df
    
    def ic_analysis_(self):
        """使用joblib调用calc_ic函数并行运算
        输出
        ----------
            在results文件下设置detailed_ic文件夹，用于存储每一天的IC值，文件名为h{h_label}_ic.csv
        """
        # 使用 joblib 并行运行
        results = Parallel(n_jobs=self.n_jobs, backend='multiprocessing')(delayed(self.calc_ic)(self.preds, self.horizons, self.ind, self.factors, p1, p2, p3, p4, p5, p6) for p1, p2, p3, p4, p5, p6 in tqdm(self.param_combinations, desc="计算ic"))
    
        res = pd.concat(results)
        res.columns = self.preds.index
        res = res.sort_index()
        res.columns = res.columns.astype(str)

        # 存储详细的ic结果到results/detailed_ic/
        self.detailed_results_path = os.path.join(self.results_path, 'detailed_ic/')
        if not os.path.exists(self.detailed_results_path):
            os.makedirs(self.detailed_results_path)
        res.to_csv(os.path.join(self.detailed_results_path, f"h{self.pred_h}_ic.csv"))
        
        return res

    def calc_turnover_das(self, preds, stk_pool, side, pct):
        """计算因子矩阵的das换手率，das指diff_abs_sum"""
        if side == 'up':
            preds_normalized = normalize_by_row(preds)
            preds1 = preds[preds_normalized>norm.ppf(pct)]
        elif side == 'down':
            preds_normalized = normalize_by_row(preds)
            preds1 = preds[preds_normalized<norm.ppf(pct)]
        else:
            preds1 = preds.copy()
        
        # 计算换手率
        stk_pool_tn = self.stk_pool_dict[stk_pool]
        preds1 = preds1[stk_pool_tn] # 过滤股票池
        preds1 = normalize_by_row(preds1) # 截面标准化
        preds1 = preds1.divide(preds1.abs().sum(axis=1), axis=0) # 截面归一化
        res = pd.DataFrame(index=preds1.index, columns=self.horizons, dtype=np.float64)
        for h in tqdm(self.horizons, desc="计算各horizon上的res"):
            res[h] = preds1.diff(periods=h).abs().dropna(how='all',axis=0).sum(axis=1)/2
        return res
        
    def gen_ic_stats(self, start_date='2010-01-04', end_date='2022-12-30'):
        self.start_date = start_date
        self.end_date = end_date
        # 从 output_folder/results/detailed_ic/ 文件夹下提取 ic dataframe
        ic_df = pd.read_csv(f"{self.results_path}detailed_ic/h{self.pred_h}_ic.csv",index_col=['ic_mode', 'ret_mode', 'trade_mode', 'stk_pool', 'side', 'pct', 'horizon'])
        
        # 设置IC统计指标的存储路径: output_folder/results/ic_{start_date}_{end_date}_stats/
        self.stats_path = os.path.join(self.results_path, f"ic_{start_date}_{end_date}_stats/")
        if not os.path.exists(self.stats_path):
            os.makedirs(self.stats_path)
        
        # 确保输入的日期在合理区间
        first_date = ic_df.columns[0]
        last_date = ic_df.columns[-1]
        if end_date > last_date:
            end_date = last_date
        if start_date < first_date:
            start_date = first_date
        
        # 计算均值、标准差和ICIR
        ic_mean = ic_df.loc[:, start_date:end_date].mean(axis=1, numeric_only=True)
        ic_stdev = ic_df.loc[:, start_date:end_date].std(axis=1, numeric_only=True)
        mean_std = ic_mean / ic_stdev
        ic_ir = mean_std * np.sqrt(242/np.tile(np.array(self.horizons), int(len(ic_df)/len(self.horizons))))
        
        # 生成包含所有统计指标的dataframe
        ic_stats = pd.DataFrame({'mean': ic_mean,
                                 'stdev': ic_stdev,
                                 'mean/stdev': mean_std,
                                 'ICIR': ic_ir,
                                })
        ic_stats.index = ic_df.index

        # 存储ic_stats
        ic_stats.to_csv(os.path.join(self.stats_path, f"h{self.pred_h}_ic_stats.csv"))
        return ic_stats

    def process_all_combinations(self, start_date='2010-01-04', end_date='2022-12-30'):
        """合并results_folder下所有h_label的ic_stats
        参数：
        ----------
            start_date: str | optional
                开始日期，选择 results folder 下现有 ic_stats 的区间开始日期
            end_date: str | optional
                结束日期，选择 results folder 下现有 ic_stats 的区间结束日期
        输出：
        ----------
            combined_stats: DataFrame
                不同预测周期IC统计值的总表   
        """
        if not os.path.exists(f"{self.results_path}ic_{start_date}_{end_date}_stats/"):
            raise KeyError(f"start_date / end_date输入错误！请检查results文件夹下现有可选择的开始和结束日期！")
        
        # 简单合并不同预测周期的 IC 统计值
        all_stats = {}
        for h in self.pred_horizons:
            ic_stats_file_path = f"{self.results_path}ic_{start_date}_{end_date}_stats/"
            ic_stats = pd.read_csv(f"{ic_stats_file_path}h{h}_ic_stats.csv", 
                                   index_col=['ic_mode', 'ret_mode', 'trade_mode', 'stk_pool', 'side', 'pct', 'horizon'])
            all_stats[h] = ic_stats
        combined_stats = pd.concat(all_stats, names=['pred_h'])
        combined_stats.to_csv(os.path.join(self.results_path, f"combined_ic_stats.csv"))
        
        # 简化表格内容并调整格式
        summarized_stats = pd.DataFrame(combined_stats['mean'].apply(lambda x: f"{x:.6f}").astype(str) + '/' + combined_stats['ICIR'].apply(lambda x: f"{x:.6f}").astype(str))
        summarized_stats.rename(columns={0:"IC_mean/ICIR"}, inplace=True)
        cols = []
        score_cols = []
        tor_cols = []
        for h in np.unique(summarized_stats.index.get_level_values("horizon")).tolist():
            cols.append(f"IC({h})/ICIR")
            score_cols.append(f"SCORE({h})")
            tor_cols.append(f"TOR({h})")
        final_results = pd.DataFrame(index=summarized_stats.index.droplevel('horizon').drop_duplicates(keep='first'), columns=cols+tor_cols)
        
        # 加入换手率计算和最终结合换手率 & IC的打分
        for idx in final_results.index:
            final_results.loc[idx,cols] = summarized_stats.loc[idx,:].transpose().values
            final_results.loc[idx,tor_cols] = ((self.calc_turnover_das(self.all_preds[idx[0]], idx[4], idx[5], idx[6])).mean()).values
        final_results.loc[:, score_cols] = (4.8*final_results.loc[:,cols].applymap(lambda x: float(x.split('/')[0]))).values - (0.34*final_results.loc[:,tor_cols]).values
        
        # 储存最终结果
        final_results.to_csv(os.path.join(self.results_path, "eval_summary.csv"))
        return 0

    def plot_ic_sum(self, output_file, baseline_ic, detailed_ic, ic_stats, ret_mode='cum', ic_mode='linear', trade_mode='vwap', stk_pool='all', horizons=[1,2,5,10,20,40,60], side='up', pct=0.5):
        '''绘制累积IC图像
        Input:
        -ic_df: detailed_ic中的csv文件
        -ic_stats: ic_startdata_enddate中的csv文件
        '''

        ic_df = detailed_ic.copy()
        
        # 读取detailed_ic中指定horizons的结果(side=='both')
        ic_df = ic_df[(ic_df[['ic_mode', 'ret_mode', 'trade_mode', 'stk_pool','side']] == [ic_mode, ret_mode, trade_mode, stk_pool,'both']).all(axis = 1)]
        ic_df = ic_df[ic_df['horizon'].isin(horizons)]
        ic_df = ic_df.iloc[:, 6:].set_index('horizon').T
        ic_df.columns = [f"IC({int(col)})" for col in ic_df.columns]
        ic_df.index.name = 'date'
        ic_df.index = pd.to_datetime(ic_df.index)

        # 计算 IC 累积和
        cumulative_ic = ic_df.cumsum()

        # 读取多/空头端数据
        if side not in self.sides:
            raise KeyError(f'{side}不在现有范围中！请重新设置evaluator的sides设定!')
        if pct not in self.pcts:
            raise KeyError(f'{pct}不在现有范围中！请重新设置evaluator的pcts设定!')
        ic_df_side = detailed_ic.copy()
        ic_df_side = ic_df_side[(ic_df_side[['ic_mode', 'ret_mode', 'trade_mode', 'stk_pool', 'side', 'pct']] == [ic_mode, ret_mode, trade_mode, stk_pool, side, pct]).all(axis = 1)]
        ic_df_side = ic_df_side[ic_df_side['horizon'].isin(horizons)]
        ic_df_side = ic_df_side.iloc[:, 6:].set_index('horizon').T
        ic_df_side.columns = [f"IC({int(col)})" for col in ic_df_side.columns]
        ic_df_side.index.name = 'date'
        ic_df_side.index = pd.to_datetime(ic_df_side.index)

        # 计算多/空头端 IC 累积和
        cumulative_ic_side = ic_df_side.cumsum()

        baseline_ic_original = baseline_ic.copy()
        # 读取baseline_ic中指定horizons的结果(side=='both')
        baseline_ic = baseline_ic[(baseline_ic[['ic_mode', 'ret_mode', 'trade_mode', 'stk_pool','side']] == [ic_mode, ret_mode, trade_mode, stk_pool,'both']).all(axis = 1)]
        baseline_ic = baseline_ic[baseline_ic['horizon'].isin(horizons)]
        baseline_ic = baseline_ic.iloc[:, 6:].set_index('horizon').T
        baseline_ic.columns = [f"IC({int(col)})" for col in baseline_ic.columns]
        baseline_ic.index.name = 'date'
        baseline_ic.index = pd.to_datetime(baseline_ic.index)

        # 读取baseline_ic中指定多/空头端的结果
        baseline_ic_side = baseline_ic_original[(baseline_ic_original[['ic_mode', 'ret_mode', 'trade_mode', 'stk_pool','side','pct']] == [ic_mode, ret_mode, trade_mode, stk_pool, side, pct]).all(axis = 1)]
        baseline_ic_side = baseline_ic_side[baseline_ic_side['horizon'].isin(horizons)]
        baseline_ic_side = baseline_ic_side.iloc[:, 6:].set_index('horizon').T
        baseline_ic_side.columns = [f"IC({int(col)})" for col in baseline_ic_side.columns]
        baseline_ic_side.index.name = 'date'
        baseline_ic_side.index = pd.to_datetime(baseline_ic_side.index)
        
        # 只保留和detailed_ic相同的时间段
        baseline_ic = baseline_ic.loc[ic_df.dropna(how='all', axis=0).index[0]:ic_df.dropna(how='all', axis=0).index[-1], :]
        baseline_ic_side = baseline_ic_side.loc[ic_df.dropna(how='all', axis=0).index[0]:ic_df.dropna(how='all', axis=0).index[-1], :]
        
        # 计算 IC 累积和
        baseline_cumulative_ic = baseline_ic.cumsum()
        baseline_cumulative_ic_side = baseline_ic_side.cumsum()

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))
        
        for col in cumulative_ic.columns:
            # 测试的score.feather的cumulative IC
            ax.plot(cumulative_ic.index, cumulative_ic[col], label=f"CUM_IC_{col}")
            ax.plot(cumulative_ic_side.index, cumulative_ic_side[col], label=f"CUM_IC_SIDE_{col}")
            
            # baseline score.feather的cumulative IC
            ax.plot(baseline_cumulative_ic.index, baseline_cumulative_ic[col], label=f"BASELINE_IC_{col}")
            ax.plot(baseline_cumulative_ic_side.index, baseline_cumulative_ic_side[col], label=f"BASELINE_IC_SIDE_{col}")
            

        ic_type = 'IC' if ic_mode == 'linear' else 'RankIC'
        ax.set_title(f'{ic_type} Analysis on {stk_pool}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative IC')
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(output_file)


    def ic_result_plot(self, h, baseline_output_folder):
        '''
        绘制不同h_label的累积IC图像
        '''
        baseline_ic = pd.read_csv(os.path.join(baseline_output_folder, f'h{h}_ic.csv'))
        detailed_ic = pd.read_csv(os.path.join(self.detailed_results_path, f'h{h}_ic.csv'))
        ic_stats = pd.read_csv(os.path.join(self.stats_path, f'h{h}_ic_stats.csv'))
        output_path = self.results_path
        output_cominations = list(itertools.product(self.stk_pools, self.ic_modes))
        for combine in output_cominations:
            ic_type = 'IC' if combine[1] == 'linear' else 'RankIC'
            output_path = os.path.join(self.results_path, f'h{h}_{ic_type}_sum_{combine[0]}')
            self.plot_ic_sum(output_file=output_path, baseline_ic=baseline_ic, detailed_ic=detailed_ic, ic_stats=ic_stats, ic_mode=combine[1], stk_pool=combine[0], horizons=[1])