# coding:-*- utf-8 -*-
import scipy.stats as st
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import datetime
import numpy as np
from pandas import Series, DataFrame
from quartz.api import *

DataAPI.settings.cache_enabled = False

IDXMAP = {
    'SH50': '000016',
    'HS300': '000300',
    'ZZ500': '000905',
}

IDXMAP_REVERSE = {
	'000016': 'SH50',
	'000300': 'HS300',
	'000905': 'ZZ500',
}

FACTORS_NAME = [
    "ADX",# 平均动向指数,趋势型因子
    "RSI",# 相对强弱指标,超买超卖型因子
    "VOL20",# 20日平均换手率,成交量型因子
    "MTM",# 动量指标,趋势型因子
    "ROA",# 资产回报率,盈利能力和收益质量类因子
    "ROE",# 权益回报率,盈利能力和收益质量类因子
    "PB",# 市净率,估值与市值类因子
    "PE",# 市盈率,估值与市值类因子
    "NetAssetGrowRate",# 净资产增长率,成长能力类因子
    "NetProfitGrowRate",# 净利润增长率,属于成长能力类因子
    "HBETA",# 历史贝塔,超买超卖型因子
    "marketValue",# 总市值
    "EBIT",# 息税前利润
    "InventoryTRate", #存货周转率
    "OperatingRevenueGrowRate",# 营业收入增长率,属于成长能力类因子
]

class MutiFactorsSelect(object):
    def __init__(self, begin_day="20140101", end_day="20160831"):
        self.begin_day = begin_day
        self.end_day = end_day

    @property    
    def day_list(self):
        '''
        获取当周最后交易日list
        '''
        begin_day = self.begin_day
        end_day = self.end_day
        cal_dates = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate=begin_day, endDate=end_day, field="calendarDate,isWeekEnd")
        trading_days = cal_dates[cal_dates["isWeekEnd"]==1]["calendarDate"].tolist()
        return [day.replace("-","") for day in trading_days]

    def rank_ic(self, factor_name, ticker="000300"):
        '''
        计算rank-ic并且画图
        @factor_name为因子名,如"RSI"
        @ticker为指数代码
        '''
        trading_days = self.day_list
        rank_ic_ls = []
        ic_ls = []
        factor_api_field = "secID," + factor_name

        for i in range(len(trading_days)-1):
            # 获取当日成分股
            cons_id_df = DataAPI.IdxConsGet(secID=u"",ticker=ticker,intoDate=trading_days[i],isNew=u"",field=u"consID",pandas="1")
            cons_id_ls = cons_id_df["consID"].tolist()
            cons_id_str = ",".join(cons_id_ls)
            #获取每周最后一个交易日的因子值
            factor_df = DataAPI.MktStockFactorsOneDayGet(tradeDate=trading_days[i],secID=cons_id_str,ticker=u"",field=factor_api_field,pandas="1")
            # 获取相应股票未来一周的收益
            weekly_return = DataAPI.MktEquwAdjGet(secID=cons_id_str,beginDate=trading_days[i+1],endDate=trading_days[i+1],field=u"secID,return",pandas="1")
            factor_return_df = factor_df.merge(weekly_return,on='secID', how="inner")
            factor_return_df[factor_return_df["return"]==0] = None
            factor_return_df.dropna(inplace=True)
            
            rank_ic, rank_ic_p_value = st.pearsonr(factor_return_df[factor_name].rank(),factor_return_df["return"].rank())
           
            rank_ic_temp_dict = {}
            rank_ic_temp_dict["date"] = trading_days[i]
            rank_ic_temp_dict["Rank-IC"] = rank_ic
            rank_ic_ls.append(rank_ic_temp_dict)

            ic, ic_p_value = st.pearsonr(factor_return_df[factor_name],factor_return_df["return"])
			
            ic_temp_dict = {}
            ic_temp_dict["date"] = trading_days[i]
            ic_temp_dict["IC"] = ic
            ic_ls.append(ic_temp_dict)

        rank_ic_result_df = pd.DataFrame(rank_ic_ls)
        rank_ic_result_df["MovingAverage"] = pd.rolling_mean(rank_ic_result_df["Rank-IC"], window=20,min_periods=10)

        ic_result_df = pd.DataFrame(ic_ls)
        ic_result_df["MovingAverage"] = pd.rolling_mean(ic_result_df["IC"], window=20,min_periods=10)
        # plot
        matplotlib.style.use('ggplot')
        rank_ic_result_df["date"] = rank_ic_result_df["date"].apply(str)
        rank_ic_result_df.index = pd.to_datetime(rank_ic_result_df["date"])
        rank_ic_result_df.plot(colors=['royalblue', 'crimson']).set_title(factor_name+" in "+IDXMAP_REVERSE[ticker])
        plt.axhline(0, color='Gray')
        
        ic_result_df["date"] = ic_result_df["date"].apply(str)
        ic_result_df.index = pd.to_datetime(ic_result_df["date"])
        ic_result_df.plot(colors=['royalblue', 'crimson']).set_title(factor_name+" in "+IDXMAP_REVERSE[ticker])
        plt.axhline(0, color='Gray')

MutiFactorsSelect().rank_ic("RSI")