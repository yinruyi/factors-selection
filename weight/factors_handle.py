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

IDX = ['000016', '000300', '000905']

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

    def _factors_one_day_get(self, factor_name, tradeDate, ticker):
        '''
        多只股票某天单因子数据
        '''
        factor_api_field = "secID," + factor_name
        # 获取当日成分股
        cons_id_df = DataAPI.IdxConsGet(secID=u"",ticker=ticker,intoDate=tradeDate,isNew=u"",field=u"consID",pandas="1")
        cons_id_ls = cons_id_df["consID"].tolist()
        cons_id_str = ",".join(cons_id_ls)
        if factor_name in ["ADX","RSI","VOL20","MTM","ROA","ROE","PB","PE","NetAssetGrowRate","NetProfitGrowRate","HBETA","InventoryTRate","OperatingRevenueGrowRate"]:
            return DataAPI.MktStockFactorsOneDayGet(tradeDate=tradeDate,secID=cons_id_str,ticker=u"",field=factor_api_field,pandas="1")
        elif factor_name in ["marketValue"]:
            return DataAPI.MktEqudGet(tradeDate=tradeDate,secID=cons_id_str,ticker=u"",field=factor_api_field,pandas="1")
        else:
            return DataFrame()

    def rank_ic(self, factor_name, ticker="000300"):
        '''
        计算rank-ic/ic并且画图
        @factor_name为因子名,如"RSI"
        @ticker为指数代码
        '''
        trading_days = self.day_list
        rank_ic_ls = []
        ic_ls = []

        for i in range(len(trading_days)-1):
            # 获取当日成分股
            cons_id_df = DataAPI.IdxConsGet(secID=u"",ticker=ticker,intoDate=trading_days[i],isNew=u"",field=u"consID",pandas="1")
            cons_id_ls = cons_id_df["consID"].tolist()
            cons_id_str = ",".join(cons_id_ls)
            #获取每周最后一个交易日的因子值
            factor_df = self._factors_one_day_get(factor_name, trading_days[i], ticker)
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

    def factors_weight_cal(self, factor_name_list, today, ticker="000300", mode="IC"):
        '''
        计算factors/因子加权权重
        根据Quantitative Equity Portfolio Management: Modern Techniques and Applications
        这种方法能较好解决因子间的相关性问题，
        他先证明股票组合收益取决于加总因子的IC，
        要获得稳定收益就需要加总因子的IC足够稳定，
        因此他采取最大化复合因子IC_IR的方式来获得各个alpha因子的权重。
        @factor_name_list 所需计算权重因子list
        @today 计算到某天为止的weight
        @ticker 计算weight股票基准
        @mode 默认通过IC计算,IC/rankIC
        '''
        day_list = self.day_list
        trading_days = self._back_day_list(today, day_list, 10)
        def _factor_ic(factor_name, trading_days=trading_days ,mode=mode, ticker=ticker):
            if mode == "IC":
                ic_ls = []
                for i in range(len(trading_days)-1):
                    cons_id_df = DataAPI.IdxConsGet(secID=u"",ticker=ticker,intoDate=trading_days[i],isNew=u"",field=u"consID",pandas="1")
                    cons_id_ls = cons_id_df["consID"].tolist()
                    cons_id_str = ",".join(cons_id_ls)
                    #获取每周最后一个交易日的因子值
                    factor_df = self._factors_one_day_get(factor_name, trading_days[i], ticker)
                    # 获取相应股票未来一周的收益
                    weekly_return = DataAPI.MktEquwAdjGet(secID=cons_id_str,beginDate=trading_days[i+1],endDate=trading_days[i+1],field=u"secID,return",pandas="1")
                    factor_return_df = factor_df.merge(weekly_return,on='secID', how="inner")
                    factor_return_df[factor_return_df["return"]==0] = None
                    factor_return_df.dropna(inplace=True)
                    ic, ic_p_value = st.pearsonr(factor_return_df[factor_name],factor_return_df["return"])
                    # ic_temp_dict = {}
                    # ic_temp_dict[factor_name] = ic
                    ic_ls.append(ic)
                return ic_ls
            elif mode == "rankIC":
                rank_ic_ls = []
                for i in range(len(trading_days)-1):
                    cons_id_df = DataAPI.IdxConsGet(secID=u"",ticker=ticker,intoDate=trading_days[i],isNew=u"",field=u"consID",pandas="1")
                    cons_id_ls = cons_id_df["consID"].tolist()
                    cons_id_str = ",".join(cons_id_ls)
                    #获取每周最后一个交易日的因子值
                    factor_df = self._factors_one_day_get(factor_name, trading_days[i], ticker)
                    # 获取相应股票未来一周的收益
                    weekly_return = DataAPI.MktEquwAdjGet(secID=cons_id_str,beginDate=trading_days[i+1],endDate=trading_days[i+1],field=u"secID,return",pandas="1")
                    factor_return_df = factor_df.merge(weekly_return,on='secID', how="inner")
                    factor_return_df[factor_return_df["return"]==0] = None
                    factor_return_df.dropna(inplace=True)
                    rank_ic, rank_ic_p_value = st.pearsonr(factor_return_df[factor_name].rank(),factor_return_df["return"].rank())
                    rank_ic_ls.append(ic)
                return rank_ic_ls
        if mode == "IC":
            IC = np.array(map(_factor_ic, factor_name_list)).T
            print IC,type(IC)



    def _back_day_list(self, today, date_list, back_day_num=""):
        index = date_list.index(today)
        if back_day_num == "":return date_list[0:index]
        if index < back_day_num:
            return date_list[0:index]
        else:
            return date_list[index-10:index]


if __name__ == '__main__':
    print 'test'
    factor_name_list = ["RSI","MTM","ROA","PE","OperatingRevenueGrowRate"]
    MutiFactorsSelect().factors_weight_cal(factor_name_list,"20150821")