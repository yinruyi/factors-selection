# 多因子策略
from CAL.PyCAL import *
import numpy as np
import pandas as pd

start = '2015-01-01'                       # 回测起始时间
end = '2016-08-31'     # 回测结束时间
universe = set_universe('HS300')    # 股票池
benchmark = 'HS300'                 # 策略参考标准
capital_base = 10000000                     # 起始资金
freq = 'd'                              # 策略类型，'d'表示日间策略使用日线回测，'m'表示日内策略使用分钟线回测
refresh_rate = 1                          # 调仓频率

# 构建日期列表，取每周最后一个交易日
data=DataAPI.TradeCalGet(exchangeCD=u"XSHG",beginDate=u"20140101",endDate='20160901',field=['calendarDate','isWeekEnd','isMonthEnd'],pandas="1")
data = data[data['isWeekEnd'] == 1]
date_list = map(lambda x: x[0:4]+x[5:7]+x[8:10], data['calendarDate'].values.tolist())

# 日期处理相关
cal = Calendar('China.SSE')
period = Period('-1B')

# 因子相关
factor_names = ['PE','ROE','RSI',"CCI5"]
factor_weight = [-1,1,-1,-1]
factor_api_field = ['secID','ticker'] + factor_names

commission = Commission(buycost=0.0003, sellcost=0.0013, unit='perValue')

def Schmidt(data):
    output = pd.DataFrame()
    mat = np.mat(data)
    output[0] = np.array(mat[:,0].reshape(len(data),))[0]
    for i in range(1,data.shape[1]):
        tmp = np.zeros(len(data))
        for j in range(i):
            up = np.array((mat[:,i].reshape(1,len(data)))*(np.mat(output[j]).reshape(len(data),1)))[0][0]
            down = np.array((np.mat(output[j]).reshape(1,len(data)))*(np.mat(output[j]).reshape(len(data),1)))[0][0]
            tmp = tmp+up*1.0/down*(np.array(output[j]))
        output[i] = np.array(mat[:,i].reshape(len(data),))[0]-np.array(tmp)
    output.index = data.index
    output.columns = data.columns
    return output

def initialize(account):                   # 初始化虚拟账户状态
    pass

def handle_data(account):                  # 每个交易日的买入卖出指令
    today = account.current_date
    today = Date.fromDateTime(account.current_date)  # 向前移动一个工作日
    yesterday = cal.advanceDate(today, period)
    yesterday = yesterday.toDateTime().strftime('%Y%m%d')
    if yesterday in date_list:  
        factordata = DataAPI.MktStockFactorsOneDayGet(tradeDate=yesterday,secID=set_universe(benchmark,yesterday),field=factor_api_field).set_index('ticker')
        for i in range(len(factor_names)):
            signal = standardize(neutralize(winsorize(factordata[factor_names[i]].dropna().to_dict()),yesterday)) #去极值，标准化，中性化
            factordata[factor_names[i]][signal.keys()] = signal.values()
        factordata = factordata.dropna()
        factordata = factordata.set_index('secID')
        factordata = Schmidt(factordata)
        # factordata['total_score'] = np.dot(factordata, np.array([1.0/len(factor_names) for i in range(len(factor_names))]))    #因子值等权求和
        factordata['total_score'] = np.dot(factordata, np.array(factor_weight))
        factordata.sort(['total_score'],ascending=False,inplace=True)                  #排序
        factordata = factordata[:20]
        
        # 先卖出
        sell_list = account.valid_secpos
        for stk in sell_list:
            order_to(stk, 0)

        # 再买入
        buy_list = list(set(factordata.index).intersection(set(account.universe)))
        total_money = account.referencePortfolioValue
        prices = account.referencePrice 
        for stk in buy_list:
            if np.isnan(prices[stk]) or prices[stk] == 0:  # 停牌或是还没有上市等原因不能交易
                continue
            order(stk, int(total_money / len(buy_list) / prices[stk] /100)*100)
    else:
        return
