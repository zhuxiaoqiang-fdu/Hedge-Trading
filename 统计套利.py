# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 13:11:02 2021

@author: Admin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from string import digits
import datetime
import math
import jqdatasdk as jq
from sklearn import svm
import sklearn
from sklearn.model_selection import train_test_split


#归一化函数
def normal(x):
    m = x.max()
    n = x.min()
    return ((x-n)/(m-n))*2 -1

def normal2(x):
    sigma = x.std()
    mu = x.mean()
    return (x-mu)/sigma
 
  
#通过rank滚动计算趋势性因子
def cal_trend_rank(data_all, nday):
    data_all.index = range(len(data_all['close']))
    data_all['fac'] = np.nan
    for i in range(nday, len(data_all['close'])):
        ts_close = data_all['close'].iloc[range(i-nday, i)]
        ts_close.index = range(nday)
        cor = pd.concat([pd.Series(range(nday)), ts_close], axis=1).corr()
        data_all['fac'][i] = cor[0]['close']
    return data_all

#计算MA
def cal_MA(data_all, nday):
#    data_all['rolling_std'] = data_all['close_delta'].rolling_std(nday)
#    data_all['rolling_mean'] = data_all['close_delta'].rolling_mean(nday)
#    data_all['rolling_std'] = pd.rolling_std(data_all['close_delta'], window = nday)
#    data_all['rolling_mean'] = pd.rolling_mean(data_all['close_delta'], window = nday)
    
    data_all['rolling_mean'] = data_all['close_delta'].rolling(nday).mean()
    data_all['rolling_std'] = data_all['close_delta'].rolling(nday).std()
    
    return data_all

#计算RSI指标    
def cal_RSI(data_all):
    data_all['RSI_5day'] = np.nan
    data_all['RSI_8day'] = np.nan
    data_all['RSI_15day'] = np.nan
    for i in range(10,len(data_all['RSI_5day'])):
        ts_data8 = data_all[['date','dif']].iloc[i-8:i]
        data_all['RSI_8day'][i] = sum(ts_data8.dif[ts_data8.dif>0])/(sum(ts_data8.dif[ts_data8.dif>0]) + abs(sum(ts_data8.dif[ts_data8.dif<0])))
    return data_all          
 
#计算绝对的累计收益金额变化
def cal_plot(data_all):
    data_all[['total_abs', 'total_abs_buy', 'total_abs_sell']] = 0.0
    data_all.index = range(len(data_all['abs_buy']))
    strftime = datetime.datetime.strptime("2016-01-01", "%Y-%m-%d")
    del_ind = [0]
    for i in range(1,len(data_all['total_abs'])):
        if data_all['time'][i] > strftime:
            data_all['total_abs'][i] = data_all['total_abs'][i-1] + data_all['abs_buy'][i] + data_all['abs_sell'][i]
            data_all['total_abs_buy'][i] = data_all['total_abs_buy'][i-1] + data_all['abs_buy'][i]
            data_all['total_abs_sell'][i] = data_all['total_abs_sell'][i-1] + data_all['abs_sell'][i]
        else:
            del_ind.append(i)
    print(del_ind)
    try:
        data_all = data_all.drop(index=del_ind, axis=0)
    except:
        pass
    data_all.index= range(len(data_all['time']))
    return data_all    
    
#计算每一年的胜率
def cal_win_rate(data_all, year_beg, year_end):
    data_temp = data_all.loc[(data_all['time'] > datetime.datetime.strptime(year_beg+"-01-01", "%Y-%m-%d")) & (data_all['time'] < datetime.datetime.strptime(year_end+"-01-01", "%Y-%m-%d"))]
    win_buy = (data_temp['abs_buy'] > 0).astype(int).sum(axis=0)
    win_sell = (data_temp['abs_sell'] > 0).astype(int).sum(axis=0)            
    lose_buy = (data_temp['abs_buy'] < 0).astype(int).sum(axis=0)
    lose_sell = (data_temp['abs_sell'] < 0).astype(int).sum(axis=0)
    return (win_buy + win_sell), (lose_buy + lose_sell)
    
#计算每一年的最大回撤
def cal_max_drawdown(data_all, year_beg, year_end):
    data_temp = data_all.copy().loc[(data_all['time'] > datetime.datetime.strptime(year_beg+"-01-01", "%Y-%m-%d")) & (data_all['time'] < datetime.datetime.strptime(year_end+"-01-01", "%Y-%m-%d"))]
    max_draw = 1
    for i in data_temp.index[1:]:
        max_draw = min(max_draw, (data_temp['total_abs'][i]+5000000)/(data_temp['total_abs'].loc[:i]+5000000).max()) #假设有100w的钱，交易保证金为10w
    max_draw = 1 - max_draw     
    return max_draw

#计算每一年的年收益率
def cal_ann_return(data_all, year_beg, year_end):
    data_temp = data_all.loc[(data_all['time'] > datetime.datetime.strptime(year_beg+"-01-01", "%Y-%m-%d")) & (data_all['time'] < datetime.datetime.strptime(year_end+"-01-01", "%Y-%m-%d"))]
    return (data_temp['total_abs'].iloc[-1]+5000000)/(data_temp['total_abs'].iloc[0]+5000000)
    
#做多开仓
def buy(res, data_all, beg, hold_unit, hold_price, hold_time, trans_tick):
    hold_unit[res].append(1)
    hold_price[res].append(data_all['close_'+res][beg] + 2 * trans_tick[res])
    hold_time[res].append(beg)
    return hold_unit , hold_price, hold_time

#做空开仓
def sell(res, data_all, beg, hold_unit, hold_price, hold_time, trans_tick):
    hold_unit[res].append(-1)
    hold_price[res].append(data_all['close_'+res][beg] - 2 * trans_tick[res])
    hold_time[res].append(beg)
    return hold_unit, hold_price, hold_time

#多头平仓    
def end_buy(res, data_all, beg, hold_unit, hold_price, hold_time, trans_tick, trans_unit):
    data_all['abs_buy'][beg] += (data_all['close_'+res][beg] - hold_price[res][0]) * trans_unit[res] * (100000//(data_all['close_'+res][hold_time[res][0]] * trans_unit[res] * 0.1))
    hold_time[res].pop(0)
    hold_unit[res].pop(0)
    hold_price[res].pop(0)
    return hold_unit, hold_price, hold_time, data_all

#空头平仓    
def end_sell(res, data_all, beg, hold_unit, hold_price, hold_time, trans_tick, trans_unit):
    data_all['abs_sell'][beg] += (hold_price[res][0] - data_all['close_'+res][beg]) * trans_unit[res] * (100000//(data_all['close_'+res][hold_time[res][0]] * trans_unit[res] * 0.1))
    hold_time[res].pop(0)
    hold_unit[res].pop(0)
    hold_price[res].pop(0)
    return hold_unit, hold_price, hold_time, data_all

#check是否到达了止盈止损，平多头        
def force_end_buy(res, data_all, beg, hold_unit, hold_price, hold_time, trans_tick, trans_unit, nday_end, stbk_buy, upbk_buy):
    delta = 0
    for k in range(len(hold_unit[res])):
        if ((data_all['close_'+res][beg] - hold_price[res][k-delta])/hold_price[res][k-delta] < stbk_buy) or ((data_all['close_'+res][beg] - hold_price[res][k-delta])/hold_price[res][k-delta] > upbk_buy) or ((beg - hold_time[res][k-delta]) > nday_end):
            hold_unit, hold_price, hold_time, data_all = end_buy(res, data_all, beg, hold_unit, hold_price, hold_time, trans_tick, trans_unit)
            delta += 1
    return hold_unit, hold_price, hold_time, data_all

#check是否到达了止盈止损，平空头
def force_end_sell(res, data_all, beg, hold_unit, hold_price, hold_time, trans_tick, trans_unit, nday_end, stbk_sell, upbk_sell):
    delta = 0
    for k in range(len(hold_unit[res])):
        if ((hold_price[res][k-delta] - data_all['close_'+res][beg])/data_all['close_'+res][beg] < stbk_sell) or ((hold_price[res][k-delta] - data_all['close_'+res][beg])/data_all['close_'+res][beg] > upbk_sell) or ((beg - hold_time[res][k-delta]) > nday_end):
            hold_unit, hold_price, hold_time, data_all = end_sell(res, data_all, beg, hold_unit, hold_price, hold_time, trans_tick, trans_unit)
            delta += 1
    return hold_unit, hold_price, hold_time, data_all

#进行回测
def backtest(data_all):
    data_all['delta_beg'] = np.nan
    data_all[['abs_buy', 'abs_sell']] = 0.0
#    stbk_sell = -0.02 #止损点
#    stbk_buy = -0.02
#    upbk_buy = 0.03
#    upbk_sell = 0.03 #止盈点
    nday_end = 60
    nday_tran = 1440
    nday = 1440
    #最小变动价格
    trans_tick = {'l':5, 'v':5 , 'pp':1 , 'TA': 2, 'MA': 1, 'bu':2 , 'ru':5 , 'au':0.02 , 'ag':1, 'jm': 0.5, 'j': 0.5, 'zc': 0.2, 'rb': 1, 'hc': 1, 'i': 0.5, 'cu': 10, 'al': 5, 'zn': 5, 'pb': 5, 'ni': 10, 'sn': 10, 'a': 1, 'c': 1, 'm': 1, 'rm': 1, 'y': 2, 'p': 2, 'oi': 2, 'cf': 5, 'sr': 1, 'cj': 5, 'jd': 1, 'cs': 1, 'sp': 2}   #元/单位
    #每手乘数
    trans_unit = {'l':5, 'v':5 , 'pp':5, 'TA': 5, 'MA': 10, 'bu': 10, 'ru': 10, 'au': 100, 'ag': 15, 'jm': 60, 'j': 100, 'zc': 100, 'rb': 10, 'hc': 10, 'i': 100, 'cu': 5, 'al': 5, 'zn': 5, 'pb': 5, 'ni': 1, 'sn': 1, 'a': 10, 'c': 10, 'm': 10, 'rm': 10, 'y': 10, 'p': 10, 'oi': 10, 'cf': 5, 'sr': 10, 'cj': 5, 'jd': 5, 'cs': 10, 'sp': 10}   #多少单位/一手
    #持仓手数
    hold_unit = {'l':[] ,'v':[] , 'pp':[], 'TA': [], 'MA': [], 'bu': [], 'ru': [], 'au': [], 'ag': [], 'jm': [], 'j': [], 'zc': [], 'rb': [], 'hc': [], 'i': [], 'cu': [], 'al': [], 'zn': [], 'pb': [], 'ni': [], 'sn': [], 'a': [], 'c': [], 'm': [], 'rm': [], 'y': [], 'p': [], 'oi': [], 'cf': [], 'sr': [], 'cj': [], 'jd': [], 'cs': [], 'sp': []}
    #入场价格
    hold_price = {'l':[] ,'v':[] , 'pp':[], 'TA': [], 'MA': [], 'bu': [], 'ru': [], 'au': [], 'ag': [], 'jm': [], 'j': [], 'zc': [], 'rb': [], 'hc': [], 'i': [], 'cu': [], 'al': [], 'zn': [], 'pb': [], 'ni': [], 'sn': [], 'a': [], 'c': [], 'm': [], 'rm': [], 'y': [], 'p': [], 'oi': [], 'cf': [], 'sr': [], 'cj': [], 'jd': [], 'cs': [], 'sp': []}
    #入场时间
    hold_time = {'l':[] ,'v':[] , 'pp':[], 'TA': [], 'MA': [], 'bu': [], 'ru': [], 'au': [], 'ag': [], 'jm': [], 'j': [], 'zc': [], 'rb': [], 'hc': [], 'i': [], 'cu': [], 'al': [], 'zn': [], 'pb': [], 'ni': [], 'sn': [], 'a': [], 'c': [], 'm': [], 'rm': [], 'y': [], 'p': [], 'oi': [], 'cf': [], 'sr': [], 'cj': [], 'jd': [], 'cs': [], 'sp': []}

    for i in range(nday, len(data_all['delta_beg']) - nday_end):
        sig = 0
        if i > nday_tran+nday_end+nday:                  
            sig = 0.0
            sig_temp_buy = 0
            sig_temp_sell = 0
            for x in range(nday_tran):  
                for res1 in ['jm', 'j']:
                    sig_temp_buy += math.atan(x+1)*(data_all['abs_buy'][i-nday_tran+x] + 2*trans_tick[res1]*trans_unit[res1] * (100000//(data_all['close_'+res1][i] * trans_unit[res1] * 0.1)))
                    sig_temp_sell += math.atan(x+1)*(data_all['abs_sell'][i-nday_tran+x] + 2*trans_tick[res1]*trans_unit[res1] * (100000//(data_all['close_'+res1][i] * trans_unit[res1] * 0.1)))
                if (data_all['abs_buy'][i-nday_tran+x] != 0): #and (data_all['position'][i-nday_tran+x] == 'up' or 'down'):
                    if sig_temp_buy > 0:
                        sig = sig + sig_temp_buy*(1)
                    else:
                        sig = sig + sig_temp_buy
                        
                elif (data_all['abs_sell'][i-nday_tran+x] != 0):
                    if sig_temp_sell < 0:
                        sig = sig - sig_temp_sell*(1)
                    else:
                        sig = sig - sig_temp_sell        
        if sig >= 0:
            if data_all['close_delta'][i] > data_all['rolling_mean'][i] + 3 * data_all['rolling_std'][i]:
                if sum(hold_unit['jm']) >= 0:
                    hold_unit, hold_price, hold_time = buy('jm', data_all, i, hold_unit, hold_price, hold_time, trans_tick)
                else:
                    hold_unit, hold_price, hold_time, data_all = end_sell('jm', data_all, i, hold_unit, hold_price, hold_time, trans_tick, trans_unit)
                
                if sum(hold_unit['j']) <= 0:
                    hold_unit, hold_price, hold_time = sell('j', data_all, i, hold_unit, hold_price, hold_time, trans_tick)
                else:
                    hold_unit, hold_price, hold_time, data_all = end_buy('j', data_all, i, hold_unit, hold_price, hold_time, trans_tick, trans_unit)
        
            if data_all['close_delta'][i] < data_all['rolling_mean'][i] - 3 * data_all['rolling_std'][i]:
                if sum(hold_unit['jm']) <= 0:
                    hold_unit, hold_price, hold_time = sell('jm', data_all, i, hold_unit, hold_price, hold_time, trans_tick)
                else:
                    hold_unit, hold_price, hold_time, data_all = end_buy('jm', data_all, i, hold_unit, hold_price, hold_time, trans_tick, trans_unit)
                
                if sum(hold_unit['j']) >= 0:
                    hold_unit, hold_price, hold_time = buy('j', data_all, i, hold_unit, hold_price, hold_time, trans_tick)
                else:
                    hold_unit, hold_price, hold_time, data_all = end_sell('j', data_all, i, hold_unit, hold_price, hold_time, trans_tick, trans_unit)
        
        if sig < 0:
            if data_all['close_delta'][i] > data_all['rolling_mean'][i] + 3 * data_all['rolling_std'][i]:
                if sum(hold_unit['j']) >= 0:
                    hold_unit, hold_price, hold_time = buy('j', data_all, i, hold_unit, hold_price, hold_time, trans_tick)
                else:
                    hold_unit, hold_price, hold_time, data_all = end_sell('j', data_all, i, hold_unit, hold_price, hold_time, trans_tick, trans_unit)
                
                if sum(hold_unit['jm']) <= 0:
                    hold_unit, hold_price, hold_time = sell('jm', data_all, i, hold_unit, hold_price, hold_time, trans_tick)
                else:
                    hold_unit, hold_price, hold_time, data_all = end_buy('jm', data_all, i, hold_unit, hold_price, hold_time, trans_tick, trans_unit)
        
            if data_all['close_delta'][i] < data_all['rolling_mean'][i] - 3 * data_all['rolling_std'][i]:
                if sum(hold_unit['j']) <= 0:
                    hold_unit, hold_price, hold_time = sell('j', data_all, i, hold_unit, hold_price, hold_time, trans_tick)
                else:
                    hold_unit, hold_price, hold_time, data_all = end_buy('j', data_all, i, hold_unit, hold_price, hold_time, trans_tick, trans_unit)
                
                if sum(hold_unit['jm']) >= 0:
                    hold_unit, hold_price, hold_time = buy('jm', data_all, i, hold_unit, hold_price, hold_time, trans_tick)
                else:
                    hold_unit, hold_price, hold_time, data_all = end_sell('jm', data_all, i, hold_unit, hold_price, hold_time, trans_tick, trans_unit)
         
        
        
        stbk_sell = - 0.012
        stbk_buy = - 0.012
        upbk_buy = 0.015
        upbk_sell = 0.015
        
        for res1 in ['jm', 'j']:
            if sum(hold_unit[res1]) > 0:
                hold_unit, hold_price, hold_time, data_all = force_end_buy(res1, data_all, i, hold_unit, hold_price, hold_time, trans_tick, trans_unit, nday_end, stbk_buy, upbk_buy)
            if sum(hold_unit[res1]) < 0:
                hold_unit, hold_price, hold_time, data_all = force_end_sell(res1, data_all, i, hold_unit, hold_price, hold_time, trans_tick, trans_unit, nday_end, stbk_sell, upbk_sell)
    return data_all      

 
              



#jq.auth('15216691628', 'Zsq968813')
jq.auth('18951734861', '734861')
j_all = jq.get_price(['J9999.XDCE'], start_date='2021-01-01', end_date='2021-05-31', frequency='minute', fields=None, skip_paused=False, fq='pre')
jm_all = jq.get_price(['JM9999.XDCE'], start_date='2021-01-01', end_date='2021-05-31', frequency='minute', fields=None, skip_paused=False, fq='pre')
jm_j_all = pd.concat([j_all['time'], j_all['close'], jm_all['close'], j_all['close'] - jm_all['close']], axis=1)
jm_j_all.columns = ['time', 'close_j', 'close_jm', 'close_delta']

nday = 1440  #往前滚动k线数量

#计算mean和std
jm_j_all = cal_MA(jm_j_all, nday)


#进行回测
jm_j_all = backtest(jm_j_all)


#计算收益率走势
jm_j_all = cal_plot(jm_j_all)

#收益率曲线画图
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
#plt.plot(jm_all['date'], jm_all['total_abs'],label="jm")
#plt.plot(j_all['date'], j_all['total_abs'],label="j")
#plt.plot(rb_all['date'], rb_all['total_abs'],label="rb")
#plt.plot(hc_all['date'], hc_all['total_abs'],label="hc")
#plt.plot(cu_all['date'], cu_all['total_abs'],label="cu")
#plt.plot(zn_all['date'], zn_all['total_abs'],label="zn")
#plt.plot(sn_all['date'], sn_all['total_abs'],label="sn")
plt.xticks(rotation=90)
#plt.title("总收益率")
plt.grid(linestyle=":")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=0, ncol=3, mode="expand", borderaxespad=0.)
plt.legend()
plt.tight_layout()

#tot_all = pd.concat([jm_all, j_all, rb_all, hc_all, cu_all, zn_all, sn_all])
tot_all = pd.concat([jm_j_all])
tot_all = tot_all.sort_values(by = ['time'])
tot_all = cal_plot(tot_all)
plt.plot(tot_all['time'], tot_all['total_abs'],label="TOTAL")
plt.plot(tot_all['time'], tot_all['total_abs_sell'],label="TOTAL")
plt.plot(tot_all['time'], tot_all['total_abs_buy'],label="TOTAL")

#计算每一年胜率：
for year_beg in ['2016', '2017', '2018', '2019', '2020', '2021']:
    year_end = str(int(year_beg)+1)
    tot_win, tot_lose = cal_win_rate(tot_all, year_beg, year_end)
    print(year_beg + '年胜率： '+str(tot_win/(tot_win+tot_lose)))
    print(year_beg + '交易点个数: '+str(tot_win+tot_lose))


#每一年最大回撤
for year_beg in ['2016', '2017', '2018', '2019', '2020', '2021']:
    year_end = str(int(year_beg)+1)
    tot_max_draw = cal_max_drawdown(tot_all, year_beg, year_end)
    print(str(year_beg)+"年最大回撤为："+str(tot_max_draw))

#计算每一年年化收益率
for year_beg in ['2016', '2017', '2018', '2019', '2020', '2021']:
    year_end = str(int(year_beg)+1)
    ann_return = cal_ann_return(tot_all, year_beg, year_end)
    print(str(year_beg)+'年化收益率：'+str(ann_return))

