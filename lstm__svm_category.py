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
from sklearn import svm
import sklearn
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense, Activation, Dropout 
from keras.layers.recurrent import LSTM 
from keras.models import Sequential
import warnings
warnings.filterwarnings("ignore")



#归一化函数
def normal(x):
    m = x.max()
    n = x.min()
    return ((x-n)/(m-n))*2 -1

def normal2(x):
    sigma = x.std()
    mu = x.mean()
    return (x-mu)/sigma

def normalise_windows(window_data): 
    normalised_data = [] 
    for window in window_data:   #window shape (sequence_length L ,)  即(51L,) 
        normalised_window = [((float(p) / float(window.iloc[0])) - 1) for p in window] 
        normalised_data.append(normalised_window) 
    return normalised_data

#合并日盘与夜盘数据
def intra_day(symbol, trade):
    if trade == "dce":
        df_day = dce_day
        df_night = dce_night
    elif trade == "shfe":
        df_day = shfe_day
        df_night = shfe_night
    elif trade == "zce":
        df_day = zce_day
        df_night = zce_night
    
    day = df_day[df_day['symbol']==symbol]
    night = df_night[df_night['symbol']==symbol]

    df_all = pd.concat([day, night])
  #  df_all = pd.concat([day])    
    df_all['date'] = pd.to_datetime(df_all['date'])
    df_all = df_all.sort_values(by = ['date','time'])
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
    return df_all

#计算趋势性指标
def cal_trend(data_all,nday):
    data_all.index = range(len(data_all['dif']))
    data_all['fac'] = np.nan
    data_all['tag'] = np.nan
    for i in range(nday,len(data_all['dif'])):
        il = range(i - nday, i)
        ts_mtm = data_all['dif'].iloc[il]
        ts_mtm = normal(ts_mtm)
        ts_mtm.index = range(len(ts_mtm))
        poi = [0]
        fq=[]
        Dn=[]
        #判断动量反转点并记录反转长度
        for j in range(len(ts_mtm)-1):
            if ts_mtm[j]*ts_mtm[j+1] < 0:
                poi.append((j+1)/(nday))
                
            elif ts_mtm[j] == 0 and j > 0 :
                if ts_mtm[j-1] * ts_mtm[j+1] < 0 :
                    poi.append((j+1)/(nday))
        poi.append(1)
        
        #计算每一反转小段的动量平方和
        for j in range(len(poi)-1):
            fq.append(poi[j+1]-poi[j])
            dn_tmp = 0
            for n in range(int(poi[j]*nday),int(poi[j+1]*nday)):
                dn_tmp = dn_tmp + ts_mtm[n]**2
            Dn.append(dn_tmp)
            
        #计算反转长度平方和    
        ent = 0
        for p in fq:
            ent = ent + p**2
        #计算趋势性指标fac
        fac = 0
        for k in range(len(Dn)):
            fac = fac + Dn[k]/((fq[k])**2)
        fac = fac * ent
        data_all['fac'][i] = fac
        
        #添加动量为正负的tag
        if np.mean(ts_mtm) < 0:
            data_all['tag'][i] = -1
        elif np.mean(ts_mtm) > 0:
            data_all['tag'][i] = 1
        else:
            data_all['tag'][i] = 0
            
    data_all = data_all.dropna(axis = 0, how = 'any')
    return data_all    
  

def cal_trend_rank(data_all, nday):
    data_all.index = range(len(data_all['close']))
  #  data_all = data_all.dropna(axis = 0, how = 'any')
  #  data_all['fac'] = np.nan
    data_all['tag'] = np.nan
    for i in range(nday, len(data_all['close'])):
        ts_close = data_all['close'].iloc[range(i-nday, i)]
   #     print(ts_close)
    #    ts_close = normal(ts_close)
        ts_close.index = range(nday)
        cor = pd.concat([pd.Series(range(nday)), ts_close], axis=1).corr()
        data_all['tag'][i] = cor[0]['close']
    return data_all

def cal_MA(data_all):
    data_all['MA_5day'] = data_all['close'].rolling(5).mean()
    data_all['MA_10day'] = data_all['close'].rolling(10).mean()
    return data_all
    

 
#计算绝对的累计收益金额变化
def cal_plot(data_all):
    data_all[['total_abs', 'total_abs_buy', 'total_abs_sell']] = 0.0
    data_all.index = range(len(data_all['abs_buy']))
    strftime = datetime.datetime.strptime("2016-01-01", "%Y-%m-%d")
    del_ind = [0]
    for i in range(1,len(data_all['total_abs'])):
        if data_all['date'][i] > strftime:
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
    data_all.index= range(len(data_all['date']))
    return data_all    
    
#计算每一年的胜率
def cal_win_rate(data_all, year_beg, year_end):
    data_temp = data_all.loc[(data_all['date'] > datetime.datetime.strptime(year_beg+"-01-01", "%Y-%m-%d")) & (data_all['date'] < datetime.datetime.strptime(year_end+"-01-01", "%Y-%m-%d"))]
    win_buy = (data_temp['abs_buy'] > 0).astype(int).sum(axis=0)
    win_sell = (data_temp['abs_sell'] > 0).astype(int).sum(axis=0)            
    lose_buy = (data_temp['abs_buy'] < 0).astype(int).sum(axis=0)
    lose_sell = (data_temp['abs_sell'] < 0).astype(int).sum(axis=0)
    return (win_buy + win_sell), (lose_buy + lose_sell)
    
#计算每一年的最大回撤
def cal_max_drawdown(data_all, year_beg, year_end):
    data_temp = data_all.copy().loc[(data_all['date'] > datetime.datetime.strptime(year_beg+"-01-01", "%Y-%m-%d")) & (data_all['date'] < datetime.datetime.strptime(year_end+"-01-01", "%Y-%m-%d"))]
    max_draw = 1
    for i in data_temp.index[1:]:
        max_draw = min(max_draw, (data_temp['total_abs'][i]+5000000)/(data_temp['total_abs'].loc[:i]+5000000).max()) #假设有100w的钱，交易保证金为10w
    max_draw = 1 - max_draw     
    return max_draw

#计算每一年的年收益率
def cal_ann_return(data_all, year_beg, year_end):
    data_temp = data_all.loc[(data_all['date'] > datetime.datetime.strptime(year_beg+"-01-01", "%Y-%m-%d")) & (data_all['date'] < datetime.datetime.strptime(year_end+"-01-01", "%Y-%m-%d"))]
    return (data_temp['total_abs'].iloc[-1]+5000000)/(data_temp['total_abs'].iloc[0]+5000000)
    

def buy(res, data_all, beg, hold_unit, hold_price, hold_time, trans_tick):
    hold_unit[res].append(1)
    hold_price[res].append(data_all['close'][beg] + 2*trans_tick[res])
    hold_time[res].append(beg)
    return hold_unit , hold_price, hold_time

def sell(res, data_all, beg, hold_unit, hold_price, hold_time, trans_tick):
    hold_unit[res].append(-1)
    hold_price[res].append(data_all['close'][beg] - 2*trans_tick[res])
    hold_time[res].append(beg)
    return hold_unit, hold_price, hold_time
    
def end_buy(res, data_all, beg, hold_unit, hold_price, hold_time, trans_tick, trans_unit, fac):
#    data_all['abs_buy'][beg] += (data_all['close'][beg] - data_all['close'][hold_time[res][0]]) * trans_unit[res] * 100000//(data_all['close'][hold_time[res][0]] * trans_unit[res] * 0.1)
    data_all['abs_buy'][beg] += (data_all['close'][beg] - hold_price[res][0]) * trans_unit[res] * 100000//(data_all['close'][hold_time[res][0]] * trans_unit[res] * 0.1)
    data_all['beg_buy'][beg] += ' ' + str(data_all['date'][hold_time[res][0]]) + ' ' + str(data_all['time'] [hold_time[res][0]])
    data_all['fac_beg'][beg] = data_all['fac'][hold_time[res][0]]
    data_all['volume_beg'][beg] = data_all['volume'][hold_time[res][0]]
    data_all['openinterest_beg'][beg] = data_all['openinterest'][hold_time[res][0]]
    data_all['time_delta'][beg] = beg - hold_time[res][0]
    hold_time[res].pop(0)
    hold_unit[res].pop(0)
    hold_price[res].pop(0)
    return hold_unit, hold_price, hold_time, data_all
    
def end_sell(res, data_all, beg, hold_unit, hold_price, hold_time, trans_tick, trans_unit, fac):
#    data_all['abs_sell'][beg] += (data_all['close'][hold_time[res][0]] - data_all['close'][beg]) * trans_unit[res] * 100000//(data_all['close'][hold_time[res][0]] * trans_unit[res] * 0.1)
    data_all['abs_sell'][beg] += (hold_price[res][0] - data_all['close'][beg]) * trans_unit[res] * 100000//(data_all['close'][hold_time[res][0]] * trans_unit[res] * 0.1)
    data_all['beg_sell'][beg] += ' ' + str(data_all['date'][hold_time[res][0]]) + ' ' + str(data_all['time'] [hold_time[res][0]])
    data_all['fac_beg'][beg] = data_all['fac'][hold_time[res][0]]
    data_all['volume_beg'][beg] = data_all['volume'][hold_time[res][0]]
    data_all['openinterest_beg'][beg] = data_all['openinterest'][hold_time[res][0]]
    data_all['time_delta'][beg] = beg -hold_time[res][0]
    hold_time[res].pop(0)
    hold_unit[res].pop(0)
    hold_price[res].pop(0)
    return hold_unit, hold_price, hold_time, data_all
        
def force_end_buy(res, data_all, beg, hold_unit, hold_price, hold_time, trans_tick, trans_unit, nday_end, stbk_buy, upbk_buy, fac):
    delta = 0
    for k in range(len(hold_unit[res])):
        if ((data_all['close'][beg] - hold_price[res][k-delta])/hold_price[res][k-delta] < stbk_buy) or ((data_all['close'][beg] - hold_price[res][k-delta])/hold_price[res][k-delta] > upbk_buy) or ((beg - hold_time[res][k-delta]) > nday_end):
            hold_unit, hold_price, hold_time, data_all = end_buy(res, data_all, beg, hold_unit, hold_price, hold_time, trans_tick, trans_unit, fac)
            delta += 1
    return hold_unit, hold_price, hold_time, data_all

def force_end_sell(res, data_all, beg, hold_unit, hold_price, hold_time, trans_tick, trans_unit, nday_end, stbk_sell, upbk_sell, fac):
    delta = 0
    for k in range(len(hold_unit[res])):
        if ((hold_price[res][k-delta] - data_all['close'][beg])/data_all['close'][beg] < stbk_sell) or ((hold_price[res][k-delta] - data_all['close'][beg])/data_all['close'][beg] > upbk_sell) or ((beg - hold_time[res][k-delta]) > nday_end):
            hold_unit, hold_price, hold_time, data_all = end_sell(res, data_all, beg, hold_unit, hold_price, hold_time, trans_tick, trans_unit, fac)
            delta += 1
    return hold_unit, hold_price, hold_time, data_all

def force_end_trans_month(res, data_all, beg, hold_unit, hold_price, hold_time, trans_tick, trans_unit, fac):
    if data_all['contract'][beg] != data_all['contract'][beg+1]:
        if sum(hold_unit[res]) > 0:
            for k in range(len(hold_unit[res])):
                hold_unit, hold_price, hold_time, data_all = end_buy(res, data_all, beg, hold_unit, hold_price, hold_time, trans_tick, trans_unit, fac)
        elif sum(hold_unit[res]) < 0:
            for k in range(len(hold_unit[res])):
                hold_unit, hold_price, hold_time, data_all = end_sell(res, data_all, beg, hold_unit, hold_price, hold_time, trans_tick, trans_unit, fac)
    return hold_unit, hold_price, hold_time, data_all             

def svm_judge(data_all, beg, nday_tran, nday_end):
    data_train = data_all.copy().loc[beg-nday_tran-nday_end:beg-nday_end]
 #   print(data_train)
 #   global fff
 #   fff=data_train.copy()
    data_train = data_train.dropna(thresh=1,subset=['abs_buy','abs_sell'])
    data_train['action'] = np.nan
#    data_train['rat_buy'].fillna(0)
#    data_train['rat_sell'].fillna(0)
#    data_train['action'] = 
  #  print(data_train)
    for i in data_train.index:
  #      print('ddd'+str(data_train['rat_buy'][i]))
        if float(data_train['abs_buy'][i]) < 0: 
            data_train['action'][i] = -1
        elif float(data_train['abs_buy'][i]) > 0: 
            data_train['action'][i] = 1
        elif float(data_train['abs_sell'][i]) < 0: 
            data_train['action'][i] = 1
        elif float(data_train['abs_sell'][i]) > 0: 
            data_train['action'][i] = -1
    x = data_train['position']
    y = data_train['action']
    train_data,test_data,train_label,test_label =train_test_split(x,y, random_state=1, train_size=0.6,test_size=0.4) #sklearn.model_selection.    
    classifier=svm.SVC(C=1,kernel='rbf',gamma=10,decision_function_shape='ovo') # ovr:一对多策略
#    print('inputtt:')
#    print(np.array(train_data).reshape(-1,1))
#    print(train_label)
    classifier.fit(np.array(train_data).reshape(-1,1),train_label.ravel()) #ravel函数在降维时默认是行序优先
    #4.计算svc分类器的准确率
  #  print("训练集：",classifier.score(np.array(train_data).reshape(-1,1),train_label))
  #  print("测试集：",classifier.score(np.array(test_data).reshape(-1,1),test_label))
    hat=classifier.predict(np.array(data_all['position'][beg]).reshape(-1,1)) 
    return hat

#使用长短期记忆神经网络LSTM进行涨跌判断
def LSTM_judge(data_all, beg, nday_train, seq_len):
    ts_data = data_all['dif'].loc[beg-nday_train: beg]
    sequence_length = seq_len + 1
    result = []
    for index in range(len(ts_data)-sequence_length):
        result.append(ts_data[index: index + sequence_length])
  #  print(result)
    result = np.array(result)
    result = normal2(result)    
    np.random.shuffle(result)
    x_train = result[:, :-1]
    y_train = result[:, -1]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) 
    x_test = data_all['dif'].loc[beg-seq_len: beg]
  #  print(x_test)
  #  global xxx
  #  xxx=x_test
    x_test = np.array(x_test) 
    x_test = np.reshape(x_test, (1, x_test.shape[0], 1))
    return x_train, y_train, x_test

def build_model(layers):  #layers [1,50,100,1] 
    model = Sequential()
    model.add(LSTM(input_dim=layers[0], units=layers[1], return_sequences=True)) 
    model.add(Dropout(0.2))
    model.add(LSTM(layers[2],return_sequences=False)) 
    model.add(Dropout(0.2))
    model.add(Dense(units=layers[3])) 
    model.add(Activation("linear"))
#    start = time.time() 
    model.compile(loss="mse", optimizer="rmsprop") 
#    print("Compilation Time : ", time.time() - start) 
    return model

def predict_point_by_point(model, data): 
    predicted = model.predict(data) 
   # print('predicted shape:',np.array(predicted).shape)  #(412L,1L) 
    predicted = np.reshape(predicted, (predicted.size,)) 
    return predicted
     

#进行回测
def backtest(data_all,nday):
    data_all[['price_high', 'price_low', 'price_avg']] = np.nan
    data_all[['fac_high', 'fac_low', 'fac_avg', 'sig', 'fac_beg', 'time_delta', 'volume_beg', 'openinterest_beg']] = np.nan
    data_all[['beg_buy', 'beg_sell']] = ''
    data_all[['abs_buy', 'abs_sell']] = 0.0
    data_all['position'] = '0'
    stbk_sell = -0.08 #止损点
    stbk_buy = -0.008
    upbk_buy = 0.015
    upbk_sell = 0.015 #止盈点
    nday_end = 15
    nday_tran = 1500
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
#   for i in range(nday+1, len(data_all['dif']) - nday_end):
    for i in range(12750, len(data_all['dif']) - nday_end):
    
        fac = data_all['fac'][i]
        fac_avg = np.mean(data_all['fac'].loc[i-nday:i])
        fac_std = np.std(data_all['fac'].loc[i-nday:i])
        price_avg = np.mean(data_all['close'].loc[i-nday:i])
        price_std = np.std(data_all['close'].loc[i-nday:i])
        data_all['price_high'][i] = price_avg + price_std
        data_all['price_low'][i] = price_avg - price_std
        data_all['price_avg'][i] = price_avg
        
        MA_5day = np.mean(data_all['close'].loc[i-5:i])
        MA_10day = np.mean(data_all['close'].loc[i-10:i])
        MA_5day_bf = np.mean(data_all['close'].loc[i-6:i-1])
        MA_10day_bf = np.mean(data_all['close'].loc[i-11:i-1])
        fac_bf1 = data_all['fac'][i-1]
     #   fac_bf2 = data_all['fac'][i-2]
     #   fac_bf3 = data_all['fac'][i-3]
        
        data_all['fac_high'][i] = fac_avg + fac_std
        data_all['fac_low'][i] = fac_avg - fac_std
        data_all['fac_avg'][i] = fac_avg
        
        s = data_all['contract'][i]
        remove_digits = str.maketrans('', '', digits)
        res1 = s.translate(remove_digits)
           
        '''
        if fac_bf1 < -0.75 and fac > fac_bf1 and data_all['close'][i] > data_all['close'][i-1] and MA_5day > MA_10day and MA_5day_bf < MA_10day_bf:
            data_all['position'][i]='down'
            sig = 1.0
            if i > nday_tran + nday_end + nday:              
                sig = 0.0
                for x in range(nday_tran):
                    if (not pd.isnull(data_all['abs_buy'][i-nday_tran+x])) and data_all['position'][i-nday_tran+x] == 'down':
                        sig = sig + math.atan(x+1)*data_all['abs_buy'][i-nday_tran+x] +  2*trans_tick[res1]*trans_unit[res1]*100000//(data_all['close'][i]* trans_unit[res1] * 0.1)
                    if (not pd.isnull(data_all['abs_sell'][i-nday_tran+x])) and data_all['position'][i-nday_tran+x] == 'down':
                        sig = sig - math.atan(x+1)*data_all['abs_sell'][i-nday_tran+x] +  2*trans_tick[res1]*trans_unit[res1]*100000//(data_all['close'][i]* trans_tick[res1] * 0.1)
            #      if MA_5day > MA_10day:   
            if sig > 0:
                if sum(hold_unit[res1]) >=0:
             #       hold_unit, hold_price, hold_time = buy(res1, data_all, i, hold_unit, hold_price, hold_time, trans_tick)
                     pass
                     data_all['fac_beg'][i] = fac
                else:
                     hold_unit, hold_price, hold_time, data_all = end_sell(res1, data_all, i, hold_unit, hold_price, hold_time, trans_tick, trans_unit, fac)
            else:
                if sum(hold_unit[res1]) > 0:
                    hold_unit, hold_price, hold_time, data_all = end_buy(res1, data_all, i, hold_unit, hold_price, hold_time, trans_tick, trans_unit, fac)
                else:
                    hold_unit, hold_price, hold_time = sell(res1, data_all, i, hold_unit, hold_price, hold_time, trans_tick)
                    data_all['fac_beg'][i] = fac
                
        elif fac_bf1 > 0.75 and fac < fac_bf1 and data_all['close'][i] < data_all['close'][i-1] and MA_5day < MA_10day and MA_5day_bf > MA_10day_bf:
    #    elif MA_5day < MA_10day:   
            data_all['position'][i]='up'
            sig = -1.0
            if i > nday_tran+nday_end+nday:                
                sig = 0.0
                for x in range(nday_tran):
                    if (not pd.isnull([data_all['abs_buy'][i-nday_tran+x]])) and data_all['position'][i-nday_tran+x] == 'up':
                        sig = sig + math.atan(x+1)*data_all['abs_buy'][i-nday_tran+x] +  2*trans_tick[res1]*trans_unit[res1]*100000//(data_all['close'][i] * 0.1)
                    elif (not pd.isnull(data_all['abs_sell'][i-nday_tran+x])) and data_all['position'][i-nday_tran+x] == 'up':
                        sig = sig - math.atan(x+1)*data_all['abs_sell'][i-nday_tran+x] +  2*trans_tick[res1]*trans_unit[res1]*100000//(data_all['close'][i] * 0.1)
            if sig > 0:
                if sum(hold_unit[res1]) >= 0:
                    hold_unit, hold_price, hold_time = buy(res1, data_all, i, hold_unit, hold_price, hold_time, trans_tick)
                    data_all['fac_beg'][i] = fac
            #    else:
            #        hold_unit, hold_price, hold_time, data_all = end_sell(res1, data_all, i, hold_unit, hold_price, hold_time, trans_tick, trans_unit, fac)
            else:
                if sum(hold_unit[res1]) > 0:
                    hold_unit, hold_price, hold_time, data_all = end_buy(res1, data_all, i, hold_unit, hold_price, hold_time, trans_tick, trans_unit, fac)
                else:
                    hold_unit, hold_price, hold_time = sell(res1, data_all, i, hold_unit, hold_price, hold_time, trans_tick)
                    data_all['fac_beg'][i] = fac
        '''        
            
        
        #寻找布林上轨线，并在未来nday_end内买入或者默认卖出，并调整出手数
        if data_all['close'][i] > price_avg + price_std and data_all['fac'][i] > 0 and data_all['fac'][i] > fac_avg + fac_std:
            data_all['position'][i]='up'
            sig = -1.0
            print(data_all['date'][i])
            if i > nday_tran+nday_end+1:  
                epochs = 1
                seq_len = 50
                x_train, y_train, x_test = LSTM_judge(data_all, i, nday_tran, seq_len)
                model = build_model([1, 50, 100, 1])
                model.fit(x_train,y_train,batch_size=512,epochs=epochs,validation_split=0.05)               
                point_pre = predict_point_by_point(model, x_test)
                if point_pre[0] > 0:
                    sig = 1
            if sig > 0 and data_all['dif'][i] > 0 :
                if sum(hold_unit[res1]) >= 0:
                    hold_unit, hold_price, hold_time = buy(res1, data_all, i, hold_unit, hold_price, hold_time, trans_tick)
                else:
                    hold_unit, hold_price, hold_time, data_all = end_sell(res1, data_all, i, hold_unit, hold_price, hold_time, trans_tick, trans_unit, fac)
            elif sig <= 0 and data_all['dif'][i] < 0 :
                if sum(hold_unit[res1]) > 0:
                    hold_unit, hold_price, hold_time, data_all = end_buy(res1, data_all, i, hold_unit, hold_price, hold_time, trans_tick, trans_unit, fac)
                else:
                    hold_unit, hold_price, hold_time = sell(res1, data_all, i, hold_unit, hold_price, hold_time, trans_tick)

            data_all['sig'][i] = point_pre[0]    
        
        
        #寻找布林下轨线，并在未来nday_end内卖出或者默认买入，并调整出手数
        elif data_all['close'][i] < price_avg - price_std and data_all['fac'][i] < 0 and fac < fac_avg - fac_std:
            data_all['position'][i]='down'
            sig = 1.0
            if i > nday_tran + nday_end + 1:    
             if i > nday_tran+nday_end+1:  
                epochs = 1
                seq_len = 50
                x_train, y_train, x_test = LSTM_judge(data_all, i, nday_tran, seq_len)
                model = build_model([1, 50, 100, 1])
                model.fit(x_train,y_train,batch_size=512,epochs=epochs,validation_split=0.05)               
                point_pre = predict_point_by_point(model, x_test)
                if point_pre[0] < 0:
                    sig = -1                        
            if sig >= 0 and data_all['dif'][i] > 0 :
                if sum(hold_unit[res1]) >= 0:
                    hold_unit, hold_price, hold_time = buy(res1, data_all, i, hold_unit, hold_price, hold_time, trans_tick)
                else:
                    hold_unit, hold_price, hold_time, data_all = end_sell(res1, data_all, i, hold_unit, hold_price, hold_time, trans_tick, trans_unit, fac)
            elif sig <0 and data_all['dif'][i] < 0 :
                if sum(hold_unit[res1]) > 0:
                    hold_unit, hold_price, hold_time, data_all = end_buy(res1, data_all, i, hold_unit, hold_price, hold_time, trans_tick, trans_unit, fac)
                else:
                    hold_unit, hold_price, hold_time = sell(res1, data_all, i, hold_unit, hold_price, hold_time, trans_tick)  
            data_all['sig'][i] = point_pre[0]         
        
            
        if sum(hold_unit[res1]) > 0:
            hold_unit, hold_price, hold_time, data_all = force_end_buy(res1, data_all, i, hold_unit, hold_price, hold_time, trans_tick, trans_unit, nday_end, stbk_buy, upbk_buy, fac)
        if sum(hold_unit[res1]) < 0:
            hold_unit, hold_price, hold_time, data_all = force_end_sell(res1, data_all, i, hold_unit, hold_price, hold_time, trans_tick, trans_unit, nday_end, stbk_sell, upbk_sell, fac)
        if sum(hold_unit[res1]) != 0:
            hold_unit, hold_price, hold_time, data_all = force_end_trans_month(res1, data_all, i, hold_unit, hold_price, hold_time, trans_tick, trans_unit, fac)
    return data_all        
              

dce_day = pd.read_csv('/Users/zhuxiaoqiang/Desktop/trend/in_day_data_update/convert_data.dce.day.csv', delim_whitespace=True )  
dce_night = pd.read_csv('/Users/zhuxiaoqiang/Desktop/trend/in_day_data_update/convert_data.dce.night.csv', delim_whitespace=True )
shfe_day = pd.read_csv('/Users/zhuxiaoqiang/Desktop/trend/in_day_data_update/convert_data.shfe.day.csv', delim_whitespace=True)
shfe_night = pd.read_csv('/Users/zhuxiaoqiang/Desktop/trend/in_day_data_update/convert_data.shfe.night.csv', delim_whitespace=True)
zce_day = pd.read_csv('/Users/zhuxiaoqiang/Desktop/trend/in_day_data_update/convert_data.zce.day.csv', delim_whitespace=True)
zce_night = pd.read_csv('/Users/zhuxiaoqiang/Desktop/trend/in_day_data_update/convert_data.zce.night.csv', delim_whitespace=True)


l_all = intra_day("cu","shfe")
v_all = intra_day("al","shfe")
pp_all = intra_day("zn","shfe")
ta_all = intra_day("pb","shfe")
#MA_all = intra_day("ni","shfe")
bu_all = intra_day("sn","shfe")


l_all['dif'] = l_all['close'].diff(periods=1)
v_all['dif'] = v_all['close'].diff(periods=1)
pp_all['dif'] = pp_all['close'].diff(periods=1)
ta_all['dif'] = ta_all['close'].diff(periods=1)
#MA_all['dif'] = MA_all['close'].diff(periods=1)
bu_all['dif'] = bu_all['close'].diff(periods=1)


l_all = l_all.dropna(axis = 0, how = 'any')
v_all = v_all.dropna(axis = 0, how = 'any')
pp_all = pp_all.dropna(axis = 0, how = 'any')
ta_all = ta_all.dropna(axis = 0, how = 'any')
#MA_all = MA_all.dropna(axis = 0, how = 'any')
bu_all = bu_all.dropna(axis = 0, how = 'any')




nday = 45  #往前滚动k线数量

#计算趋势性指标
l_all = cal_trend(l_all,nday)
v_all = cal_trend(v_all,nday)
pp_all = cal_trend(pp_all,nday)
ta_all = cal_trend(ta_all,nday)
#MA_all = cal_trend(MA_all,nday)
bu_all = cal_trend(bu_all,nday)
    
'''
l_all = cal_trend_rank(l_all,nday)
v_all = cal_trend_rank(v_all,nday)
pp_all = cal_trend_rank(pp_all,nday)
ta_all = cal_trend_rank(ta_all,nday)
#MA_all = cal_trend(MA_all,nday)
bu_all = cal_trend_rank(bu_all,nday)
'''

l_all['fac'] = l_all['fac'] * l_all['tag']
v_all['fac'] = v_all['fac'] * v_all['tag']
pp_all['fac'] = pp_all['fac'] * pp_all['tag']
ta_all['fac'] = ta_all['fac'] * ta_all['tag']
#MA_all['fac'] = MA_all['fac'] * MA_all['tag']
bu_all['fac'] = bu_all['fac'] * bu_all['tag']




#计算MA
l_all = cal_MA(l_all)
v_all = cal_MA(v_all)
pp_all = cal_MA(pp_all)
ta_all = cal_MA(ta_all)
bu_all = cal_MA(bu_all)


#进行回测
l_all = backtest(l_all, nday)
v_all = backtest(v_all, nday)
pp_all = backtest(pp_all, nday)
ta_all = backtest(ta_all, nday)
#MA_all = backtest(MA_all, nday)
bu_all = backtest(bu_all, nday)



#去除没有交易的点
l_all = l_all.dropna(axis=0,how='all',subset=['abs_buy', 'abs_sell'])
v_all = v_all.dropna(axis=0,how='all',subset=['abs_buy', 'abs_sell'])
pp_all = pp_all.dropna(axis=0,how='all',subset=['abs_buy', 'abs_sell'])
ta_all = ta_all.dropna(axis=0,how='all',subset=['abs_buy', 'abs_sell'])
#MA_all = MA_all.dropna(axis=0,how='all',subset=['abs_buy', 'abs_sell'])
bu_all = bu_all.dropna(axis=0,how='all',subset=['abs_buy', 'abs_sell'])



#计算收益率走势
money = 10000000 #交易本金
l_all = cal_plot(l_all)
v_all = cal_plot(v_all)
pp_all = cal_plot(pp_all)
ta_all = cal_plot(ta_all)
#MA_all = cal_plot(MA_all)
bu_all = cal_plot(bu_all)





#to_csv输出
#l_all.to_csv(r'C:\Users\Administrator\Desktop\trend\in_day_data_update\l.csv')
#v_all.to_csv(r'C:\Users\Administrator\Desktop\trend\in_day_data_update\v.csv')
#pp_all.to_csv(r'C:\Users\Administrator\Desktop\trend\in_day_data_update\pp.csv')
#ta_all.to_csv(r'C:\Users\Administrator\Desktop\trend\in_day_data_update\ta.csv')
#MA_all.to_csv(r'C:\Users\Administrator\Desktop\trend\in_day_data_update\MA.csv')
#bu_all.to_csv(r'C:\Users\Administrator\Desktop\trend\in_day_data_update\bu.csv')



#收益率曲线画图
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.plot(l_all['date'], l_all['total_abs'],label="cu")
plt.plot(v_all['date'], v_all['total_abs'],label="al")
plt.plot(pp_all['date'], pp_all['total_abs'],label="zn")
plt.plot(ta_all['date'], ta_all['total_abs'],label="pb")
#plt.plot(MA_all['date'], MA_all['total_abs'],label="ni")
plt.plot(bu_all['date'], bu_all['total_abs'],label="sn")
plt.xticks(rotation=90)
#plt.title("总收益率")
plt.grid(linestyle=":")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=0, ncol=3, mode="expand", borderaxespad=0.)
plt.legend()
plt.tight_layout()

#tot_all = pd.concat([l_all, v_all, pp_all, ta_all, MA_all, bu_all])
tot_all = pd.concat([l_all, v_all, pp_all, ta_all, bu_all])
tot_all = tot_all.sort_values(by = ['date','time'])
tot_all = cal_plot(tot_all)
plt.plot(tot_all['date'], tot_all['total_abs'],label="TOTAL")
plt.plot(tot_all['date'], tot_all['total_abs_sell'],label="TOTAL")
plt.plot(tot_all['date'], tot_all['total_abs_buy'],label="TOTAL")
#tot_all.to_csv(r'C:\Users\Administrator\Desktop\trend\in_day_data_update\tot_new_0.013buy_29sig.csv')

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

