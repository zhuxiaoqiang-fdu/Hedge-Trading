#!/usr/local/anaconda3/bin/python3

# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import datetime
import warnings
warnings.filterwarnings("ignore")

def normal(x):
    '''
    归一化计算公式（1）：min_max 方法
    '''
    m = x.max()
    n = x.min()
    return ((x-n)/(m-n))*2 -1

def normal2(x):
    '''
    归一化计算公式（2）: (x-mean)/std

    '''
    sigma = x.std()
    mu = x.mean()
    return (x-mu)/sigma

def get_trend(data_all, nk, freq):
    '''

    趋势性计算模块：
    data_all: Type: Dataframe, 所有品类的dataframe
    nk : int, 回溯k线的数量

    Returns:
    -------
    TYPE:dataframe
    ['date','symbol','contract','factor_name','factor_value']


    '''
    df_output = pd.DataFrame()
    for sym in data_all['symbol'].unique():
        df_temp = data_all[data_all['symbol']==sym] 
        df_temp['time'] = df_temp['date']
        df_temp['factor_freq'] = freq+'_'+str(nk)+'bars' 
        df_temp['factor_value'] = np.nan               
        df_temp['mtm'] = df_temp['close'].diff(1)  
        df_temp['mtm_shift'] = df_temp['mtm'].shift(1)  
        df_temp['turn'] = np.where(df_temp['mtm']*df_temp['mtm_shift']<0,1,0) 
        df_temp['ix'] = range(len(df_temp))  
        for i in range(nk, len(df_temp)):     
            wd = df_temp.iloc[i-nk+1:i+1]     
            poi = np.cumsum(list(wd[wd['turn']==1]['ix'].diff(1).dropna()))/nk  
            poi = np.insert(poi,0,0)
            poi = np.append(poi, 1)
            fq = np.diff(poi)  
            mtm = np.square(list(normal(wd['mtm']))) 
            Dn2 = []
            idx = poi*nk
            for j in range(len(idx)-1):
                Dn2.append(np.sum(mtm[int(idx[j]):int(idx[j+1])])) 
                
            fac = np.sum(Dn2/np.square(fq))*np.sum(np.square(fq)) 
            if df_temp['close'].iloc[i]>=df_temp['close'].iloc[i-nk+1]:
                df_temp['factor_value'].iloc[i] = fac
            else:
                df_temp['factor_value'].iloc[i] =-fac  

        print(df_temp)             
        df_temp = df_temp[['date', 'time', 'symbol','contract', 'factor_freq', 'factor_value']].dropna()
        df_temp['rank_value'] = np.nan
        for i in range(nk*3, len(df_temp)):
            fac = df_temp['factor_value'].iloc[i-nk+1:i+1]
            fac_norm = normal2(fac).iloc[-1]
            df_temp['rank_value'].iloc[i] = fac_norm    
        print(df_temp)
         
        df_temp['factor_name'] = "trend"
        print(df_temp[['date', 'time', 'symbol','contract','factor_name', 'factor_freq', 'factor_value', 'rank_value']])
        df_temp = df_temp[['date', 'time', 'symbol','contract','factor_name', 'factor_freq', 'factor_value', 'rank_value']].dropna()       
        df_output = pd.concat([df_output, df_temp])
        
    return df_output
   

def real_vol_method1(price_seq):
    '''
    波动率计算公式（1）
    '''
    price_seq = np.array(price_seq)
    rtn = np.divide(price_seq[1:], price_seq[0:(len(price_seq)-1)])
    lg_rtn = np.log(rtn)
    return np.square(lg_rtn).sum()


def real_vol_method2(price_seq):
    """
    波动率计算公式(2)
    annualized sample std of log return series
    https://www.zhihu.com/question/19770602    
    """
    n_observe = len(price_seq)
    price_seq = np.array(price_seq)
    rtn = np.divide(price_seq[1:], price_seq[0:(len(price_seq)-1)])
    lg_rtn = np.log(rtn)
    smpl_std = np.power((np.divide(np.square(lg_rtn).sum(), (n_observe-1))+
    np.divide(np.square(lg_rtn.sum()), (n_observe-1)*n_observe)), 0.5)

    return smpl_std*np.sqrt(25)


def get_vol(data_all, nk, freq):
    '''
    波动率计算模块：
 
    data_all : TYPE: dataframe
        DESCRIPTION: 所有品种的dataframe
    nk : TYPE: int
        DESCRIPTION: 回溯的k线数量

    Returns
    -------
    TYPE:dataframe
        ['date','symbol','contract','factor_name','factor_value']

    '''
    df_output = pd.DataFrame()
    for code in np.unique(data_all['symbol']):
        df1 = data_all[data_all['symbol'] == code].copy()[['date','time', 'symbol','contract', 'close']]
        df2 = df1.copy()
        df1['factor_name'] = 'volatility2'
        df2['factor_name'] = 'volatility3'
        df1['factor_freq'] = freq+'_'+str(nk)+'bars'
        df2['factor_freq'] = freq+'_'+str(nk)+'bars'
        df1['factor_value'] = np.nan
        df2['factor_value'] = np.nan
        df1['rank_value'] = np.nan
        df2['rank_value'] = np.nan

        for i in range(20, len(df1.index)):
            df1['factor_value'].iloc[i] = real_vol_method1(df1['close'].iloc[i-nk:i])
            df2['factor_value'].iloc[i] = real_vol_method2(df2['close'].iloc[i-nk:i])

        for i in range(nk*3, len(df1.index)):
            fac1 = df1['factor_value'].iloc[i-nk+1:i+1]
            fac2 = df2['factor_value'].iloc[i-nk+1:i+1]
            fac_norm1 = normal2(fac1).iloc[-1]
            fac_norm2 = normal2(fac2).iloc[-1]
            df1['rank_value'].iloc[i] = fac_norm1
            df2['rank_value'].iloc[i] = fac_norm2
        df_return = pd.concat([df1, df2])
        df_output = pd.concat([df_output, df_return])
    return df_output[['date', 'time', 'symbol','contract','factor_name', 'factor_freq', 'factor_value', 'rank_value']].dropna()   

def get_cor_ascol(data_all, nk):
    '''
    相关性计算模块：
    data_all : TYPE:dataframe
        DESCRIPTION: 所有品类的dataframe
    nk : TYPE:int
        DESCRIPTION: 回溯的k线数量

    Returns
    -------
    vs_name: meaning/value: corrlation between symbol and name
    TYPE:dataframe
        ['date','contract', 'symbol', 'name1vs_name2']
    '''
    code_list = np.unique(data_all['symbol'])
    col_list = ['date',  'contract', 'symbol']
    for i in range(len(code_list)-1):
        df1 = data_all[data_all['symbol'] == code_list[i]].copy()[['date',  'close']]
        df1[code_list[i]] = df1['close'].diff(1)/df1['close']
        for j in range(i+1, len(code_list)):
            col_list.append(code_list[i]+'_vs_'+code_list[j])
            df2 = data_all[data_all['symbol'] == code_list[j]].copy()[['date',  'close']]
            df2[code_list[j]] = df2['close'].diff(1)/df2['close']
            df3 = pd.merge(df1,df2,on=['date'],how='inner')
            cor = df3[code_list[i]].rolling(nk).corr(df3[code_list[j]])
            df3 = pd.concat([df3,cor], axis=1, join='outer').rename(columns={0:code_list[i]+'_vs_'+code_list[j]})
            data_all = pd.merge(data_all, df3, on=['date'],how='outer')
    return data_all[col_list]

list_ferrous = ['i','j','rb','hc','jm','ZC','SM','SF','FG','SA']
list_base_metals = ['cu','al','zn','ni','sn','ss']
list_precious_metals = ['au', 'ag']
list_petchems = ['ru','sc','fu','bu','l','TA','MA','v','pp','pg','eg','eb','PF']
list_ags = ['UR','SR','CF','RM','OI','WH','RS','JR','CY','AP','m','a','b','y','p','c','cs']
list_others = ['jd', 'CJ', 'sp']
list_index = ['IF','IH','IC']

all_commodity_list = list_ferrous + list_base_metals + list_precious_metals + list_petchems + list_ags + list_others

from data_pool.futures import futures_half_hourly
def main(nk, freq):
    '''
    主程序：输出所有结果
    '''
    data_all = futures_half_hourly(all_commodity_list ,'2015-01-01','2021-06-30',True)
#    data_all = futures_half_hourly(['au','ag'] ,'2015-01-01','2021-06-30',True)
    data_all['time'] = data_all['date'].dt.time
    if freq == 'daily' or freq == 'd':
        data_all = data_all[data_all['time']==datetime.time(15,0)]
    elif freq == 'half_hourly' or freq == 'h':
        data_all = data_all.copy()
    else:
        print("Please input the right frequency: daily or half-hourly, simple version : d or h")
    
    trend_output = get_trend(data_all, nk, freq)
    vol_output = get_vol(data_all, nk, freq)
    
    trend_output.to_csv('./trend_output_'+str(nk)+str(freq)+'.csv')
    vol_output.to_csv('./vol_output_'+str(nk)+str(freq)+'.csv')
    #print(get_cor_ascol(data_all, nk))

import argparse        
parser = argparse.ArgumentParser(description='number of bars to look at and path of input file')
parser.add_argument('--nk', type=int, help='input the number of bars')
parser.add_argument('--freq', type=str, help='daily or half-hourly, simple version: d or h')
#parser.add_argument('path',type=str, help='path of input file')
args = parser.parse_args()    

if __name__ =="__main__":
    main(args.nk, args.freq)
       
       
