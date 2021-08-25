# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 13:17:12 2021

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 17:20:06 2021

@author: Administrator
"""

startTime = '2021-05-01 00:00:00'
endTime = '2021-07-16 23:59:59'
import requests
import  datetime
import pymysql
from requests.adapters import HTTPAdapter
import time
import pandas as pd
import re
from aip import AipNlp
import argparse
from aip import AipNlp
import numpy as np 
APP_ID = '24574902'
API_KEY = 'xmRAlwg6H9SyqEnoRXQkuaiR'
SECRET_KEY = 'BdPl7GPiUmmywsCZbYGCE00NXFAhARPs'

client = AipNlp(APP_ID, API_KEY, SECRET_KEY)
end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def format_jin10(df):
    rejected_words =  ['行情', '直播', '开盘', '收盘', '复盘','早盘', '图示', '机构观点', 
                           '期货报告', '要闻汇总', '现货报价', '研报', '热图', '盘前必读',
                           '一周精选', '期货日历', '金十期货整理', '期货早高峰', '科普',
                           '今日重点关注', '期市热点分析', '期货', '证券', '品种交易逻辑',
                           '库存调查统计', '周评', '点击阅读','现货价格']
    
    filter_out_content = ['欢迎查看', '阅读', '点击', '链接', '分析师']
    # print(len(re.findall(r'\u3010(.*?)\u3011',df['data'][4])))
    df['prefix'] = df['data'].apply(lambda x : 'no_preflix' if len(re.findall(r'\u3010(.*?)\u3011',str(x)))==0 else re.findall(r'\u3010(.*?)\u3011',str(x))[0])
    
    for keyword in rejected_words:
        df = df[(df['prefix'].str.contains(keyword)==False)]
    for k in filter_out_content:
        df = df[(df['data'].str.contains(k)==False)]
        
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values(by = 'time', ascending = True)
    df = df.reset_index().drop(columns=['index'])
    return df
def NLP(processed_df):
    df = processed_df.copy()
    df['org'] = np.nan
    df['per'] = np.nan
    df['org']=df['org'].astype('object')
    df['per']=df['per'].astype('object')
    for i in range(len(df)):
        text = df['data'].iloc[i].encode("gbk", "ignore").decode("gbk","ignore")
        try:
            f_results = client.lexer(text)['items']
        except:
            time.sleep(15)
            f_results = client.lexer(text)['items']
        org = []
        loc = []
        per = []
        verb = []
        for f in f_results:
            if f['ne'] == 'ORG':
                org.append(f['item'])
                org.append(f['item'])
            
            if f['ne'] == 'PER':
                per.append(f['item'])
        df['org'].iloc[i] = org
        df['per'].iloc[i] = per
    return df 
 
def main():  
    channel_dict = {7.0:'NA',8.0:'NA',9.0:'NA',11.0:'综合',19.0:'油脂油料',20.0:'钢矿',
                    21.0:'煤炭',22.0:'油气',23.0:'贵金属',24.0:'有色',25.0:'化工',26.0:'碳排放',
                    27.0:'糖棉果蛋',28.0:'谷物',29.0:'生猪',30.0:'宏观',31.0:'平衡表',32.0:'热图科普'}
    
    
    
    url = "https://qh-flash-api.jin10.com/get_flash_list?channel=-1"
    header = {
            "x-app-id": "KxBcVoDHStE6CUkQ",
            "x-version": "1.0.0",
        }
    params = {
        endTime : end_time
        }
    news_dict = requests.get(url,headers = header).json()['data']
    
    #id , time ,type ,data(pic, content), important , tags , channel ,remark
    for i in range(len(news_dict)):
        if 'content' in news_dict[i]['data'] and news_dict[i]['data']['content'] != None:
            news_dict[i]['data'] = news_dict[i]['data']['content'].replace('<b>','').replace('</b>','').replace('<br />','').replace('<br/>','')
            
        else:
            news_dict[i]['data'] = None
        commodity_list =[]
        
        if 'channel' in news_dict[i] and news_dict[i]['channel'] !=None : 
            for t in news_dict[i]['channel']: 
                commodity_tpye = channel_dict[t]
                commodity_list.append(commodity_tpye)
            news_dict[i]['sector'] = commodity_list 
            
    
    data = pd.DataFrame(news_dict)
    data.drop(['id','type','tags','remark'],axis =1,inplace=True)


    formated = format_jin10(data)
    # print(formated)
    data = NLP(formated)
    print(data)
    # data.to_excel('test.xlsx')

if __name__ == '__main__':
    while True:
        time.sleep(15)
        main()
    
