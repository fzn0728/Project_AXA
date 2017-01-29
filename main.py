# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 11:03:09 2017

@author: Chandler
"""

import os
import pandas as pd
import numpy as np
import datetime
import dateutil.relativedelta

if __name__ == '__main__':
    ### Set working directory
    os.getcwd()
    file_path = r'C:/Users/Chandler/Desktop/application/Interview/AXA/Project_AXA'
    os.chdir(file_path)
    ### Import the data and clean the format
    ROA_df = pd.read_csv('ROA.csv',sep='\s+')
    ROA_df['datadate'] = pd.to_datetime(ROA_df['datadate'],format="%d%b%Y")
    ROA_df = ROA_df.dropna(axis=0)
    # datetime.datetime.strptime(ROA_df['datadate'],'%Y%m')
    # ROA_df['datadate_1'] = [x+dateutil.relativedelta.relativedelta(months=1)-dateutil.relativedelta.relativedelta(days=1) for x in ROA_df['datadate']]
    return_df = pd.read_csv('stockreturns.csv',sep='\s+',converters={'firm_id':np.int64,'ret':np.float64,'year':np.str,'month':np.str},engine='python')
    return_df = return_df[return_df['month']!='None'] # Drop None data
    return_df['datadate'] = pd.to_datetime((return_df['year'] + return_df['month']), format='%Y%m')
    return_df['datadate_1'] = return_df['datadate']+dateutil.relativedelta.relativedelta(months=1)-dateutil.relativedelta.relativedelta(days=1)
    return_df['datadate_1'] = [x+dateutil.relativedelta.relativedelta(months=1)-dateutil.relativedelta.relativedelta(days=1) for x in return_df['datadate']]
        
    
    num_group = ROA_df.groupby('datadate').count()


    top_ROA = ROA_df.groupby('datadate')['ROA'].quantile(0.8)
    bottom_ROA = ROA_df.groupby('datadate')['ROA'].quantile(0.2)




df = pd.DataFrame(np.array([[1, 1], [2, 10], [3, 100], [4, 100]]),columns=['a', 'b'])    
    
'''    
    
                            ,parse_dates={'datetime':['year']},date_parser=lambda x: pd.datetime.strptime(x, '%Y'))
    
    
    
    
    
    
    return_df['datadate'] = pd.to_datetime(return_df[['year','month']])
    datetime.date(year=return_df['year'].values,month=return_df['month'].values)
    
    
    t = ROA_df.groupby('datadate').quantile(0.2)['ROA']
    
    num_group = ROA_df.groupby('datadate').count()
    
    
    
    
    
    
    
    
    a = pd.qcut(ROA_df['ROA'].values,5)
    
    
    aa = ROA_df.groupby('datadate')
    
    pd.qcut(ROA_df.groupby('datadate').values,5)
'''