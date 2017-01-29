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
import matplotlib.pyplot as plt

if __name__ == '__main__':
    ### Set working directory
    os.getcwd()
    file_path = r'C:/Users/Chandler/Desktop/application/Interview/AXA/Project_AXA'
    os.chdir(file_path)
    ### Import the data and clean the format
    ROA_df = pd.read_csv('ROA.csv',sep='\s+')
    ROA_df['datadate'] = pd.to_datetime(ROA_df['datadate'],format="%d%b%Y")
    ROA_df = ROA_df.dropna(axis=0)
    ROA_df['notified_datadate'] = [x+dateutil.relativedelta.relativedelta(months=5) for x in ROA_df['datadate']]
    
    # datetime.datetime.strptime(ROA_df['datadate'],'%Y%m')
    # ROA_df['datadate_1'] = [x+dateutil.relativedelta.relativedelta(months=1)-dateutil.relativedelta.relativedelta(days=1) for x in ROA_df['datadate']]
    return_df = pd.read_csv('stockreturns.csv',sep='\s+',converters={'firm_id':np.int64,'ret':np.float64,'year':np.str,'month':np.str},engine='python')
    return_df = return_df[return_df['month']!='None'] # Drop None data
    return_df['datadate'] = pd.to_datetime((return_df['year'] + return_df['month']), format='%Y%m')
    # return_df['datadate_1'] = return_df['datadate']+dateutil.relativedelta.relativedelta(months=1)-dateutil.relativedelta.relativedelta(days=1)
    return_df['notified_datadate'] = [x+dateutil.relativedelta.relativedelta(months=1)-dateutil.relativedelta.relativedelta(days=1) for x in return_df['datadate']]
            
    ### Merge two dataframe
    merged_df = pd.merge(return_df,ROA_df,how='left',on=['notified_datadate','firm_id'])
    # drop insufficient data
    valid_firm_list = list(merged_df.groupby('firm_id')['ROA'].mean().dropna().index) # keep the firm that has valid ROA (at least one ROA)
    merged_df = merged_df[merged_df['firm_id'].isin(valid_firm_list)]
    merged_df = merged_df.reset_index(drop=True) # Reset the index
    ### Fill other NaN with lastest data
    merged_df = merged_df.sort_values(['firm_id','notified_datadate'])
    firm_id_init = 10016
    d=0; r=0; s=0;
    for i in range(len(merged_df.index)):
        if merged_df.loc[i,'firm_id'] != firm_id_init:
            firm_id_init = merged_df.loc[i,'firm_id']
            # print(firm_id_init)
            # print(i)
            d=0; r=0; s=0;
            merged_df.loc[i,'datadate_y']=d
            merged_df.loc[i,'ROA']=r
            merged_df.loc[i,'sic2']=s
        else:
            if merged_df.loc[i,:].isnull().any():
                merged_df.loc[i,'datadate_y']=d
                merged_df.loc[i,'ROA']=r
                merged_df.loc[i,'sic2']=s
            else:
                d = merged_df.loc[i,'datadate_y']
                r = merged_df.loc[i,'ROA']
                s = merged_df.loc[i,'sic2'] 
    # merged_df.to_csv('merged_df.csv') # Save to csv for faster read in next time
    merged_df =pd.read_csv('merged_df.csv')
        
    ### Clean the inrelevant date and data
    num_group = merged_df.groupby('notified_datadate').count()
    invalid_date_list = num_group.index[0:11] # Find the datatime variable that doesn't have enough observation
    merged_df = merged_df.dropna() # Drop the row that doesn't have valid ROA
    merged_df = merged_df.drop(merged_df[merged_df['sic2']==0].index) # Drop the row that doesn't have valid ROA
    merged_df = merged_df[~merged_df['notified_datadate'].isin(invalid_date_list)] # drop the datetime variable that has less than 1000 observation
    
    ### Calculate the top 20% and bottom 20%
    merged_df = merged_df.reset_index(drop=True) # Reset the index
    merged_df['percentile']=merged_df.groupby('notified_datadate')['ROA'].rank(pct=True) # Calculate the ROA percentile across the same notified date
    portfolio_return_top = merged_df.groupby('notified_datadate').apply(lambda x: x[x['percentile']>=0.8].mean()) # calcualte top 20% portfolio return 
    portfolio_return_bottom = merged_df.groupby('notified_datadate').apply(lambda x: x[x['percentile']<=0.2].mean()) # calcualte the bottom 20% portfolio return

    ### Calculate the combined portfolio
    portfolio_return = pd.concat([portfolio_return_top[['ret','ROA','percentile']],portfolio_return_bottom[['ret','ROA','percentile']]],axis=1)
    portfolio_return.columns=['ret_top','ROA_top','percentile_top','ret_bottom','ROA_bottom','percentile_bottom'] # Change the column name
    portfolio_return['combined_ret'] = portfolio_return['ret_top']-portfolio_return['ret_bottom']

    ### Calculate the cumulative return
    portfolio_return['cum_return'] = (portfolio_return['combined_ret']+1).cumprod()-1

    ### Plot the Cummulative Return
    plt.style.use('fivethirtyeight')



    merged_df.loc[1,'datadate_y']






    # Another method
    date_list = num_group.index
    
    for i in date_list:
        monthly_table = ROA_df[ROA_df['datadate']==i]
        monthly_table.loc[:,'decile'] = pd.Series(np.array(pd.qcut(monthly_table['ROA'].values,5,labels=['0.2','0.4','0.6','0.8','1'])),index=monthly_table.index)
        # monthly_table = monthly_table.assign(e=pd.Series(pd.qcut(monthly_table['ROA'].values,5,labels=['0.2','0.4','0.6','0.8','1'])))

    # one method

    top_ROA = ROA_df.groupby('datadate')['ROA'].quantile(0.8)
    bottom_ROA = ROA_df.groupby('datadate')['ROA'].quantile(0.2)

    date_list = top_ROA.index
    
    for i in date_list:
        ROA_df[ROA_df['datadate']==i]

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