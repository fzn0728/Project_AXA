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
import pandas_datareader.data as wb
import matplotlib
%matplotlib qt


def vol(returns):
    # Return the standard deviation of returns
    return np.std(returns)
    
def information_ratio(returns, benchmark):
    diff = returns - benchmark
    return np.mean(diff) / vol(diff)    
    
    
    

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
    a = portfolio_return['cum_return'].plot()
    plt.legend(prop={'size':12})
    plt.xlabel('Year')
    plt.ylabel('Cumulative Return')
    plt.title('Cumulative Return of All Market Portfolio')    
    
    
    ### Get the Benchmark Data
    SP500_df = pd.read_csv('SP500.csv',index_col=0)
    SP500_df.index = pd.to_datetime(SP500_df.index,format="%Y%m%d")
    SP500_df['cum_sp_equal_ret'] = (SP500_df['ewretx']+1).cumprod()-1
    portfolio_return['cum_sp_equal_ret'] = SP500_df['cum_sp_equal_ret'].values
    ### Calculate the Information Ratio
    IC = information_ratio(portfolio_return['cum_return'].values,portfolio_return['cum_sp_equal_ret'].values)
    
    ### Rolling vol
    rolling_vol = pd.rolling_std(portfolio_return[['cum_return','cum_sp_equal_ret']],window=12,min_periods=12)
    rolling_vol.plot()
    plt.legend(prop={'size':12})
    plt.xlabel('Year')
    plt.ylabel('Volatility')
    plt.title('12 Months Rolling Volatility of Portfolio Returns and SP500')   
    
    ### If industrial neutral
    merged_df['percentile_ind']=merged_df.groupby(['notified_datadate','sic2'])['ROA'].rank(pct=True) # Calculate the ROA percentile across the same notified date
    portfolio_ind_return_top = merged_df.groupby('notified_datadate').apply(lambda x: x[x['percentile_ind']>=0.8].mean()) # calcualte top 20% portfolio return 
    portfolio_ind_return_bottom = merged_df.groupby('notified_datadate').apply(lambda x: x[x['percentile_ind']<=0.2].mean()) # calcualte the bottom 20% portfolio return    
    
    portfolio_ind_return = pd.concat([portfolio_ind_return_top[['ret','ROA','percentile_ind']],portfolio_ind_return_bottom[['ret','ROA','percentile_ind']]],axis=1)
    portfolio_ind_return.columns=['ret_top','ROA_top','percentile_top','ret_bottom','ROA_bottom','percentile_bottom'] # Change the column name
    portfolio_ind_return['combined_ret'] = portfolio_ind_return['ret_top']-portfolio_ind_return['ret_bottom']    
    portfolio_ind_return['cum_return'] = (portfolio_ind_return['combined_ret']+1).cumprod()-1
    portfolio_ind_return['cum_sp_equal_ret'] = SP500_df['cum_sp_equal_ret'].values
    
    ### Plot Cumulative Return Comparison
    compare_df = pd.concat([portfolio_return[['cum_return']],portfolio_ind_return[['cum_return']]],axis=1)
    compare_df.columns = ['All Market Portfolio','Industrial Neutral Portfolio']
    compare_df.plot()
    plt.legend(prop={'size':12})
    plt.xlabel('Year')
    plt.ylabel('Cumulative Return')
    plt.title('Cumulative Return of All Market Portfolio and Industrial Neutral Portfolio')
    
    ### Calculate the Information Ratio
    IC_ind = information_ratio(portfolio_ind_return['cum_return'].values,portfolio_ind_return['cum_sp_equal_ret'].values)    
    
    ### Calculate Beta
    SPY_df = pd.read_csv('SPY.csv',index_col=0)
    portfolio_ind_return.ix[19:,'SPY_ret'] = SPY_df.ix[1:,'Return'].values
    cov_matrix = portfolio_ind_return.ix[19:,['combined_ret','SPY_ret']].rolling(window=24,min_periods=24).cov(portfolio_ind_return.ix[19:,['combined_ret','SPY_ret']],pairwise=True)
    
    beta_df = pd.DataFrame(index = portfolio_ind_return.index[19:], columns = ['Beta'])
    for i in range(0,len(beta_df.index)-23):
        beta_df.ix[i+23,'Beta'] = cov_matrix.iloc[i+23].iloc[0][1]/cov_matrix.iloc[i+23].iloc[1][1]
    
    beta_df.plot()
    plt.legend(prop={'size':12})
    plt.xlabel('Year')
    plt.ylabel('Beta Value')
    plt.title('24 Months Rolling Beta with SPY')    
    
    
    
    
    df.plot(portfolio_return['cum_return'].index,portfolio_return[['cum_return']])



    portfolio_return['cum_return'].plot()


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