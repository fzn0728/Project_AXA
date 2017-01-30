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

def fill_na(merged_df):
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
    return merged_df

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

    return_df = pd.read_csv('stockreturns.csv',sep='\s+',converters={'firm_id':np.int64,'ret':np.float64,'year':np.str,'month':np.str},engine='python')
    return_df = return_df[return_df['month']!='None'] # Drop None data
    return_df['datadate'] = pd.to_datetime((return_df['year'] + return_df['month']), format='%Y%m')
    return_df['notified_datadate'] = [x+dateutil.relativedelta.relativedelta(months=1)-dateutil.relativedelta.relativedelta(days=1) for x in return_df['datadate']]
            
    ### Merge two dataframes
    merged_df = pd.merge(return_df,ROA_df,how='left',on=['notified_datadate','firm_id'])
    valid_firm_list = list(merged_df.groupby('firm_id')['ROA'].mean().dropna().index) # keep the firm that has valid ROA (at least one ROA)
    merged_df = merged_df[merged_df['firm_id'].isin(valid_firm_list)] # drop insufficient data
    merged_df = merged_df.reset_index(drop=True) # Reset the index
    
    ### Fill other NaN with lastest data
    # merged_df = fill_na(merged_df)
    merged_df =pd.read_csv('merged_df.csv')
        
    ### Clean the inrelevant date and data
    num_group = merged_df.groupby('notified_datadate').count()
    invalid_date_list = num_group.index[0:11] # Find the datatime variable that doesn't have enough observations
    merged_df = merged_df.dropna() # Drop the row that doesn't have valid ROA
    merged_df = merged_df.drop(merged_df[merged_df['sic2']==0].index) # Drop the row that doesn't have valid ROA
    merged_df = merged_df[~merged_df['notified_datadate'].isin(invalid_date_list)] # drop the datetime variable that has less than 1000 observations
    
    ### Calculate the top 20% and bottom 20%
    merged_df = merged_df.reset_index(drop=True) # Reset the index
    merged_df['percentile']=merged_df.groupby('notified_datadate')['ROA'].rank(pct=True) # Calculate the ROA percentile across the same notified date
    portfolio_return_top = merged_df.groupby('notified_datadate').apply(lambda x: x[x['percentile']>=0.8].mean()) # calcualte top 20% portfolio return 
    portfolio_return_bottom = merged_df.groupby('notified_datadate').apply(lambda x: x[x['percentile']<=0.2].mean()) # calcualte the bottom 20% portfolio return

    ### ROA mean across industries
    sic_df = pd.DataFrame(pd.read_table('sic2.txt')['Industry'].values)
    ROA_ind = merged_df.groupby('sic2')['ROA'].mean()
    sic_df.index = ROA_ind.index
    ROA_ind = pd.concat([ROA_ind,sic_df],axis=1)
    ROA_ind_plot = ROA_ind.set_index([0])
    ROA_ind_plot.plot(kind='bar')
    plt.ylabel('Average ROA')
    plt.title('Average ROA Across Different Industries')      


    ### Calculate the combined portfolio and get the cumulative return
    portfolio_return = pd.concat([portfolio_return_top[['ret','ROA','percentile']],portfolio_return_bottom[['ret','ROA','percentile']]],axis=1)
    portfolio_return.columns=['ret_top','ROA_top','percentile_top','ret_bottom','ROA_bottom','percentile_bottom'] # Change the column name
    portfolio_return['combined_ret'] = portfolio_return['ret_top']-portfolio_return['ret_bottom']
    portfolio_return['cum_return'] = (portfolio_return['combined_ret']+1).cumprod()-1

    ### Get the Benchmark Data
    SP500_df = pd.read_csv('SP500.csv',index_col=0)
    SP500_df.index = pd.to_datetime(SP500_df.index,format="%Y%m%d")
    SP500_df['cum_sp_equal_ret'] = (SP500_df['ewretx']+1).cumprod()-1
    portfolio_return['ewretx'] = SP500_df['ewretx'].values
    portfolio_return['cum_sp_equal_ret'] = SP500_df['cum_sp_equal_ret'].values

    ### Plot the Cummulative Return
    plt.style.use('fivethirtyeight')
    portfolio_return[['cum_return','cum_sp_equal_ret']].plot()
    L1 = plt.legend(prop={'size':12})
    L1.get_texts()[0].set_text('Cumulative Return of Portfolio')
    L1.get_texts()[1].set_text('Cumulative Return of S&P500')
    plt.xlabel('Year')
    plt.ylabel('Cumulative Return')
    plt.title('Cumulative Return of Portfolio and S&P500')  
    
    ### Calculate the Information Ratio
    IR = information_ratio(portfolio_return['combined_ret'].values,portfolio_return['ewretx'].values)
    
    ### Rolling vol
    rolling_vol = pd.rolling_std(portfolio_return[['combined_ret','ewretx']],window=12,min_periods=12)
    rolling_vol.plot()
    L2 = plt.legend(prop={'size':12})
    L2.get_texts()[0].set_text('12 Months Rolling Volatility of Portfolio')
    L2.get_texts()[1].set_text('12 Months Rolling Volatility of S&P500')
    plt.xlabel('Year')
    plt.ylabel('Volatility')
    plt.title('12 Months Rolling Volatility of Portfolio Returns and SP500')   
    
    ### Vol Leverage Ratio
    rolling_vol['Hedge Ratio'] = rolling_vol['ewretx']/rolling_vol['combined_ret']
    rolling_vol['Hedge Ratio'].plot()
    plt.legend(prop={'size':12})
    plt.xlabel('Year')
    plt.ylabel('Volatility Leverage')
    plt.title('Volatility Leverage to S&P500 Index')       
    
    ### If industrial neutral
    merged_df['percentile_ind']=merged_df.groupby(['notified_datadate','sic2'])['ROA'].rank(pct=True) # Calculate the ROA percentile across the same notified date and same industry
    portfolio_ind_return_top = merged_df.groupby('notified_datadate').apply(lambda x: x[x['percentile_ind']>=0.8].mean()) # calcualte top 20% portfolio return 
    portfolio_ind_return_bottom = merged_df.groupby('notified_datadate').apply(lambda x: x[x['percentile_ind']<=0.2].mean()) # calcualte the bottom 20% portfolio return    
    portfolio_ind_return = pd.concat([portfolio_ind_return_top[['ret','ROA','percentile_ind']],portfolio_ind_return_bottom[['ret','ROA','percentile_ind']]],axis=1)
    portfolio_ind_return.columns=['ret_top','ROA_top','percentile_top','ret_bottom','ROA_bottom','percentile_bottom'] # Change the column name
    portfolio_ind_return['combined_ret'] = portfolio_ind_return['ret_top']-portfolio_ind_return['ret_bottom']    
    portfolio_ind_return['cum_return'] = (portfolio_ind_return['combined_ret']+1).cumprod()-1
    portfolio_ind_return['ewretx'] = SP500_df['ewretx'].values
    portfolio_ind_return['cum_sp_equal_ret'] = SP500_df['cum_sp_equal_ret'].values
    
    ### Plot Cumulative Return Comparison
    compare_df = pd.concat([portfolio_return[['cum_return']],portfolio_ind_return[['cum_return']]],axis=1)
    compare_df.columns = ['Before Industrial Neutral Portfolio','After Industrial Neutral Portfolio']
    compare_df.plot()
    plt.legend(prop={'size':12})
    plt.xlabel('Year')
    plt.ylabel('Cumulative Return')
    plt.title('Cumulative Return of Portfolio before and after Industrial Neutral')
    
    ### Calculate the Information Ratio
    IR_ind = information_ratio(portfolio_ind_return['combined_ret'].values,portfolio_ind_return['ewretx'].values)    
    
    ### Calculate Beta
    SPY_df = pd.read_csv('SPY.csv',index_col=0)
    portfolio_ind_return.ix[20:,'SPY_ret'] = SPY_df.ix[1:-1,'Return'].values
    cov_matrix = portfolio_ind_return.ix[20:,['combined_ret','SPY_ret']].rolling(window=24,min_periods=24).cov(portfolio_ind_return.ix[20:,['combined_ret','SPY_ret']],pairwise=True)
    
    beta_df = pd.DataFrame(index = portfolio_ind_return.index[20:], columns = ['Beta'])
    for i in range(0,len(beta_df.index)-23):
        beta_df.ix[i+23,'Beta'] = cov_matrix.iloc[i+23].iloc[0][1]/cov_matrix.iloc[i+23].iloc[1][1]
    
    beta_df.plot()
    plt.legend(prop={'size':12})
    plt.xlabel('Year')
    plt.ylabel('Beta Value')
    plt.title('24 Months Rolling Beta with SPY')    
    
    ### Make Beta Neutral
    beta_df['Adj_change'] = beta_df.Beta*(-1)*SPY_df.ix[1:-1,'Return'].values
    portfolio_ind_return['Adj_ret'] = portfolio_ind_return.ix[20:,'combined_ret'] +beta_df['Adj_change'] 
    IR_adj = information_ratio(portfolio_ind_return.ix[43:,'Adj_ret'].values,portfolio_ind_return.ix[43:,'ewretx'].values)  
    
    ### Plot ETF position
    ((-1)*beta_df.Beta).plot()
    plt.ylabel('Negative Beta Value')
    plt.title('Position in ETF')    
    
    ### Consider without industrial Neutral -- Beta
    portfolio_return.ix[20:,'SPY_ret'] = SPY_df.ix[1:-1,'Return'].values
    cov_matrix = portfolio_return.ix[20:,['combined_ret','SPY_ret']].rolling(window=24,min_periods=24).cov(portfolio_return.ix[20:,['combined_ret','SPY_ret']],pairwise=True)
    beta_df_w = pd.DataFrame(index = portfolio_return.index[20:], columns = ['Beta'])
    for i in range(0,len(beta_df_w.index)-23):
        beta_df_w.ix[i+23,'Beta'] = cov_matrix.iloc[i+23].iloc[0][1]/cov_matrix.iloc[i+23].iloc[1][1]    
    
    beta_df_w['Adj_change'] = beta_df_w.Beta*(-1)*SPY_df.ix[1:-1,'Return'].values
    portfolio_return['Adj_ret'] = portfolio_return.ix[20:,'combined_ret'] +beta_df_w['Adj_change'] 
    IR_adj_w = information_ratio(portfolio_return.ix[43:,'Adj_ret'].values,portfolio_return.ix[43:,'ewretx'].values)     
    
