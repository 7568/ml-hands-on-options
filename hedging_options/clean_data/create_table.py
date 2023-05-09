import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from openpyxl import Workbook


def create_table_001(ta):
    up_rete = 0  # 1.0 0, 0.1 , 0.05, 0.01
    PREPARE_HOME_PATH = f'/home/liyu/data/hedging-option/20190701-20221124_{up_rete}/h_sh_300'
    df = pd.read_csv(f'{PREPARE_HOME_PATH}/all_raw_data.csv', parse_dates=['TradingDate'])
    df.loc[:, 'RemainingTerm'] = np.round((df['RemainingTerm'] * 365).to_numpy())
    print(df.shape)
    df_c = df[df['CallOrPut'] ==0]
    df_p = df[df['CallOrPut'] ==1]


    print(df.shape[0])
    print('call info ',df_c.shape[0],"{:.4g}".format(df_c.shape[0]/df.shape[0]))
    print('put info ',df_p.shape[0],"{:.4g}".format(df_p.shape[0]/df.shape[0]))

def create_table_002(ta):
    up_rete = 0  # 1.0 0, 0.1 , 0.05, 0.01
    PREPARE_HOME_PATH = f'/home/liyu/data/hedging-option/20190701-20221124_{up_rete}/h_sh_300'
    df = pd.read_csv(f'{PREPARE_HOME_PATH}/all_raw_data.csv', parse_dates=['TradingDate'])
    df.loc[:, 'RemainingTerm'] = np.round((df['RemainingTerm'] * 365).to_numpy())
    print(df.shape)
    df_year_2020 = df[df['TradingDate'] < pd.Timestamp('2021-01-01')]
    mask_2021 = (df['TradingDate'] < pd.Timestamp('2022-01-01')) & (df['TradingDate'] > pd.Timestamp('2021-01-01'))
    df_year_2021 = df[mask_2021]
    mask_2022 = (df['TradingDate'] < pd.Timestamp('2022-12-24')) & (df['TradingDate'] > pd.Timestamp('2022-01-01'))
    df_year_2022 = df[mask_2022]

    print(df.shape[0])
    print(df_year_2020.shape[0],df_year_2020.shape[0]/df.shape[0])
    print(df_year_2021.shape[0],df_year_2021.shape[0]/df.shape[0])
    print( df_year_2022.shape[0],df_year_2022.shape[0]/df.shape[0])





def caculate_df_year(df):
    # moneyness = spot/strike
    spot = df['UnderlyingScrtClose'].to_numpy()
    strike = df['StrikePrice'].to_numpy()
    df_year = spot/strike
    df.loc[:, 'moneyness'] = df_year
    return df


def save_data_statistics(ta,tb,tc):
    wb = Workbook()
    ws = wb.active
    ws.append(['Year', 'number', 'r'])
    date = ta
    for index_0, m in enumerate(tb):
        ws.append([date[index_0], m.shape[0], ''])
    wb.save(f'{tc}.xlsx')


def create_table_003(ta):
    up_rete = 0  # 1.0 0, 0.1 , 0.05, 0.01
    PREPARE_HOME_PATH = f'/home/liyu/data/hedging-option/20190701-20221124_{up_rete}/h_sh_300'
    df = pd.read_csv(f'{PREPARE_HOME_PATH}/all_raw_data.csv', parse_dates=['TradingDate'])
    df.loc[:, 'RemainingTerm'] = np.round((df['RemainingTerm'] * 365).to_numpy())
    print(df.shape)
    df_year_2020 = df[df['TradingDate'] < pd.Timestamp('2021-01-01')]
    mask_2021 = (df['TradingDate'] < pd.Timestamp('2022-01-01')) & (df['TradingDate'] > pd.Timestamp('2021-01-01'))
    df_year_2021 = df[mask_2021]
    mask_2022 = (df['TradingDate'] < pd.Timestamp('2022-12-24')) & (df['TradingDate'] > pd.Timestamp('2022-01-01'))
    df_year_2022 = df[mask_2022]
    for i,df_year_i in enumerate([df_year_2020,df_year_2021,df_year_2022]):
        spot = df_year_i['UnderlyingScrtClose'].to_numpy()
        strike = df_year_i['StrikePrice'].to_numpy()
        moneyness = spot / strike
        print(f'{i} ')
        print((moneyness<0.97).sum(),'\t',"{:.4g}".format(((moneyness<0.97).sum())/(moneyness.size)))
        print(((0.97<moneyness) & (moneyness<1.03)).sum(),'\t',"{:.4g}".format(((0.97<moneyness) & (moneyness<1.03)).sum()/moneyness.size))
        print((moneyness>1.03).sum(),'\t',"{:.4g}".format(((moneyness>1.03).sum())/moneyness.size))

def create_table_004(ta):
    up_rete = 0  # 1.0 0, 0.1 , 0.05, 0.01
    PREPARE_HOME_PATH = f'/home/liyu/data/hedging-option/20190701-20221124_{up_rete}/h_sh_300'
    training = pd.read_csv(f'{PREPARE_HOME_PATH}/training.csv', parse_dates=['TradingDate'])
    validation = pd.read_csv(f'{PREPARE_HOME_PATH}/validation.csv', parse_dates=['TradingDate'])
    testing = pd.read_csv(f'{PREPARE_HOME_PATH}/testing.csv', parse_dates=['TradingDate'])

    for i,df_year_i in enumerate([training,validation,testing]):
        spot = df_year_i['UnderlyingScrtClose'].to_numpy()
        strike = df_year_i['StrikePrice'].to_numpy()
        moneyness = spot / strike
        print(f'{i} ')
        print((moneyness<0.97).sum(),'\t',"{:.4g}".format(((moneyness<0.97).sum())/(moneyness.size)))
        print(((0.97<moneyness) & (moneyness<1.03)).sum(),'\t',"{:.4g}".format(((0.97<moneyness) & (moneyness<1.03)).sum()/moneyness.size))
        print((moneyness>1.03).sum(),'\t',"{:.4g}".format(((moneyness>1.03).sum())/moneyness.size))

def analysis_by_moneyness_maturity(result_df,max_day=210,tc='table'):

    year_info = result_df
    moneyness_less_097 = year_info[year_info['moneyness'] <= 0.97]
    print(f'moneyness_less_097.shape : {moneyness_less_097.shape}')
    moneyness_in_097_103 = year_info.loc[(year_info['moneyness'] > 0.97) & (year_info['moneyness'] <= 1.03)]
    print(f'moneyness_in_097_103.shape : {moneyness_in_097_103.shape}')
    moneyness_more_103 = year_info[year_info['moneyness'] > 1.03]
    print(f'moneyness_more_103.shape : {moneyness_more_103.shape}')
    moneyness_infos = {'<0.97':moneyness_less_097,'0.97 ~ 1.03':moneyness_in_097_103,'≥1.03':moneyness_more_103,}
    for moneyness_str in moneyness_infos:
        moneyness_info = moneyness_infos[moneyness_str]
        span_days = 30
        for d in range(0,210,span_days):
            maturity = moneyness_info.loc[(moneyness_info['RemainingTerm'] >= d) & (moneyness_info['RemainingTerm'] < (d+span_days))]
            num=len(maturity)

        if max_day>270:
            d = 210
            maturity = moneyness_info.loc[
                (moneyness_info['RemainingTerm'] >= d) & (moneyness_info['RemainingTerm'] < (max_day))]
            num=len(maturity)



def create_chart_001(param,max_day=360):
    up_rete = 0  # 1.0 0, 0.1 , 0.05, 0.01
    PREPARE_HOME_PATH = f'/home/liyu/data/hedging-option/20190701-20221124_{up_rete}/h_sh_300'
    training = pd.read_csv(f'{PREPARE_HOME_PATH}/training.csv', parse_dates=['TradingDate'])
    validation = pd.read_csv(f'{PREPARE_HOME_PATH}/validation.csv', parse_dates=['TradingDate'])
    testing = pd.read_csv(f'{PREPARE_HOME_PATH}/testing.csv', parse_dates=['TradingDate'])
    training.loc[:, 'RemainingTerm'] = np.round((training['RemainingTerm'] * 365).to_numpy())
    validation.loc[:, 'RemainingTerm'] = np.round((validation['RemainingTerm'] * 365).to_numpy())
    testing.loc[:, 'RemainingTerm'] = np.round((testing['RemainingTerm'] * 365).to_numpy())
    title=['training dataset','validation dataset','testing dataset']
    for index, df in enumerate([training,validation,testing]):
        moneyness_info = df
        span_days = 30
        x_label=[]
        y_value=[]
        for d in range(0, 210, span_days):
            x_label.append(f'{d}~{d+span_days}')
            maturity = moneyness_info.loc[
                (moneyness_info['RemainingTerm'] >= d) & (moneyness_info['RemainingTerm'] < (d + span_days))]
            y_value.append(len(maturity))

        if max_day > 270:
            d = 210
            x_label.append(f'{d}~{max_day}')
            maturity = moneyness_info.loc[
                (moneyness_info['RemainingTerm'] >= d) & (moneyness_info['RemainingTerm'] < (max_day))]
            y_value.append(len(maturity))
        create_bar_chart(x_label,y_value,title[index])


def create_bar_chart(x,y,title):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import host_subplot
    font_size = 56
    plt.figure(figsize=(28, 28))
    ax = host_subplot(211)
    ax.set_xlabel("Day", size=font_size)
    ax.set_ylabel("numbers", size=font_size)
    ax.set_title(title, size=font_size)

    ax.bar(x, y, color='#E50000')
    ax.tick_params(labelcolor='#000000', labelsize=font_size)

    plt.xticks(fontsize=font_size,rotation=45)
    # 显示图形
    plt.show()

def create_chart_002(param):
    pass

HOME_PATH = f'/home/liyu/data/hedging-option/20170101-20230101/'
if __name__ == '__main__':
    OPTION_SYMBOL = 'index-option/h_sh_300_option'
    DATA_HOME_PATH = HOME_PATH + "/" + OPTION_SYMBOL + "/"
    # create_table_001('h_sh_300_sorted_by_years')
    # create_table_002('h_sh_300_sorted_by_years')
    # create_table_003('h_sh_300_sorted_by_moneyness')
    # create_table_004('h_sh_300_sorted_by_moneyness')
    create_chart_001('h_sh_300_sorted_by_moneyness')
    # create_chart_002('h_sh_300_sorted_by_moneyness')


