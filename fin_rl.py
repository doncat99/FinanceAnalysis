import warnings
warnings.filterwarnings("ignore")
import faulthandler
faulthandler.enable()

import logging
from os import listdir
import time
from datetime import datetime

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from zvt.contract.common import Region, Provider
from zvt.factors.algorithm import tech_indicator
from zvt.factors.candlestick_factor import CandleStickFactor, candlestick_patterns
from zvt.trader.backtest import BackTestStats, BaselineStats, BackTestPlot

from zvt.models.stock_env.environment import EnvSetup
from zvt.models.stock_env.EnvMultipleStock_train import StockEnvTrain
from zvt.models.stock_env.EnvMultipleStock_trade import StockEnvTrade
from zvt.models.drl_agent_models import DRLAgent



logger = logging.getLogger(__name__)


def data_split(df, start, end):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df.timestamp >= start) & (df.timestamp < end)]
    data = data.sort_values(['timestamp', 'entity_id'], ignore_index=True)
    data.index = data.timestamp.factorize()[0]
    return data


if __name__ == '__main__':
    now = time.time()
    pd.options.display.max_columns = 15
    pd.options.display.width = 10

    factor = CandleStickFactor(region=Region.US, 
                               codes=['FB', 'AMD'], 
                               start_timestamp='2015-01-01', 
                               kdata_overlap=0,
                               provider=Provider.Yahoo,
                               entity_provider=Provider.Yahoo)

    train = data_split(factor.result_df, '2015-01-01', '2019-01-01')
    trade = data_split(factor.result_df, '2019-01-01', '2020-01-01')
    
    print(train.size, train.shape, train.ndim)
    print(train)
    print(trade.size, trade.shape, trade.ndim)
    print(trade)

    tech_list = tech_indicator + list(candlestick_patterns.keys())

    stock_dimension = len(train.entity_id.unique())
    state_space = 1 + 2*stock_dimension + (len(tech_list))*stock_dimension

    print(stock_dimension, state_space)

    env_setup = EnvSetup(stock_dim = stock_dimension,
                         state_space = state_space,
                         hmax = 100,
                         initial_amount = 1000000,
                         transaction_cost_pct = 0.001,
                         tech_indicator_list = tech_list)

    env_train = env_setup.create_env_training(data = train, env_class = StockEnvTrain)
    env_trade, obs_trade = env_setup.create_env_trading(data = trade, env_class = StockEnvTrade)

    agent = DRLAgent(env = env_train)

    print("==============Model Training===========")
    now = datetime.now().strftime('%Y%m%d-%Hh%M')
    a2c_params_tuning = {
                'n_steps':512, 
                'ent_coef':0.005, 
                'learning_rate':0.0002,
                'verbose':0,
                'timesteps':150000}
    model_a2c = agent.train_A2C(model_name = "A2C_{}".format(now), model_params = a2c_params_tuning)


    # print("==============Model Training===========")
    # now = datetime.now().strftime('%Y%m%d-%Hh%M')
    # ddpg_params_tuning = {
    #             'batch_size':512,
    #             'buffer_size':100000, 
    #             'verbose':0,
    #             'timesteps':50000}
    # model_ddpg = agent.train_DDPG(model_name = "DDPG_{}".format(now), model_params = ddpg_params_tuning)


    print("==============Model Training===========")
    now = datetime.now().strftime('%Y%m%d-%Hh%M')
    ppo_params_tuning = {
                'n_steps':512, 
                'nminibatches':4,
                'ent_coef':0.005, 
                'learning_rate':0.00025,
                'verbose':0,
                'timesteps':50000}
    model_ppo = agent.train_PPO(model_name = "PPO_{}".format(now), model_params = ppo_params_tuning)


    print("==============Model Training===========")
    now = datetime.now().strftime('%Y%m%d-%Hh%M')
    td3_params_tuning = {
                'batch_size': 512,
                'buffer_size':200000, 
                'learning_rate': 0.0002,
                'verbose':0,
                'timesteps':50000}
    model_td3 = agent.train_TD3(model_name = "TD3_{}".format(now), model_params = td3_params_tuning)


    # env_train = env_setup.create_env_training(data = train, env_class = StockEnvTrain)
    # agent = DRLAgent(env = env_train)
    # print("==============Model Training===========")
    # now = datetime.now().strftime('%Y%m%d-%Hh%M')
    # sac_params_tuning={
    #             'batch_size': 512,
    #             'buffer_size': 100000,
    #             'ent_coef':'auto_0.1',
    #             'learning_rate': 0.0001,
    #             'learning_starts':200,
    #             'timesteps': 50000,
    #             'verbose': 0}
    # model_sac = agent.train_SAC(model_name = "SAC_{}".format(now), model_params = sac_params_tuning)

    df = factor.result_df
    data_turbulence = df[(df.timestamp<'2019-01-01') & (df.timestamp>='2009-01-01')]
    insample_turbulence = data_turbulence.drop_duplicates(subset=['timestamp'])

    insample_turbulence.turbulence.describe()
    turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 1)

    env_trade, obs_trade = env_setup.create_env_trading(data = trade, env_class = StockEnvTrade,
                                                        turbulence_threshold=turbulence_threshold) 

    df_account_value, df_actions = DRLAgent.DRL_prediction(model=model_sac, test_data = trade,
                                                           test_env = env_trade, test_obs = obs_trade)

    print("==============Get Backtest Results===========")
    perf_stats_all = BackTestStats(account_value=df_account_value)
    perf_stats_all = pd.DataFrame(perf_stats_all)
    perf_stats_all.to_csv("./"+"results"+"/perf_stats_all_"+now+'.csv')

    print("==============Compare to DJIA===========")
    # S&P 500: ^GSPC
    # Dow Jones Index: ^DJI
    # NASDAQ 100: ^NDX
    BackTestPlot(df_account_value, 
                baseline_ticker = '^DJI', 
                baseline_start = '2019-01-01',
                baseline_end = '2020-09-30')

    print("==============Get Baseline Stats===========")
    baesline_perf_stats=BaselineStats('^DJI',
                                    baseline_start = '2019-01-01',
                                    baseline_end = '2020-09-30')
