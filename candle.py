import os
import talib
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from zvt import zvt_env
from zvt.contract.common import Region, Provider
from zvt.contract.reader import DataReader
from zvt.domain import Stock1dKdata, Stock

import faulthandler
faulthandler.enable()


candlestick_patterns = {
    'CDL2CROWS':'Two Crows',
    'CDL3BLACKCROWS':'Three Black Crows',
    'CDL3INSIDE':'Three Inside Up/Down',
    'CDL3LINESTRIKE':'Three-Line Strike',
    'CDL3OUTSIDE':'Three Outside Up/Down',
    'CDL3STARSINSOUTH':'Three Stars In The South',
    'CDL3WHITESOLDIERS':'Three Advancing White Soldiers',
    'CDLABANDONEDBABY':'Abandoned Baby',
    'CDLADVANCEBLOCK':'Advance Block',
    'CDLBELTHOLD':'Belt-hold',
    'CDLBREAKAWAY':'Breakaway',
    'CDLCLOSINGMARUBOZU':'Closing Marubozu',
    'CDLCONCEALBABYSWALL':'Concealing Baby Swallow',
    'CDLCOUNTERATTACK':'Counterattack',
    'CDLDARKCLOUDCOVER':'Dark Cloud Cover',
    'CDLDOJI':'Doji',
    'CDLDOJISTAR':'Doji Star',
    'CDLDRAGONFLYDOJI':'Dragonfly Doji',
    'CDLENGULFING':'Engulfing Pattern',
    'CDLEVENINGDOJISTAR':'Evening Doji Star',
    'CDLEVENINGSTAR':'Evening Star',
    'CDLGAPSIDESIDEWHITE':'Up/Down-gap side-by-side white lines',
    'CDLGRAVESTONEDOJI':'Gravestone Doji',
    'CDLHAMMER':'Hammer',
    'CDLHANGINGMAN':'Hanging Man',
    'CDLHARAMI':'Harami Pattern',
    'CDLHARAMICROSS':'Harami Cross Pattern',
    'CDLHIGHWAVE':'High-Wave Candle',
    'CDLHIKKAKE':'Hikkake Pattern',
    'CDLHIKKAKEMOD':'Modified Hikkake Pattern',
    'CDLHOMINGPIGEON':'Homing Pigeon',
    'CDLIDENTICAL3CROWS':'Identical Three Crows',
    'CDLINNECK':'In-Neck Pattern',
    'CDLINVERTEDHAMMER':'Inverted Hammer',
    'CDLKICKING':'Kicking',
    'CDLKICKINGBYLENGTH':'Kicking - bull/bear determined by the longer marubozu',
    'CDLLADDERBOTTOM':'Ladder Bottom',
    'CDLLONGLEGGEDDOJI':'Long Legged Doji',
    'CDLLONGLINE':'Long Line Candle',
    'CDLMARUBOZU':'Marubozu',
    'CDLMATCHINGLOW':'Matching Low',
    'CDLMATHOLD':'Mat Hold',
    'CDLMORNINGDOJISTAR':'Morning Doji Star',
    'CDLMORNINGSTAR':'Morning Star',
    'CDLONNECK':'On-Neck Pattern',
    'CDLPIERCING':'Piercing Pattern',
    'CDLRICKSHAWMAN':'Rickshaw Man',
    'CDLRISEFALL3METHODS':'Rising/Falling Three Methods',
    'CDLSEPARATINGLINES':'Separating Lines',
    'CDLSHOOTINGSTAR':'Shooting Star',
    'CDLSHORTLINE':'Short Line Candle',
    'CDLSPINNINGTOP':'Spinning Top',
    'CDLSTALLEDPATTERN':'Stalled Pattern',
    'CDLSTICKSANDWICH':'Stick Sandwich',
    'CDLTAKURI':'Takuri (Dragonfly Doji with very long lower shadow)',
    'CDLTASUKIGAP':'Tasuki Gap',
    'CDLTHRUSTING':'Thrusting Pattern',
    'CDLTRISTAR':'Tristar Pattern',
    'CDLUNIQUE3RIVER':'Unique 3 River',
    'CDLUPSIDEGAP2CROWS':'Upside Gap Two Crows',
    'CDLXSIDEGAP3METHODS':'Upside/Downside Gap Three Methods'
}

def get_cache():
    file = zvt_env['cache_path'] + '/' + 'candle.pkl'
    if os.path.exists(file) and os.path.getsize(file) > 0:
        with open(file, 'rb') as handle:
            return pickle.load(handle)
    return None

def dump(data):
    file = zvt_env['cache_path'] + '/' + 'candle.pkl'
    with open(file, 'wb+') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    pd.set_option('max_colwidth', 200)

    gb = get_cache()

    if not gb:
        reader = DataReader(region=Region.US, 
                            start_timestamp='2020-05-01',
                            data_schema=Stock1dKdata, 
                            entity_schema=Stock,
                            provider=Provider.Yahoo,
                            entity_provider=Provider.Yahoo)
        gb = reader.data_df.groupby('code')
        dump(gb)

    stocks = []

    for symbol in gb.groups:
        df = gb.get_group(symbol)

        patterns = []

        for pattern in candlestick_patterns:
            pattern_function = getattr(talib, pattern)

            results = pattern_function(df.open, df.high, df.low, df.close)
            
            last = results.tail(1).values[0]
            patterns.append(last)

        stocks.append(patterns)

    def normalization(data):
        _range = np.max(abs(data))
        return data / _range
    
    stocks = normalization(np.array(stocks))

    df = pd.DataFrame(data=stocks, columns=candlestick_patterns.keys(), index=gb.groups.keys())
    df['sum'] = df.sum(axis=1)
    df.sort_values(by=['sum'], ascending=False, inplace=True)
    
    f, ax = plt.subplots(figsize = (6,4))
    cmap = sns.cubehelix_palette(start = 0, rot = 3, gamma=0.8, as_cmap = True)   
    sns.heatmap(df, cmap = cmap, linewidths = 0, linecolor= 'white', ax = ax)   
    ax.set_title('Amounts per kind and region')
    ax.set_xlabel('pattern')
    ax.set_ylabel('stock')
    plt.show()


        

    
