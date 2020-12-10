import os
# import talib
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from zvt import zvt_env
from zvt.contract.common import Region, Provider
from zvt.factors.candlestick_factor import CandleStickFactor, candlestick_patterns

# import faulthandler
# faulthandler.enable()

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
        factor = CandleStickFactor(region=Region.US, 
                                   start_timestamp='2015-01-01', 
                                   kdata_overlap=0,
                                   provider=Provider.Yahoo,
                                   entity_provider=Provider.Yahoo)
        gb = factor.result_df.groupby('code')
        dump(gb)

    stocks = []

    for symbol in gb.groups:
        df = gb.get_group(symbol)

        patterns = []
        for pattern in candlestick_patterns:            
            last = df[pattern].tail(1).values[0]
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
