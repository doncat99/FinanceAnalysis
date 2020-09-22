import warnings
warnings.filterwarnings("ignore")

import logging

import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from zvt import zvt_env
from zvt.contract.common import Region, Provider
from zvt.factors.squeeze_factor import SqueezeFactor
# from zvt.contract.reader import DataReader
# from zvt.domain import Stock1dKdata, Stock
import zvt.stats as qs

logger = logging.getLogger(__name__)


def chart(dfs):
    def sub_chart(title, df):
        candlestick = go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'])
        upper_band = go.Scatter(x=df['timestamp'], y=df['upper_band'], name='Upper Bollinger Band', line={'color': 'red'})
        lower_band = go.Scatter(x=df['timestamp'], y=df['lower_band'], name='Lower Bollinger Band', line={'color': 'red'})

        upper_keltner = go.Scatter(x=df['timestamp'], y=df['upper_keltner'], name='Upper Keltner Channel', line={'color': 'blue'})
        lower_keltner = go.Scatter(x=df['timestamp'], y=df['lower_keltner'], name='Lower Keltner Channel', line={'color': 'blue'})

        data=[candlestick, upper_band, lower_band, upper_keltner, lower_keltner]

        layout = go.Layout(
            title='Stock Market Data Analysis - ' + title,
            xaxis=dict(
                title='Date',
                titlefont=dict(
                    family='Courier New, monospace',
                    size=18,
                    color='#7f7f7f'),
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                            label="1m",
                            step="month",
                            stepmode="backward"),
                        dict(count=6,
                            label="6m",
                            step="month",
                            stepmode="backward"),
                        dict(count=1,
                            label="YTD",
                            step="year",
                            stepmode="todate"),
                        dict(count=1,
                            label="1y",
                            step="year",
                            stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(
                    visible=True
                ),
                type="date"
            ),
            yaxis=dict(
                title='Stock market price',
                titlefont=dict(
                    family='Courier New, monospace',
                    size=18,
                    color='#7f7f7f')
            )
        )

        return go.Figure(data=data, layout=layout)

    total = len(dfs)
    if total > 0:
        for key, df in dfs.items():
            fig = sub_chart(key, df)
            fig.show()   

if __name__ == '__main__':
    import time

    now = time.time()
    # reader = DataReader(region=Region.US, 
    #                     codes=['FB', 'AMD'], 
    #                     data_schema=Stock1dKdata, 
    #                     entity_schema=Stock,
    #                     provider=Provider.Yahoo,
    #                     entity_provider=Provider.Yahoo)

    # gb = reader.data_df.groupby('code')
    # dfs = {x : gb.get_group(x) for x in gb.groups}

    factor = SqueezeFactor(region=Region.US, 
                           codes=['FB', 'AMD'], 
                           start_timestamp='2015-01-01', 
                           end_timestamp='2020-07-01',
                           kdata_overlap=4,
                           provider=Provider.Yahoo,
                           entity_provider=Provider.Yahoo)

    gb = factor.result_df.groupby('code')
    dfs = {x : gb.get_group(x) for x in gb.groups}

    print("1", time.time()-now)
    target = pd.Series(dfs['FB'].close.pct_change().tolist(), index=dfs['FB'].timestamp)
    bench = pd.Series(dfs['AMD'].close.pct_change().tolist(), index=dfs['AMD'].timestamp)

    target_len = len(target)
    bench_len = len(bench)
    if bench_len > target_len:
        bench = bench[-target_len:]
    
    print("2", time.time()-now)
    qs.reports.html(returns=target, benchmark=bench, output='b.html')
    print("3", time.time()-now)
    
    chart(dfs)
    print("4", time.time()-now)

