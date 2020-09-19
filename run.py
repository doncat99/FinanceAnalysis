import warnings
warnings.filterwarnings("ignore")

import logging

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from zvt import zvt_env
from zvt.contract.common import Region, Provider
from zvt.factors.squeeze_factor import SqueezeFactor

logger = logging.getLogger(__name__)


def chart(dfs):
    def sub_chart(df):
        candlestick = go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'])
        upper_band = go.Scatter(x=df['timestamp'], y=df['upper_band'], name='Upper Bollinger Band', line={'color': 'red'})
        lower_band = go.Scatter(x=df['timestamp'], y=df['lower_band'], name='Lower Bollinger Band', line={'color': 'red'})

        upper_keltner = go.Scatter(x=df['timestamp'], y=df['upper_keltner'], name='Upper Keltner Channel', line={'color': 'blue'})
        lower_keltner = go.Scatter(x=df['timestamp'], y=df['lower_keltner'], name='Lower Keltner Channel', line={'color': 'blue'})

        fig = go.Figure(data=[candlestick, upper_band, lower_band, upper_keltner, lower_keltner])
        fig.layout.xaxis.type = 'category'
        fig.layout.xaxis.rangeslider.visible = False
        
        return fig

    total = len(dfs)
    if total > 0:
        for index, df in enumerate(dfs):
            fig = sub_chart(df)
            fig.show()   

if __name__ == '__main__':
    factor = SqueezeFactor(region=Region.US, entity_ids=['stock_NYSE_A', 'stock_NYSE_DDD'], 
                           start_timestamp='2015-01-01', end_timestamp='2020-07-01',
                           kdata_overlap=4)

    gb = factor.result_df.groupby('entity_id')
    dfs = [gb.get_group(x) for x in gb.groups]
    chart(dfs)

