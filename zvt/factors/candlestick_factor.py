# -*- coding: utf-8 -*-
from typing import List, Union

import pandas as pd
import talib

from zvt.contract import IntervalLevel, EntityMixin
from zvt.contract.common import Region, Provider
from zvt.domain import Stock
from zvt.factors import Accumulator
from zvt.factors.factor import Transformer
from zvt.factors.technical_factor import TechnicalFactor


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
    'CDLXSIDEGAP3METHODS':'Upside/Downside Gap Three Methods',
}

class CandleStickTransformer(Transformer):
    def __init__(self, kdata_overlap=0) -> None:
        super().__init__()
        self.kdata_overlap = kdata_overlap

    def transform(self, input_df) -> pd.DataFrame:
        for pattern in candlestick_patterns:
            pattern_function = getattr(talib, pattern)
            input_df[pattern] = pattern_function(input_df.open, input_df.high, input_df.low, input_df.close)
            self.indicators.append(pattern)
        return input_df


class CandleStickFactor(TechnicalFactor):
    def __init__(self, region: Region, entity_schema: EntityMixin = Stock, 
                 provider: Provider = Provider.Default, entity_provider: Provider = Provider.Default,
                 entity_ids: List[str] = None, exchanges: List[str] = None, codes: List[str] = None,
                 the_timestamp: Union[str, pd.Timestamp] = None, start_timestamp: Union[str, pd.Timestamp] = None,
                 end_timestamp: Union[str, pd.Timestamp] = None,
                 columns: List = ['entity_id', 'timestamp', 'open', 'close', 'high', 'low', 'volume'],
                 filters: List = None, order: object = None, limit: int = None,
                 level: Union[str, IntervalLevel] = IntervalLevel.LEVEL_1DAY, category_field: str = 'entity_id',
                 time_field: str = 'timestamp', computing_window: int = None, keep_all_timestamp: bool = False,
                 fill_method: str = 'ffill', effective_number: int = None,
                 accumulator: Accumulator = None, need_persist: bool = False, dry_run: bool = False,
                 kdata_overlap=3) -> None:
        self.kdata_overlap = kdata_overlap

        super().__init__(region, entity_schema, provider, entity_provider, entity_ids, exchanges, codes, the_timestamp,
                         start_timestamp, end_timestamp, columns, filters, order, limit, level, category_field,
                         time_field, computing_window, keep_all_timestamp, fill_method, effective_number,
                         accumulator=accumulator, need_persist=need_persist, dry_run=dry_run)

    def do_compute(self):
        super().do_compute()

        def cal_pattern(df):
            for pattern in candlestick_patterns:
                pattern_function = getattr(talib, pattern)
                df[pattern] = pattern_function(df.open, df.high, df.low, df.close)
            return df
        
        new_df = self.factor_df.reset_index(drop=True)
        gb = new_df.groupby('entity_id', sort=False)
        dfs = [cal_pattern(gb.get_group(x).copy()) for x in gb.groups]
        self.result_df = pd.concat(dfs)


if __name__ == '__main__':
    factor = CandleStickFactor(region=Region.US, entity_ids=['stock_NASDAQ_FB'], 
                               start_timestamp='2015-01-01', end_timestamp='2020-07-01',
                               kdata_overlap=4)

    gb = factor.result_df.groupby('entity_id')
    dfs = {x : gb.get_group(x) for x in gb.groups}
