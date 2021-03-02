# -*- coding: utf-8 -*-
from typing import List, Union

import pandas as pd

from zvt.api.data_type import Region, Provider
from zvt.domain import Stock
from zvt.contract import IntervalLevel, EntityMixin
from zvt.contract.factor import Accumulator, Transformer
from zvt.factors.algorithm import IntersectTransformer
from zvt.factors.technical_factor import TechnicalFactor


class SqueezeFactor(TechnicalFactor):
    def __init__(self, region: Region, entity_schema: EntityMixin = Stock,
                 provider: Provider = Provider.Default,
                 entity_ids: List[str] = None, exchanges: List[str] = None, codes: List[str] = None,
                 the_timestamp: Union[str, pd.Timestamp] = None, start_timestamp: Union[str, pd.Timestamp] = None,
                 end_timestamp: Union[str, pd.Timestamp] = None,
                 columns: List = ['id', 'entity_id', 'timestamp', 'open', 'close', 'high', 'low', 'code'],
                 filters: List = None, order: object = None, limit: int = None,
                 level: Union[str, IntervalLevel] = IntervalLevel.LEVEL_1DAY, category_field: str = 'entity_id',
                 time_field: str = 'timestamp', computing_window: int = None, keep_all_timestamp: bool = False,
                 fill_method: str = 'ffill', effective_number: int = None,
                 accumulator: Accumulator = None, need_persist: bool = False, dry_run: bool = False,
                 kdata_overlap=3) -> None:
        self.kdata_overlap = kdata_overlap
        transformer: Transformer = IntersectTransformer(kdata_overlap=self.kdata_overlap)

        super().__init__(region, entity_schema, provider, entity_ids, exchanges, codes, the_timestamp,
                         start_timestamp, end_timestamp, columns, filters, order, limit, level, category_field,
                         time_field, computing_window, keep_all_timestamp, fill_method, effective_number, transformer,
                         accumulator, need_persist, dry_run)

    def do_compute(self):
        super().do_compute()

        def squeeze(df):
            df['20sma'] = df['close'].rolling(window=20).mean()
            df['stddev'] = df['close'].rolling(window=20).std()
            df['lower_band'] = df['20sma'] - (2 * df['stddev'])
            df['upper_band'] = df['20sma'] + (2 * df['stddev'])

            df['TR'] = abs(df['high'] - df['low'])
            df['ATR'] = df['TR'].rolling(window=20).mean()

            df['lower_keltner'] = df['20sma'] - (df['ATR'] * 1.5)
            df['upper_keltner'] = df['20sma'] + (df['ATR'] * 1.5)

            def in_squeeze(df):
                return df['lower_band'] > df['lower_keltner'] and df['upper_band'] < df['upper_keltner']

            df['squeeze_on'] = df.apply(in_squeeze, axis=1)
            return df

        new_df = self.factor_df.reset_index(drop=True)
        gb = new_df.groupby('entity_id', sort=False)
        dfs = [squeeze(gb.get_group(x).copy()) for x in gb.groups]

        self.result_df = pd.concat(dfs)


if __name__ == '__main__':
    factor = SqueezeFactor(region=Region.US, entity_ids=['stock_NASDAQ_FB'],
                           start_timestamp='2015-01-01', end_timestamp='2020-07-01',
                           kdata_overlap=4)

    gb = factor.result_df.groupby('entity_id')
    dfs = {x: gb.get_group(x) for x in gb.groups}
