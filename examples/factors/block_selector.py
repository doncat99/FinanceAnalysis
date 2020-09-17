# -*- coding: utf-8 -*-
from zvt.contract import IntervalLevel
from zvt.contract.common import Region, Provider
from zvt.domain import Block
from zvt.factors.money_flow_factor import BlockMoneyFlowFactor
from zvt.factors.target_selector import TargetSelector


class BlockSelector(TargetSelector):

    def __init__(self, region, entity_ids=None, entity_schema=Block, exchanges=None, codes=None, the_timestamp=None,
                 start_timestamp=None, end_timestamp=None, long_threshold=0.8, short_threshold=0.2,
                 level=IntervalLevel.LEVEL_1DAY, provider=Provider.Sina, block_selector=None) -> None:
        super().__init__(entity_ids, region, entity_schema, exchanges, codes, the_timestamp, start_timestamp, end_timestamp,
                         long_threshold, short_threshold, level, provider, block_selector)

    def init_factors(self, entity_ids, entity_schema, exchanges, codes, the_timestamp, start_timestamp, end_timestamp,
                     level):
        block_factor = BlockMoneyFlowFactor(region=self.region, start_timestamp=start_timestamp, end_timestamp=end_timestamp,
                                            provider=Provider.Sina, window=10)
        self.score_factors.append(block_factor)


if __name__ == '__main__':
    block_selector = BlockSelector(region=Region.CHN, start_timestamp='2019-01-01')
    block_selector.run()
    entity_ids = block_selector.get_targets(timestamp='2020-01-23')
    df = Block.query_data(region=Region.CHN, provider=Provider.Sina, entity_ids=entity_ids)
    print(df)
