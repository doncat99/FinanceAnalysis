import warnings
warnings.filterwarnings("ignore")

import logging

from zvt import zvt_env
from zvt.contract.common import Region
from zvt.factors.solo_factor import SoloFactor

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    factor = SoloFactor(region=Region.CHN, entity_ids=['stock_sz_000338'], 
                        start_timestamp='2015-01-01', end_timestamp='2020-07-01',
                        kdata_overlap=4)
    print(factor.result_df[factor.result_df['score']])
    print(len(factor.result_df))

