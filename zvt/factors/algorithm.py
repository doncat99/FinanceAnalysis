# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from stockstats import StockDataFrame

from zvt.factors.factor import Scorer, Transformer
from zvt.utils.pd_utils import normal_index_df


tech_indicator = ['macd', 'rsi_30', 'cci_30', 'dx_30']

class TechnicalIndicator:
    """Provides methods for preprocessing the stock price data

    Attributes
    ----------
        df: DataFrame
            data downloaded from Yahoo API
            7 columns: A timestamp, open, high, low, close, volume and entity_df symbol
            for the specified stock ticker
        use_technical_indicator : boolean
            we technical indicator or not
        tech_indicator_list : list
            a list of technical indicator names (modified from config.py)
        use_turbulence : boolean
            use turbulence index or not
        user_defined_feature:boolean
            user user defined features or not

    Methods
    -------
    get_indicators()
        main method to do the feature engineering

    """
    def __init__(self,
        use_technical_indicator=True,
        tech_indicator_list = tech_indicator,
        use_turbulence=False,
        user_defined_feature=False):

        self.use_technical_indicator = use_technical_indicator
        self.tech_indicator_list = tech_indicator_list
        self.use_turbulence=use_turbulence
        self.user_defined_feature=user_defined_feature

        #type_list = self._get_type_list(5)
        #self.__features = type_list
        #self.__data_columns = config.DEFAULT_DATA_COLUMNS + self.__features


    def get_indicators(self, data):
        """main method to do the feature engineering
        @:param config: source dataframe
        @:return: a DataMatrices object
        """
        df = data.copy()

        # add technical indicators
        # stockstats require all 5 columns
        if (self.use_technical_indicator==True):
            # add technical indicators using stockstats
            df=self.add_technical_indicator(df)
            print("Successfully added technical indicators")

        # add turbulence index for multiple stock
        if self.use_turbulence==True:
            df = self.add_turbulence(df)
            print("Successfully added turbulence index")

        # add user defined feature
        if self.user_defined_feature == True:
            df = self.add_user_defined_feature(df)
            print("Successfully added user defined features")

        # fill the missing values at the beginning and the end
        df=df.fillna(method='bfill').fillna(method="ffill")
        return df


    def add_technical_indicator(self, data):
        """
        calcualte technical indicators
        use stockstats package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df_indicator = []
        for _, new_df in data.groupby(level=0):
            stock = StockDataFrame.retype(new_df)
            new_df = new_df.copy()
            for indicator in self.tech_indicator_list:
                new_df[indicator] = stock[indicator]
            df_indicator.append(new_df)
        df = pd.concat(df_indicator)
        return df

    def add_user_defined_feature(self, data):
        """
         add user defined features
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """          
        df = data.copy()
        df['daily_return']=df.close.pct_change(1)
        #df['return_lag_1']=df.close.pct_change(2)
        #df['return_lag_2']=df.close.pct_change(3)
        #df['return_lag_3']=df.close.pct_change(4)
        #df['return_lag_4']=df.close.pct_change(5)
        return df


    def add_turbulence(self, data):
        """
        add turbulence index from a precalcualted dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df.reset_index(drop=True, inplace=True)

        turbulence_index = self.calcualte_turbulence(df)
        df = df.merge(turbulence_index, on='timestamp')
        df = df.sort_values(['timestamp','entity_id']).reset_index(drop=True)
        return df


    def calcualte_turbulence(self, data):
        """calculate turbulence index based on dow 30"""
        # can add other market assets
        df = data.copy()
        df_price_pivot=df.pivot(index='timestamp', columns='entity_id', values='close')
        unique_date = df.timestamp.unique()
        # start after a year
        start = 252
        turbulence_index = [0]*start
        #turbulence_index = [0]
        count = 0
        for i in range(start,len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            hist_price = df_price_pivot[[n in unique_date[0:i] for n in df_price_pivot.index ]]
            cov_temp = hist_price.cov()
            current_temp=(current_price - np.mean(hist_price,axis=0))
            temp = current_temp.values.dot(np.linalg.inv(cov_temp)).dot(current_temp.values.T)
            if temp > 0:
                count += 1
                turbulence_temp = temp[0][0] if count > 2 else 0
            else:
                turbulence_temp = 0
            turbulence_index.append(turbulence_temp)
        
        
        turbulence_index = pd.DataFrame({'timestamp':df_price_pivot.index,
                                         'turbulence':turbulence_index})
        return turbulence_index

    def _get_type_list(self, feature_number):
        """
        :param feature_number: an int indicates the number of features
        :return: a list of features n
        """
        if feature_number == 1:
            type_list = ["close"]
        elif feature_number == 2:
            type_list = ["close", "volume"]
            #raise NotImplementedError("the feature volume is not supported currently")
        elif feature_number == 3:
            type_list = ["close", "high", "low"]
        elif feature_number == 4:
            type_list = ["close", "high", "low", "open"]
        elif feature_number == 5:
            type_list = ["close", "high", "low", "open","volume"]  
        else:
            raise ValueError("feature number could not be %s" % feature_number)
        return type_list


def point_in_range(point, range):
    return range[0] <= point <= range[1]


def intersect_ranges(range_list):
    result = intersect(range_list[0], range_list[1])
    for range_i in range_list[2:]:
        result = intersect(result, range_i)
    return result


def intersect(range_a, range_b):
    if not range_a or not range_b:
        return None
    # 包含
    if point_in_range(range_a[0], range_b) and point_in_range(range_a[1], range_b):
        return range_a
    if point_in_range(range_b[0], range_a) and point_in_range(range_b[1], range_a):
        return range_b

    if point_in_range(range_a[0], range_b):
        return range_a[0], range_b[1]

    if point_in_range(range_b[0], range_a):
        return range_b[0], range_a[1]
    return None


class IntersectTransformer(Transformer):
    def __init__(self, kdata_overlap=0) -> None:
        super().__init__()
        self.kdata_overlap = kdata_overlap

    def transform(self, input_df) -> pd.DataFrame:
        if self.kdata_overlap > 0:
            # 没有重叠，区间就是(0,0)
            input_df['overlap'] = [(0, 0)] * len(input_df.index)

            def cal_overlap(s):
                high = input_df.loc[s.index, 'high']
                low = input_df.loc[s.index, 'low']
                intersection = intersect_ranges(list(zip(low.to_list(), high.to_list())))
                if intersection:
                    # 设置column overlap为intersection,即重叠区间
                    input_df.at[s.index[-1], 'overlap'] = intersection
                return 0

            input_df[['high', 'low']].groupby(level=0).rolling(window=self.kdata_overlap,
                                                               min_periods=self.kdata_overlap).apply(
                cal_overlap, raw=False)

        return input_df


class MaAndVolumeTransformer(Transformer):
    def __init__(self, windows=[5, 10], vol_windows=[30], kdata_overlap=0) -> None:
        super().__init__()
        self.windows = windows
        self.vol_windows = vol_windows
        self.kdata_overlap = kdata_overlap

    def transform(self, input_df) -> pd.DataFrame:
        for window in self.windows:
            col = 'ma{}'.format(window)
            self.indicators.append(col)

            ma_df = input_df['close'].groupby(level=0).rolling(window=window, min_periods=window).mean()
            ma_df = ma_df.reset_index(level=0, drop=True)
            input_df[col] = ma_df

        for vol_window in self.vol_windows:
            col = 'vol_ma{}'.format(vol_window)
            self.indicators.append(col)

            vol_ma_df = input_df['volume'].groupby(level=0).rolling(window=vol_window, min_periods=vol_window).mean()
            vol_ma_df = vol_ma_df.reset_index(level=0, drop=True)
            input_df[col] = vol_ma_df

        if self.kdata_overlap > 0:
            input_df['overlap'] = [(0, 0)] * len(input_df.index)

            def cal_overlap(s):
                high = input_df.loc[s.index, 'high']
                low = input_df.loc[s.index, 'low']
                intersection = intersect_ranges(list(zip(low.to_list(), high.to_list())))
                if intersection:
                    input_df.at[s.index[-1], 'overlap'] = intersection
                return 0

            input_df[['high', 'low']].groupby(level=0).rolling(window=self.kdata_overlap,
                                                               min_periods=self.kdata_overlap).apply(
                cal_overlap, raw=False)

        return input_df


class TechnicalTransformer(Transformer):
    def __init__(self) -> None:
        super().__init__()

    def transform(self, input_df) -> pd.DataFrame:
        return TechnicalIndicator(use_turbulence=True).get_indicators(input_df)


class RankScorer(Scorer):
    def __init__(self, ascending=True) -> None:
        self.ascending = ascending

    def score(self, input_df) -> pd.DataFrame:
        result_df = input_df.groupby(level=1).rank(ascending=self.ascending, pct=True)
        return result_df


# class QuantileScorer(Scorer):
#     def __init__(self, score_levels=[0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]) -> None:
#         self.score_levels = score_levels

#     def score(self, input_df):
#         self.score_levels.sort(reverse=True)

#         quantile_df = input_df.groupby(level=1).quantile(self.score_levels)
#         quantile_df.index.names = [self.time_field, 'score']

#         self.logger.info('factor:{},quantile:\n{}'.format(self.factor_name, quantile_df))

#         result_df = input_df.copy()
#         result_df.reset_index(inplace=True, level='entity_id')
#         result_df['quantile'] = None
#         for timestamp in quantile_df.index.levels[0]:
#             length = len(result_df.loc[result_df.index == timestamp, 'quantile'])
#             result_df.loc[result_df.index == timestamp, 'quantile'] = [quantile_df.loc[
#                                                                            timestamp].to_dict()] * length

#         self.logger.info('factor:{},df with quantile:\n{}'.format(self.factor_name, result_df))

#         # result_df = result_df.set_index(['entity_id'], append=True)
#         # result_df = result_df.sort_index(level=[0, 1])
#         #
#         # self.logger.info(result_df)
#         #
#         def calculate_score(df, factor_name, quantile):
#             original_value = df[factor_name]
#             score_map = quantile.get(factor_name)
#             min_score = self.score_levels[-1]

#             if original_value < score_map.get(min_score):
#                 return 0

#             for score in self.score_levels[:-1]:
#                 if original_value >= score_map.get(score):
#                     return score

#         for factor in input_df.columns.to_list():
#             result_df[factor] = result_df.apply(lambda x: calculate_score(x, factor, x['quantile']),
#                                                 axis=1)

#         result_df = result_df.reset_index()
#         result_df = normal_index_df(result_df)
#         result_df = result_df.loc[:, self.factors]

#         result_df = result_df.loc[~result_df.index.duplicated(keep='first')]

#         self.logger.info('factor:{},df:\n{}'.format(self.factor_name, result_df))

#         return result_df


def consecutive_count(input_df, col, pattern=[-5, 1]):
    for entity_id, df in input_df.groupby(level=0):
        count = 0

        negative = 0

        current_state = None
        for index, item in df[col].iteritems():
            if item:
                state = 'up'
            else:
                state = 'down'

            # 计算维持状态（'up','down'）的 次数
            if current_state == state:
                if count > 0:
                    count = count + 1
                else:
                    count = count - 1
                    negative = count
            else:
                current_state = state

                if current_state == 'up':
                    count = 1
                else:
                    count = -1
                    negative = count

            if (count >= pattern[1]) and (negative <= pattern[0]):
                input_df.loc[index, 'score'] = True
            else:
                input_df.loc[index, 'score'] = True

            # 设置目前状态
            input_df.loc[index, 'count'] = count

        print(f'consecutive_count for {entity_id}')
