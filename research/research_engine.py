#!/usr/bin/env python3
from datetime import datetime, timedelta, timezone
from typing import NewType, Union, Iterator, Callable
from abc import abstractmethod, abstractstaticmethod
from pathlib import Path
from utils.io_utils import profile
import dateutil.parser
import numpy as np
import pandas as pd
import os, sys, copy, itertools, sklearn, json, pickle
from scipy.interpolate import CubicSpline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier # so that supported models appear in globals()
from sklearn.covariance._robust_covariance import MinCovDet

from scipy.stats import multivariate_normal

# import shap
Instrument = NewType("Instrument", str)  # eg 'ETHUSDT for binance, or cvxETHSTETH for defillama
FileData = NewType("Data", dict[Instrument, pd.DataFrame])

RawFeature = NewType("RawFeature", str)  # eg 'volume'
RawData = NewType("Data", dict[tuple[Instrument, RawFeature], pd.Series])

FeatureExpansion = NewType("FeatureExpansion", Union[int, str])  # frequency in min
Feature = NewType("Feature", tuple[Instrument, RawFeature, FeatureExpansion])
Data = NewType("Data", dict[Feature, pd.Series])

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(index=df.index,
                        columns=df.columns,
                        data=StandardScaler().fit_transform(df))

def remove_duplicate_rows(df):
    return df[~df.index.duplicated()]

def weighted_ew_mean(temp_data: pd.DataFrame,
                     halflife: timedelta,
                     weights: pd.Series=None) -> pd.DataFrame:
    if weights is not None:
        result = (temp_data * weights)\
                     .ewm(times=temp_data.index, halflife=halflife).mean() \
                 / weights \
                     .ewm(times=temp_data.index, halflife=halflife).mean()
    else:
        result = temp_data \
                .ewm(times=temp_data.index, halflife=halflife).mean()
    return result

def weighted_ew_vol(temp_data: pd.DataFrame,
                    increment: int,
                    halflife: timedelta,
                    weights: pd.Series = None) -> pd.DataFrame:
    increments = temp_data.apply(np.log).diff(increment)
    if weights is not None:
        incr_sq = (increments * increments * weights)\
                     .ewm(times=temp_data.index, halflife=halflife).mean() \
                 / weights \
                     .ewm(times=temp_data.index, halflife=halflife).mean()
        incr_mean = (increments * weights)\
                     .ewm(times=temp_data.index, halflife=halflife).mean() \
                 / weights \
                     .ewm(times=temp_data.index, halflife=halflife).mean()
        result = (incr_sq - incr_mean*incr_mean).apply(np.sqrt)
    else:
        result = increments.ewm(times=temp_data.index, halflife=halflife).std()
    return result

def winning_trade(log_perf: pd.Series,
                  horizon: int,
                  lvl_scaling_window=72 * 60,
                  takeprofit=1.5,  # in stdev
                  stop=-0.25,
                  verbose=False) -> pd.Series:
    '''
    df are log returns not standardized
    returns:
    * 1  if long ended up takeprofit
    * -1 if short ended up takeprofit
    * 0  if both ended up stopped
    * note that both can't takeprofit if stop+takeprofit > 0
    '''
    result = pd.DataFrame({'log_perf': log_perf})
    result['rolling_std'] = result['log_perf'].rolling(lvl_scaling_window).std()
    result['cum_perf'] = result['log_perf'].cumsum()
    result['trade_outcome'] = 0
    result.dropna(inplace=True)

    cum_perf = result['cum_perf'].values
    rolling_std = result['rolling_std'].values
    trade_outcome = result['trade_outcome'].values

    for i in range(len(result.index)):
        future = (cum_perf[i:(i+horizon)] - cum_perf[i])/rolling_std[i]
        # longs: break on stop or takeprofit
        for x in future:
            if x < stop:
                break
            elif x > takeprofit:
                trade_outcome[i] = 1
                break
        # shorts: break on stop or takeprofit
        for x in future:
            if x > -stop:
                break
            elif x < -takeprofit:
                trade_outcome[i] = -1
                break
    return pd.Series(index=result.index,
                     name='winning_trade',
                     data=trade_outcome)

class ResearchEngine:

    # TODO: Proper value should be set instead of 0.
    execution_lag = 0                   # added just to fix the bug in compute_labels()
    data_interval = timedelta(days=1)   # created to fix the bug in ewma_expansion()

    def __init__(self, feature_map, label_map, run_parameters, input_data,**paramsNOTUSED):
        self.feature_map = feature_map
        self.label_map = label_map
        self.run_parameters = run_parameters
        self.input_data = input_data

        self.temp_feature_data: Data = Data(dict()) # temporary, until we have a DataFrame
        self.temp_label_data: Data = Data(dict())  # temporary, until we have a DataFrame
        self.performance: dict[Instrument, pd.Series] = dict()

        self.X: pd.DataFrame = pd.DataFrame()
        self.Y: pd.DataFrame = pd.DataFrame()

        self.models: list[sklearn.base.BaseEstimator] = []
        self.fitted_model: dict[tuple[RawFeature, FeatureExpansion, str, int], sklearn.base.BaseEstimator] = dict()

    def get_model(self, index=0):
        return list(self.fitted_model.values())[index]

    @staticmethod
    @abstractmethod
    def read_data(dirpath, selected_instruments: list[Instrument], start_date) -> FileData:
        raise NotImplementedError

    # def linear_unit_test(self, feature_data: pd.DataFrame, target_vol=1.0) -> Iterator[pd.DataFrame]:
    #     '''
    #     generate performance as a weighted mean of feature/windows (like LinearRegression)
    #     generate sign as an indicator over a weighted mean of feature/windows (like LogisticRegressionCV)
    #     '''
    #     nb_feature_per_instrument = int(feature_data.shape[1] / feature_data.columns.levshape[0])
    #     weights = np.random.normal(loc=0,
    #                                scale=1,
    #                                size=nb_feature_per_instrument)
    #
    #     result = pd.DataFrame(columns=pd.MultiIndex.from_tuples([], names=['instrument', 'feature', 'window']))
    #     for instrument, instrument_df in feature_data.groupby(level='instrument', axis=1):
    #         # record performance for use in backtest
    #         self.performance[instrument] = file_data[instrument]['close']
    #
    #         performance = pd.Series(index=instrument_df.index,
    #                                 data=StandardScaler().fit_transform(pd.DataFrame((instrument_df * weights).sum(axis=1)))[:, 0])
    #         performance *= target_vol * np.sqrt(ResearchEngine.data_interval.total_seconds() / timedelta(days=365.25).total_seconds())
    #
    #         for label, label_data in self.label_map.items():
    #             for horizon in label_data['horizons']:
    #                 temp = performance.rolling(horizon).sum()
    #                 if label == 'quantized_zscore':
    #                     result[(instrument, 'quantized_zscore', horizon)] = temp.apply(np.sign)
    #                 else:
    #                     raise NotImplementedError
    #
    #     self.temp_label_data = copy.deepcopy(result)
    #
    #     yield result

# TODO: Class FeatureEngine: transform_features, expand_features, xxx_exansions, compute_labels, build X_Y

#     def transform_features(self,
#                            file_data: FileData,
#                            feature_map: dict[RawFeature, dict]) -> RawData:
    @staticmethod
    def transform_features(file_data: FileData, feature_map: dict[RawFeature, dict]) -> RawData:
        '''
        transforms file_data (eg log volume)
        '''
        result: RawData = RawData(dict())
        for instrument, df in file_data.items():
            for raw_feature, params in feature_map.items():
                if not raw_feature in file_data[instrument]:
                    continue
                if 'transform' in params:
                    data = file_data[instrument][raw_feature]
                    if params['transform'] == 'log':
                        result[(instrument, raw_feature)] = data.apply(lambda x: np.log(max([x, 1e-32])))
                    if params['transform'] == 'compound':
                        temp = pd.DataFrame(data)
                        temp['dt'] = [0] + list(map(lambda dt: dt.total_seconds()/timedelta(days=365).total_seconds(),
                                              [data.index[i+1]-data.index[i] for i in range(len(data.index)-1)]))
                        temp[data.name] = np.exp(temp.apply(lambda x: x[data.name]*x['dt'], axis=1).cumsum())
                        result[(instrument, raw_feature)] = temp[data.name]
                    elif params['transform'] == 'arctanh':
                        result[(instrument, raw_feature)] = (
                                    file_data[instrument]['taker_buy_volume'] / file_data[instrument]['volume']).apply(
                            lambda x: np.arctanh(np.clip((2 * x - 1), 1e-8 - 1, 1 - 1e-8)))
                else:
                    result[(instrument, raw_feature)] = file_data[instrument][raw_feature]

                result[(instrument, raw_feature)] = remove_duplicate_rows(result[(instrument, raw_feature)])
                result[(instrument, raw_feature)].dropna(inplace=True)

        return result

    def expand_features(self, data_dict: RawData) -> Data:
        '''
        expands features
        '''
        result: Data = Data(dict())
        for (instrument, raw_feature), data in data_dict.items():
            # add several ewma, volume_weighted if requested
            for method, params in self.feature_map[raw_feature].items():
                if hasattr(self, method):
                    getattr(self, method)(data_dict, instrument, raw_feature, result, params)
        return result

    def as_is(self, data_dict, instrument, raw_feature, result, params) -> None:
        result[(instrument, raw_feature, f'as_is')] = data_dict[(instrument, raw_feature)]

    def ewma_expansion(self, data_dict, instrument, raw_feature, result, params) -> None:
        '''add several ewma, volume_weighted if requested'''
        temp: Data = Data(dict())
        if 'weight_by' in params:
            # exp because self.feature_map[volume] = log
            weights = np.exp(data_dict[(instrument, params['weight_by'])])
        else:
            weights = None

        all_windows = params['windows']
        for window in all_windows:
            # weighted mean = mean of weight*temp / mean of weigths
            data = weighted_ew_mean(data_dict[(instrument, raw_feature)],
                                    halflife=window * self.data_interval,
                                    weights=weights)
            if 'quantiles' in params:
                data = data.rolling(5*window).rank(pct=True)
            else:
                rolling = data.rolling(5 * window)
                data = (data-rolling.mean())/rolling.std()
            temp[window] = data
        if 'cross_ewma' in params and params['cross_ewma']:
            for i1, w1 in enumerate(all_windows):
                for i2, w2 in enumerate(all_windows[:i1]):
                    result[(instrument, raw_feature, f'ewma_{w2}_{w1}')] = temp[w2] - temp[w1]
        else:
            for window in all_windows:
                result[(instrument, raw_feature, f'ewma_{window}')] = temp[window]

    @staticmethod
    def min_expansion(data_dict, instrument, raw_feature, result, params) -> None:
        '''
        add several min max, may use volume clock.
        for windows = [t0, t1] returns minmax over [0,t0] and [t0,t1]
        '''
        if 'weight_by' in params:
            # exp because self.feature_map[volume] = log
            raise NotImplementedError
            volume = np.exp(data_dict[(instrument, params['weight_by'])])
            volume_cumsum = volume.cumsum() / volume.mean()
            volume_cumsum.drop(volume_cumsum.index[volume_cumsum.diff() == 0],inplace=True)
            volume_2_time = CubicSpline(volume_cumsum.values, volume_cumsum.index)
            volume_clocked = [data_dict[(instrument, raw_feature)].iloc[int(volume_2_time(i))]
                              for i in range(volume_cumsum.shape[0])]
        else:
            volume_clocked = data_dict[(instrument, raw_feature)].values

        windows = params['windows']
        for i, window in enumerate(windows):
            start = windows[i-1] if i>0 else 0
            length = window - start
            temp = data_dict[(instrument, raw_feature)].shift(start).rolling(length).min()
            result[(instrument, raw_feature, f'min_{start}_{window}')] = temp

    @staticmethod
    def max_expansion(data_dict, instrument, raw_feature, result, params) -> None:
        '''
        add several min max, may use volume clock.
        for windows = [t0, t1] returns minmax over [0,t0] and [t0,t1]
        '''
        if 'weight_by' in params:
            # exp because self.feature_map[volume] = log
            raise NotImplementedError
            volume = np.exp(data_dict[(instrument, params['weight_by'])])
            volume_cumsum = volume.cumsum() / volume.mean()
            volume_cumsum.drop(volume_cumsum.index[volume_cumsum.diff() == 0],inplace=True)
            volume_2_time = CubicSpline(volume_cumsum.values, volume_cumsum.index)
            volume_clocked = [data_dict[(instrument, raw_feature)].iloc[int(volume_2_time(i))]
                              for i in range(volume_cumsum.shape[0])]
        else:
            volume_clocked = data_dict[(instrument, raw_feature)].values

        windows = params['windows']
        for i, window in enumerate(windows):
            start = windows[i-1] if i>0 else 0
            length = window - start
            temp = data_dict[(instrument, raw_feature)].shift(start).rolling(length).max()
            result[(instrument, raw_feature, f'max_{start}_{window}')] = temp

    def hvol_expansion(self, data_dict, instrument, raw_feature, result, params) -> None:
        '''add several historical vols quantiles, volume_weighted if requested'''
        if 'weight_by' in params:
            # exp because self.feature_map[volume] = log
            weights = np.exp(data_dict[(instrument, params['weight_by'])])
        else:
            weights = None

        for increment_window in params['incr_window']:
            # weigthed std  = weighted mean of squares - square of weighted mean
            temp = weighted_ew_vol(data_dict[(instrument, raw_feature)],
                                   increment=increment_window[0],
                                   halflife=increment_window[1] * self.data_interval,
                                   weights=weights)
            if 'quantiles' in params:
                temp = temp.rolling(5*increment_window[1]).rank(pct=True)
            else:
                rolling = temp.rolling(5 * increment_window[1])
                temp = (temp-rolling.mean())/rolling.std()
            result[(instrument, raw_feature, f'hvol_{increment_window[0]}_{increment_window[1]}')] = temp

    def compute_labels(self, file_data: FileData, label_map: dict[RawFeature, dict]) -> Data:
        '''
        performance, sign, big and stop_limit from t+1 to t+1+horizon
        '''
        result: Data = Data(dict())
        for instrument, raw_feature in itertools.product(file_data.keys(), label_map.keys()):
            # record performance for use in backtest
            self.performance[instrument] = file_data[instrument]['close']

            for horizon in label_map[raw_feature]['horizons']:
                future_perf = (file_data[instrument]['close'].shift(-horizon - ResearchEngine.execution_lag) / file_data[instrument]['close'].shift(-ResearchEngine.execution_lag)).apply(np.log)
                if raw_feature == 'quantized_zscore':
                    z_score = future_perf / future_perf.shift(horizon).rolling(50*horizon).std()

                    def quantized(x,buckets=label_map[raw_feature]['stdev_buckets']):
                        return next((key for key,values in buckets.items() if values[0] < x <= values[1]), np.NaN)
                    temp = z_score.apply(quantized).dropna()
                elif raw_feature == 'stop_limit':
                    log_perf = (file_data[instrument]['close'].shift(-1) /
                                   file_data[instrument]['close']).apply(np.log)
                    temp = winning_trade(log_perf, horizon)
                else:
                    raise NotImplementedError

                feature = (instrument, raw_feature, horizon)
                result[feature] = remove_duplicate_rows(temp).rename(feature)

        return result

    def build_X_Y(self, file_data: FileData, feature_map: dict[RawFeature, dict], label_map: dict[RawFeature, dict], unit_test=False) -> None:
        '''
        aggregates features and labels
        '''
        raw_data = self.transform_features(file_data, feature_map)
        self.temp_feature_data = self.expand_features(raw_data)

        if unit_test:
            self.X = pd.concat(self.temp_feature_data, axis=1).dropna()
            self.X.columns.rename(['instrument', 'feature', 'window'], inplace=True)
            self.Y = next(self.linear_unit_test(self.X))
            X_Y = pd.concat([self.X, self.Y], axis=1).dropna()
            # concat / dropna to have same index
            self.X = normalize(X_Y[self.X.columns])
            if 'normalize' in self.run_parameters and self.run_parameters['normalize']:
                self.X = normalize(self.X)
            self.Y = X_Y[self.Y.columns]
        else:
            self.temp_label_data = self.compute_labels(file_data, label_map)
            # concat / dropna to have same index
            X_Y = pd.concat(self.temp_feature_data | self.temp_label_data, axis=1)#TODO:.dropna()
            X_Y.columns.rename(['instrument', 'feature', 'window'], inplace=True)
            self.X = X_Y[self.temp_feature_data.keys()]
            if 'normalize' in self.run_parameters and self.run_parameters['normalize']:
                self.X = normalize(self.X)
            self.Y = X_Y[self.temp_label_data.keys()]

    @abstractmethod
    def fit(self):
        raise NotImplementedError


class BinanceExtremeResearchEngine(ResearchEngine):
    data_interval = timedelta(seconds=60)
    execution_lag = 1
    stdev_quantum = 1.5
    stdev_cap = 3


    @staticmethod
    def read_data(dirpath, selected_instruments: list[Instrument], start_date) -> FileData:
        file_data: FileData = FileData(dict())
        start_date = dateutil.parser.isoparse(start_date)
        dirname = os.path.join(os.sep, Path.home(), *dirpath)
        for filename in os.listdir(dirname):
            filesplit = filename.split('-')
            instrument: Instrument = Instrument(filesplit[0])
            data_type: RawFeature = RawFeature(filesplit[-1].replace('.csv', ''))
            if Instrument(instrument) in selected_instruments:
                new_df = pd.read_csv(os.path.join(os.sep, dirname, filename))
                new_df['open_time'] = new_df['open_time'].apply(lambda t: datetime.fromtimestamp(t / 1000))
                new_df = new_df.set_index('open_time').sort_index(ascending=True)[start_date:]
                if instrument in file_data and data_type in file_data[instrument]:
                    file_data[instrument][data_type] = pd.concat([file_data[instrument][data_type], new_df])
                else:
                    if instrument in file_data:
                        file_data[instrument][data_type] = new_df
                    else:
                        file_data[instrument] = {data_type: new_df}
        for instrument, data in file_data.items():
            file_data[instrument] = remove_duplicate_rows(
                data['klines'].join(
                    data['premium'], rsuffix='_premium', how='outer'
                ).ffill())

        #  just inserted a hack to print out marks, here. nothing to do with CTA.
        if False:
            marks = pd.concat({instrument: remove_duplicate_rows(data['mark'].ffill())
                               for instrument, data in file_data.items()}, axis=1)
            marks.to_csv(os.path.join(os.sep, Path.home(), "mktdata", "binance", "results", "marks.csv"))

        return file_data

    def linear_unit_test(self, feature_data: pd.DataFrame, target_vol=1.0) -> Iterator[pd.DataFrame]:
        '''
        generate performance as a weighted mean of feature/windows (like LinearRegression)
        generate sign as an indicator over a weighted mean of feature/windows (like LogisticRegressionCV)
        '''
        nb_feature_per_instrument = int(feature_data.shape[1] / feature_data.columns.levshape[0])
        weights = np.random.normal(loc=0,
                                   scale=1,
                                   size=nb_feature_per_instrument)

        result = pd.DataFrame(columns=pd.MultiIndex.from_tuples([], names=['instrument', 'feature', 'window']))
        for instrument, instrument_df in feature_data.groupby(level='instrument', axis=1):
            # record performance for use in backtest
            self.performance[instrument] = file_data[instrument]['close']

            performance = pd.Series(index=instrument_df.index,
                                    data=StandardScaler().fit_transform(pd.DataFrame((instrument_df * weights).sum(axis=1)))[:, 0])
            performance *= target_vol * np.sqrt(self.data_interval.total_seconds() / timedelta(days=365.25).total_seconds())

            for label, label_data in self.label_map.items():
                for horizon in label_data['horizons']:
                    temp = performance.rolling(horizon).sum()
                    if label == 'quantized_zscore':
                        result[(instrument, 'quantized_zscore', horizon)] = temp.apply(np.sign)
                    else:
                        raise NotImplementedError

        self.temp_label_data = copy.deepcopy(result)

        yield result

    def fit(self) -> None:
        '''
        for each label/model/fold, fits one model for all instruments.
        '''
        # for each (all instruments, 1 feature, 1 horizon)...
        for (raw_feature, frequency), label_df in self.Y.groupby(level=['feature', 'window'], axis=1):
            # for each model for that feature...
            if raw_feature not in self.run_parameters['models']:
                continue
            for model_name, model_params in self.run_parameters['models'][raw_feature].items():
                if 'hidden_layer_sizes' in model_params['params']:
                    model_params['params']['hidden_layer_sizes'] = (model_params['params']['hidden_layer_sizes'],)
                model_obj = globals()[model_name](**model_params['params'])
                # ...cross validation
                n_splits = int(model_params['splits_for_1000'] * np.log2(len(self.X.index) / frequency) / 10)
                tscv = TimeSeriesSplit(n_splits=max(2, n_splits),
                                       gap=frequency)
                # cross_val = cross_validate(model_obj, X_allinstruments, Y_allinstruments,
                #                #scoring=[],
                #                n_jobs=-1,
                #                cv=tscv,
                #                return_estimator=True, return_train_score=True)
                # for split_index in range(n_splits):
                #     self.fitted_model[(raw_feature, frequency, model_name, split_index)] = cross_val['estimator'][split_index]
                # fitted_model = VotingClassifier(estimators=list(zip(map(str, range(n_splits)), cross_val['estimator'])), voting='soft')
                for split_index, (train_index, test_index) in enumerate(tscv.split(self.X)):
                    # fit 1 model on all instruments,on train_set index.
                    X_allinstruments_train = pd.concat([self.X.iloc[train_index].xs(instrument, level='instrument', axis=1)
                                                  for instrument in self.input_data['selected_instruments']],
                                                 axis=0)
                    Y_allinstruments_train = pd.concat([label_df.iloc[train_index].xs(instrument, level='instrument', axis=1)
                                                  for instrument in self.input_data['selected_instruments']],
                                                 axis=0).squeeze()
                    fitted_model = model_obj.fit(X_allinstruments_train, Y_allinstruments_train)
                    self.fitted_model[(raw_feature, frequency, model_name, split_index)] = (copy.deepcopy(fitted_model),test_index)

                    # display score
                    X_allinstruments_test = pd.concat([self.X.iloc[test_index].xs(instrument, level='instrument', axis=1)
                                                  for instrument in self.input_data['selected_instruments']],
                                                 axis=0)
                    Y_allinstruments_test = pd.concat([label_df.iloc[test_index].xs(instrument, level='instrument', axis=1)
                                                  for instrument in self.input_data['selected_instruments']],
                                                 axis=0).squeeze()
                    #predY_allinstruments_test = fitted_model.predict(X_allinstruments_test)
                    print(f'      r2 = {fitted_model.score(X_allinstruments_test, Y_allinstruments_test)} for split {split_index} / model {model_name} / label {(raw_feature, frequency)}')

                    if 'pickle_output' in self.run_parameters \
                            and not os.path.isfile(self.run_parameters['pickle_output']):
                        with open(self.run_parameters['pickle_output'], 'wb') as fp:
                            pickle.dump(self, fp)
                    print(f'    ran split {split_index} for {model_name} for {(raw_feature, frequency)}')
                print(f'  ran {model_name} for {(raw_feature, frequency)}')
            print(f'ran {(raw_feature, frequency)}')


class TrivialEwmPredictor(sklearn.base.BaseEstimator):
    def __init__(self, halflife: str, cap: float, horizon: timedelta):
        self.halflife = pd.Timedelta(halflife)
        self.cap = cap
        self.horizon = pd.Timedelta(horizon)
        self.apy: pd.DataFrame = pd.DataFrame()
        self.tvl: pd.DataFrame = pd.DataFrame()

    def predict(self, index: datetime) -> tuple[np.array, np.array]:
        '''returns APYs and tvls (used to estimate dilution)'''
        return self.apy.loc[index].values, self.tvl.loc[index].values

    def fit(self, raw_X: pd.DataFrame) -> None:
        '''
        precompute, for speed
        '''
        apy_X = raw_X.xs(level=['feature', 'window'], key=['apy', 'as_is'], axis=1).ffill().fillna(0.0)
        # select only rows of X that deviate from mean by less than 3 stdev
        mean = apy_X.ewm(times=apy_X.index, halflife=self.halflife).mean()
        stdev = apy_X.ewm(times=apy_X.index, halflife=self.halflife).std()
        X = pd.DataFrame(index=apy_X.index, columns=apy_X.columns)
        for col in X.columns:
            mask = np.abs(apy_X[col] - mean[col]) <= self.cap * stdev[col]
            X[col] = np.where(mask, apy_X[col], mean[col])

        # fillna implies pools that do not yet exist have 0 yield / tvl
        self.apy = X.ewm(times=X.index, halflife=self.halflife).mean().fillna(0.0)

        # add apy from mean reversion
        for instrument, subframe in raw_X.groupby(level='instrument', axis=1):
            depeg = dict()
            for col, price in subframe.groupby(level='feature', axis=1):
                if 'underlying' in col:
                    depeg[col] = (price - price.ewm(times=price.index, halflife=self.halflife).mean()).ffill().fillna(0.0)
            if depeg != dict():
                self.apy[instrument] -= pd.concat(depeg, axis=1).mean(axis=1).fillna(0.0) / self.horizon.total_seconds() * 365.25 * 24 * 60 * 60
                self.apy[instrument] = self.apy[instrument].clip(lower=0.0)
        self.tvl = raw_X.xs(level=['feature', 'window'], key=['tvl', 'as_is'], axis=1).ffill().fillna(0.0)

        return
        #
        # decay = pd.Series(index=X.index, data=np.exp(
        #     -(X.index[-1] - X.index) / self.halflife))
        # decayed_X = X.mul(decay, axis=0)
        # if raw_X.shape[0] > 1:
        #     if False:
        #         # TODO: remove outliers and interpolate them
        #         outlier_remover = MinCovDet().fit(raw_X)
        #         # decayed_X[~outlier_remover.support_] = np.nan
        #         # decayed_X = decayed_X.interpolate(method='linear')
        #         mean = outlier_remover.location_/decay.sum()
        #         cov = outlier_remover.covariance_/(decay * decay).sum()
        #     else:
        #         # we take the std over entire history !
        #         mean = (decayed_X.sum()/decay.sum())
        #         cov = decayed_X.cov()/(decay * decay).sum()
        #
        #         # some series may have NaNs. cleanup.
        #         mean = mean.fillna(0.0).values
        #         for i in range(cov.shape[0]):
        #             if np.isnan(cov.iloc[i, i]):
        #                 cov.iloc[i, i] = 1.0
        #         cov = cov.fillna(0.0)
        #
        #         #TODO: somehow still not posdef...give up for now
        #         cov = np.eye(raw_X.shape[1])
        # else:
        #     mean = decayed_X.squeeze().fillna(0.0).values
        #     cov = np.identity(raw_X.shape[1])
        # self.distribution = multivariate_normal(mean=mean, cov=cov, allow_singular=True)


class DefillamaResearchEngine(ResearchEngine):
    '''
    simple ewma predictor for yield
    '''
    data_interval = timedelta(days=1)
    execution_lag = 0

    @staticmethod
    def read_data(dirpath, selected_instruments: list[Instrument], start_date) -> FileData:
        file_data: FileData = FileData(dict())
        start_date = dateutil.parser.isoparse(start_date).replace(tzinfo=timezone.utc)
        # Made path relative to project root directory instead of absolute one.
        dirname = os.path.join(os.sep, Path.cwd(), *dirpath)
        # dirname = os.path.join(os.sep, Path.home(), *dirpath)

        for filename in os.listdir(dirname):
            filesplit = filename.split('.csv')
            instrument: Instrument = Instrument(filesplit[0])
            if Instrument(instrument) in selected_instruments:
                new_df = pd.read_csv(os.path.join(os.sep, dirname, filename),
                                     parse_dates=['date'],
                                     date_parser=lambda x: pd.to_datetime(x, unit='ns', utc=True))
                new_df = new_df.set_index('date').sort_index(ascending=True)[start_date:]
                for data_type in new_df.columns:
                    if instrument in file_data and data_type in file_data[instrument]:
                        file_data[instrument][data_type] = pd.concat([file_data[instrument][data_type], new_df[data_type]])
                    else:
                        if instrument in file_data:
                            file_data[instrument][data_type] = new_df[data_type]
                        else:
                            file_data[instrument] = {data_type: new_df[data_type]}

        return file_data

    def compute_labels(self, file_data: FileData, label_map: dict[RawFeature, dict]) -> Data:
        '''
        performance, sign, big and stop_limit from t+1 to t+1+horizon
        '''
        result: Data = Data(dict())
        for instrument, raw_feature in itertools.product(file_data.keys(), label_map.keys()):
            for horizon in label_map[raw_feature]['horizons']:
                temp = self.performance[instrument].rolling(window=horizon).mean().shift(-horizon)
                feature = (instrument, raw_feature, horizon)
                result[feature] = remove_duplicate_rows(temp).rename(feature)

        return result


    def fit(self):
        '''
        for each label/model/fold, fits one model for all instruments.
        '''
        # for each (all instruments, 1 feature, 1 horizon)...
        for (raw_feature, frequency), label_df in self.Y.groupby(level=['feature', 'window'], axis=1):
            # for each model for that feature...
            if raw_feature not in self.run_parameters['models']:
                continue
            for model_name, model_params in self.run_parameters['models'][raw_feature].items():
                model_obj = globals()[model_name](**model_params['params'])
                # no fit needed for trivial model
                model_obj.fit(self.X)
                split_index = 0
                self.fitted_model[(raw_feature, frequency, model_name, split_index)] = copy.deepcopy(model_obj)


def model_analysis(engine: ResearchEngine):
    feature_list=list(engine.X.xs(engine.input_data['selected_instruments'][0], level='instrument', axis=1).columns)
    result = []
    for model_tuple, (model, _) in engine.fitted_model.items():
        # result.append(pd.Series(name=model_tuple,
        #                         index=feature_list,
        #                         data=shap.Explainer(model)(engine.X)))
        if hasattr(model, 'intercept_') and hasattr(model, 'coef_'):
            if len(model.classes_) == 2:
                data = np.append(model.intercept_,model.coef_)
            else:
                data = np.append(np.dot(model.classes_, model.intercept_), np.dot(model.classes_, model.coef_))
            result.append(pd.Series(name=model_tuple,
                                    index=[('intercept', None)] + feature_list,
                                    data=data))
        if hasattr(model, 'feature_importances_'):
            result.append(pd.Series(name=model_tuple,
                                    index=feature_list,
                                    data=model.feature_importances_))
    return pd.concat(result, axis=1)


def build_ResearchEngine(input_parameters) -> ResearchEngine:
    parameters = copy.deepcopy(input_parameters)
    research_name = parameters['input_data'].pop("ResearchName")
    if research_name == "defillama":
        '''defillama'''
        result = DefillamaResearchEngine(**parameters)
    elif research_name == "binance_extreme":
        '''binance_extreme'''
        result = BinanceExtremeResearchEngine(**parameters)
    else:
        raise ValueError(f"Invalid 'research_name' value specified: {research_name}")

    file_data = result.read_data(**{**parameters['input_data'], 'start_date': parameters['backtest']['start_date']})

    # record performance for use in backtest
    for instrument in file_data:
        result.performance[instrument] = file_data[instrument]['apy']

    result.build_X_Y(file_data, parameters['feature_map'], parameters['label_map'],
                     unit_test=parameters['run_parameters']['unit_test'])
    result.fit()
    print(f'fit engine')

    return result
