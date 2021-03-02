import warnings
warnings.filterwarnings("ignore")
# import faulthandler
# faulthandler.enable()

import logging
import time
from collections import Counter

import numpy as np
from ta import add_all_ta_features
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoLarsIC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import RocCurveDisplay, roc_curve, auc, classification_report, confusion_matrix, precision_score
from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from zvt.api.data_type import Region, Provider
from zvt.contract.reader import DataReader
from zvt.domain import Stock1dKdata, Stock


logger = logging.getLogger(__name__)


def dataXY(df, train_size=0.7, pred_future=10):
    def create_labels(y_cohort, pred_future):
        y = (y_cohort.diff(periods=pred_future).shift(-pred_future).dropna()>=0).astype(int)
        return y

    def create_ta(x_cohort, pred_future):
        x = add_all_ta_features(x_cohort, 'open', 'high', 'low', 'close', 'volume', fillna=True).shift(-pred_future).dropna()
        return x

    train_cohort = df[0:round(df.shape[0] * train_size)]
    x_train_cohort = train_cohort.iloc[:, 1:7]
    x_train = create_ta(x_train_cohort, pred_future)
    y_train_cohort = train_cohort['close']
    y_train = create_labels(y_train_cohort, pred_future)

    test_cohort = df[round(df.shape[0] * (train_size)):]
    x_test_cohort = test_cohort.iloc[:, 1:7]
    x_test = create_ta(x_test_cohort, pred_future)
    y_test_cohort = test_cohort['close']
    y_test = create_labels(y_test_cohort, pred_future)
    y_test_cohort = y_test_cohort.shift(-pred_future).dropna()

    return x_train, y_train, x_test, y_test, y_test_cohort


class LASSOJorn(BaseEstimator, TransformerMixin):
    def __init__(self):
        None

    def fit(self, X, y):
        self.model = LassoLarsIC(criterion='aic').fit(X, y)
        return self

    def transform(self, X):
        return np.asarray(X)[:, abs(self.model.coef_) > 0]


if __name__ == '__main__':
    now = time.time()
    reader = DataReader(region=Region.US,
                        codes=['FB', 'AMD'],
                        data_schema=Stock1dKdata,
                        entity_schema=Stock,
                        provider=Provider.Yahoo)

    gb = reader.data_df.groupby('code')
    dfs = {x: gb.get_group(x) for x in gb.groups}

    df = dfs['AMD'][['open', 'close', 'volume', 'high', 'low']].copy()
    x_train, y_train, x_test, y_test, y_test_cohort = dataXY(df)

    plt.close()

    parameters = {
    #    'clf__base_estimator__n_estimators': np.round(np.linspace(100,400,10)).astype('int'),
    #    'clf__base_estimator__max_depth': [10,11,12],
    #    'clf__base_estimator__min_child_weight': [1],
    #    'clf__base_estimator__gamma': np.linspace(0,0.5,5),
    #    'clf__base_estimator__subsample': np.linspace(0.2,0.4,3),
    #    'clf__base_estimator__colsample_bytree': np.linspace(0.2,0.4,3),
    #    'clf__base_estimator__reg_alpha': np.linspace(0.01,0.03,10)
    #    'clf__method': ['isotonic','sigmoid'],
    }

    scale_pos_weight = Counter(y_train)[0] / Counter(y_train)[1]
    clf = xgb.XGBRFClassifier(objective='binary:logistic',
                              scale_pos_weight=scale_pos_weight,
                              learning_rate=0.01,
                              n_estimators=5000,
                              max_depth=10,
                              min_child_weight=1,
                              gamma=0,
                              subsample=0.3,
                              colsample_bytree=0.3,
                              reg_alpha=0.014,
                              nthread=4,
                              seed=27)

    PL = Pipeline(steps=[('PreProcessor', StandardScaler()),
                         ('PCA', PCA()),
                         ('EmbeddedSelector', LASSOJorn()),
                         ('clf', CalibratedClassifierCV(base_estimator=clf, method='sigmoid'))])

    # tss = TimeSeriesSplit(n_splits=3)
    # optimizer = GridSearchCV(PL, parameters, cv=tss, n_jobs=-1, verbose=10, scoring='roc_auc')
    # optimizer.fit(x_train, y_train)
    # print(optimizer.best_params_)
    # final_model = optimizer.best_estimator_
    final_model = PL.fit(x_train, y_train)

    # plt.plot(optimizer.cv_results_['mean_test_score'])
    # xgb.plot_importance(final_model.named_steps['clf'])

    y_pred_proba = final_model.predict_proba(x_test)[:, 1]
    y_pred = final_model.predict(x_test)

    fraction_of_positives, mean_predicted_value = calibration_curve(np.array(y_test), y_pred_proba, strategy='uniform', n_bins=20)
    plt.figure()
    plt.plot(mean_predicted_value, fraction_of_positives, "sr-")
    plt.title("Calibration")
    plt.xlabel("mean_predicted_value")
    plt.ylabel("fraction_of_positives")

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=None)
    display.plot()
    plt.title("ROC")

    range_class = np.linspace(np.min(y_pred_proba), np.max(y_pred_proba), 100)
    range_class = np.delete(range_class, 0)
    range_class = np.delete(range_class, -1)
    PPV = np.zeros(len(range_class))
    NPV = np.zeros(len(range_class))

    j = 0
    for i in range_class:
        PPV[j] = precision_score(y_test, y_pred_proba > i, pos_label=1)
        NPV[j] = precision_score(y_test, y_pred_proba > i, pos_label=0)
        j += 1

    plt.figure()
    plt.plot(range_class, PPV, label='PPV')
    plt.plot(range_class, NPV, label='NPV')
    plt.legend()
    threshold = 0.98
    threshold_high = range_class[np.where(PPV > threshold)[0][0]]
    threshold_low = range_class[np.where(NPV < threshold)[0][0]]
    plt.plot(threshold_high, PPV[np.where(np.isin(range_class, threshold_high))[0][0]], 'r*')
    plt.plot(threshold_low, NPV[np.where(np.isin(range_class, threshold_low))[0][0]], 'r*')

    plt.figure(figsize=(10, 10))
    idx = np.linspace(0, 100, 101).astype('int')
    plt.plot(range(len(y_test_cohort.iloc[idx])), y_test_cohort.iloc[idx], 'b')
    idx_high = np.where(y_pred_proba[idx] > threshold_high)[0]
    plt.plot(idx_high, np.asarray(y_test_cohort)[idx_high], 'g.')
    idx_low = np.where(y_pred_proba[idx] < threshold_low)[0]
    plt.plot(idx_low, np.asarray(y_test_cohort)[idx_low], 'r.')

    idx_sure = np.sort(np.concatenate((idx_high, idx_low)))
    print(classification_report(y_test.iloc[idx_sure], y_pred[idx_sure]))
    print(confusion_matrix(y_test.iloc[idx_sure], y_pred[idx_sure]))

    plt.close('all')

    def bot(threshold_high, threshold_low):
        koers = df.iloc[round(df.shape[0] * (0.7)):, 4]

        start = np.zeros(len(x_test) + 1)
        start[0] = 10000
        bought = 0
        sellat = 0
        buyat = 0
        for i in range(len(x_test)):
            if y_pred_proba[i] > threshold_high and bought == 0:
                # print("Buy at i=",i)
                buyat = koers.iloc[i]

                if sellat != 0:
                    interest = -(buyat - sellat) / sellat
                    start[i + 1] = start[i] * (1 + interest)
                else:
                    start[i + 1] = start[i]
                bought = 1

            elif y_pred_proba[i] < threshold_low and bought == 1:
                # print("Sell at i=",i)
                sellat = koers.iloc[i]

                if buyat != 0:
                    interest = (sellat - buyat) / buyat
                    start[i + 1] = start[i] * (1 + interest)
                else:
                    start[i + 1] = start[i]
                bought = 0

            else:
                start[i + 1] = start[i]

        return start


    # range_class = np.linspace(min(y_pred_proba),max(y_pred_proba),100)
    # interest = np.zeros((range_class.shape[0],range_class.shape[0]))
    # ii=0
    # for i in range_class:
    #     jj=0
    #     for j in range_class:
    #         start = bot(i,j)
    #         interest[ii,jj] = start[-1]/start[0]*100/len(start)
    #         jj+=1
    #     ii+=1
    # ind = np.unravel_index(np.argmax(interest), interest.shape)

    start = bot(0.6, 0.46)
    interest = start[-1] / start[0] * 100 / len(start)
    print("interest: ", interest)
    plt.figure()
    plt.plot(start[:5 * 250])
    plt.show()
