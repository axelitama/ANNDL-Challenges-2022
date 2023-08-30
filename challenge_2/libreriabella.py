import numpy as np
import pandas as pd

def get_normalize_params(x_ds):
    min_max = []
    
    for feature in range(x_ds.shape[2]):
        feature_ds = []
        for i in range(x_ds.shape[0]):
            for k in range(x_ds.shape[1]):
                feature_ds.append(x_ds[i][k][feature])
        min_max.append((min(feature_ds), max(feature_ds)))
    return min_max

def get_standardize_params(data):
    normalized = {}
    for m in range(data.shape[2]):
        x = np.reshape(data[:, :, m], (data.shape[0] * data.shape[1]))
        normalized[f'feature {m}'] = x

    normalized = pd.DataFrame(normalized)
    stats_df = normalized.describe()
    list = []
    for i in range(normalized.shape[1]):
        list.append([stats_df[f'feature {i}'].loc['mean'], stats_df[f'feature {i}'].loc['std']])

    return list

def get_robust_params(data):
    normalized = {}
    for m in range(data.shape[2]):
        x_ = np.reshape(data[:, :, m], (data.shape[0] * data.shape[1]))
        normalized[f'feature {m}'] = x_

    normalized = pd.DataFrame(normalized)
    stats_df = normalized.describe()
    list = []
    for i in range(normalized.shape[1]):
        list.append([stats_df[f'feature {i}'].loc['25%'], stats_df[f'feature {i}'].loc['50%'], stats_df[f'feature {i}'].loc['75%']])
    return list

def normalize(x_ds, params=None):
    if params is None:
        min_max = np.array(get_normalize_params(x_ds))
    else:
        min_max = np.array(params)
    
    x_ds_out = x_ds - min_max[:, 0]
    x_ds_out = x_ds_out / (min_max[:, 1] - min_max[:, 0])

    return x_ds_out, min_max

def standardize(x_ds, params=None):
    if params is None:
        avg_var = np.array(get_standardize_params(x_ds))
    else:
        avg_var = np.array(params)
    
    x_ds_out = x_ds - avg_var[:, 0]
    x_ds_out = x_ds_out / avg_var[:, 1]

    return x_ds_out, avg_var

def robustScale(x_ds, params=None):
    if params is None:
        quantiles = np.array(get_robust_params(x_ds))
    else:
        quantiles = np.array(params)

    x_ds_out = x_ds - quantiles[:, 1]
    x_ds_out = x_ds_out / (quantiles[:, 2] - quantiles[:, 0])

    return x_ds_out, quantiles

def single_norm(x_ds):        
    samples = []
    for i in range(x_ds.shape[0]):
        sample = []
        for feature in range(x_ds.shape[2]):
            feature_ds = []
            for k in range(x_ds.shape[1]):
                feature_ds.append(x_ds[i][k][feature])
            feature_ds = np.array(feature_ds)
            feature_ds = (feature_ds-min(feature_ds))/(max(feature_ds)-min(feature_ds))
            sample.append(feature_ds)
        samples.append(np.transpose(sample))
    return np.array(samples)

def log(x_ds):
    x_ds_out = np.zeros(x_ds.shape)
    pos_mask = x_ds > 0
    neg_mask = x_ds < 0
    x_ds_out[pos_mask] = np.log(1 + x_ds[pos_mask])
    x_ds_out[neg_mask] = -np.log(1 - x_ds[neg_mask])
    return x_ds_out

def _merge_features(ds1, ds2):
    # Dataset x and y must be consistent
    assert ds1.shape[0] == ds2.shape[0]
    assert ds1.shape[1] == ds2.shape[1]
    
    ds_out = []
    for i in range(ds1.shape[0]):
        line = []
        for k in range(ds1.shape[1]):
            line.append((list(ds1[i][k])+list(ds2[i][k])))
        ds_out.append(line)
    return np.array(ds_out)

def merge_features(dses):
    # Must merge more than one dataset
    assert len(dses) > 0
    
    if len(dses) == 1:
        return dses[0]
    
    ds_tot = dses[0]
    for ds in dses[1:]:
        ds_tot = _merge_features(ds_tot, ds)
    return ds_tot

def reduce_window(target_window, x_ds, y_ds, stride = 1):
    window = x_ds.shape[1]
    
    # Dataset x and y must be consistent
    assert x_ds.shape[0] == y_ds.shape[0]
    
    # Target window must be less or equal than window
    assert target_window <= window
    
    x_ds_out=[]
    y_ds_out=[]
    
    for i in range(x_ds.shape[0]):
        for k in range(0,window-target_window+1,stride):
            x_ds_out.append(x_ds[i][0+k:target_window+k])
            y_ds_out.append(y_ds[i])
    
    return np.array(x_ds_out), np.array(y_ds_out)

def hypersample(x_ds, y_ds, division = 1):
    window = x_ds.shape[1]
    target_window = window - 1
    
    # Dataset x and y must be consistent
    assert x_ds.shape[0] == y_ds.shape[0]
    
    # Not implemented AHAHA
    assert division == 1
    
    x_ds_out=np.mean((x_ds[:,:-1], x_ds[:,1:]), axis=0)
    y_ds_out=y_ds
    
    return np.array(x_ds_out), np.array(y_ds_out)

class preprocess_data:
    def __init__(self):
        self._normParams = None
        self._stdParams = None
        self._lognormParams = None
        self._logstdParams = None

    def setPredictParams(self, normParams=None, stdParams=None, lognormParams=None, logstdParams=None):
        self._normParams = normParams
        self._stdParams = stdParams
        self._lognormParams = lognormParams
        self._logstdParams = logstdParams

    def getPredictParams(self):
        return self._normParams, self._stdParams, self._lognormParams, self._logstdParams

    def preprocess_train(self, x_train, x_val, y_train, y_val, window=36, stride=1, val_stride=1, features=['nothing']):
        features_train = []
        features_val = []
        for feature in features:
            match feature:
                case 'nothing':
                    features_train.append(x_train)
                    features_val.append(x_val)
                case 'norm':
                    x_norm, param_norm = normalize(x_train)
                    x_val_norm, _ = normalize(x_val, param_norm)
                    features_train.append(x_norm)
                    features_val.append(x_val_norm)
                    self._normParams = param_norm
                case 'std':
                    x_std, param_std = standardize(x_train)
                    x_val_std, _ = standardize(x_val, param_std)
                    features_train.append(x_std)
                    features_val.append(x_val_std)
                    self._stdParams = param_std
                case 'log':
                    features_train.append(log(x_train))
                    features_val.append(log(x_val))
                case 'lognorm':
                    x_lognorm, param_lognorm = normalize(log(x_train))
                    x_val_lognorm, _ = normalize(log(x_val), param_lognorm)
                    features_train.append(x_lognorm)
                    features_val.append(x_val_lognorm)
                    self._lognormParams = param_lognorm
                case 'logstd':
                    x_logstd, param_logstd = standardize(log(x_train))
                    x_val_logstd, _ = standardize(log(x_val), param_logstd)
                    features_train.append(x_logstd)
                    features_val.append(x_val_logstd)
                    self._logstdParams = param_logstd
                case 'singlenorm':
                    features_train.append(single_norm(x_train))
                    features_val.append(single_norm(x_val))
                # case 'robust':
                #     x_robust, param_robust = robustScale(x_train)
                #     x_val_robust, _ = robustScale(x_val, param_logrobust)
                #     features_train.append(x_robust)
                #     features_val.append(x_val_robust)
                #     self._stdRobust = param_robust
                # case 'logrobust':
                #     x_logrobust, param_logrobust = robustScale(log(x_train))
                #     x_val_logrobust, _ = robustScale(log(x_val), param_logrobust)
                #     features_train.append(x_logrobust)
                #     features_val.append(x_val_logrobust)
                #     self._stdRobust = param_logrobust

        x_train_merge = merge_features(features_train)
        x_val_merge = merge_features(features_val)   

        x_train_merge_reduced, y_train_reduced = reduce_window(window, x_train_merge, y_train, stride)
        x_val_merge_reduced, y_val_reduced = reduce_window(window, x_val_merge, y_val, val_stride)

        return x_train_merge_reduced, x_val_merge_reduced, y_train_reduced, y_val_reduced

    def preprocess_predict(self, x, window=36, stride=1, features=['nothing']):
        features_predict = []
        for feature in features:
            match feature:
                case 'nothing':
                    features_predict.append(x)
                case 'norm':
                    x_norm, _ = normalize(x, self._normParams)
                    features_predict.append(x_norm)
                case 'std':
                    x_std, _ = standardize(x, self._stdParams)
                    features_predict.append(x_std)
                case 'log':
                    features_predict.append(log(x))
                case 'lognorm':
                    x_lognorm, _ = normalize(log(x), self._lognormParams)
                    features_predict.append(x_lognorm)
                case 'logstd':
                    x_logstd, _ = standardize(log(x), self._logstdParams)
                    features_predict.append(x_logstd)
                case 'singlenorm':
                    features_predict.append(single_norm(x))
                # case 'robust':

        x_predict_merge = merge_features(features_predict)  

        x_predict_merge_reduced, _ = reduce_window(window, x_predict_merge, x_predict_merge, stride)

        return x_predict_merge_reduced