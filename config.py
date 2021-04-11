TARGET_COLS = {'日期' : 'date', '股票代码' : 'name', '次日vwap卖出收益率' : 'y'}
CONST_COLS = ['是否被st', '是否交易']

INITIAL_PARAM = {'learning_rate': 0.1,
         'verbosity': 0,
         #'objective': pseudohubererror, # added on spot
         'tree_method': 'hist',
         'n_estimators': 100,
         'n_jobs': -1,
         'gamma': 0,
         'subsample': 0.8,
         'colsample_bytree': 0.8,
         'alpha': 0}
PARAM_1 = { 'max_depth': [1, 3, 5] }
PARAM_2 = { 'min_child_weight': [1, 3, 5] }

WIW_SIZE = 10
TOP = 50
RANK =  5