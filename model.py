import os
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from utils import huber_approx_obj



def search_best_param(X_train, y_train, X_val, y_val, param, param1, param2 = None, estimator = XGBRegressor, score = mean_absolute_error):
    '''
    通过grind search来调整模型超参数
        - max_depth: Maximum tree depth for base learners.单位树(base learner)最大深度
        - min_child_weight Minimum sum of instance weight(hessian) needed in a child.单位树最小分裂阀值。
        - MAE用作loss function
    Inputs: X_train, y_train, X_val, y_val: CV训练集和测试集
            param: 其他起始参数
            param1, param2: 目标超参数范围[1, 3, 5]
    Output: 分数和最佳参数集
    '''
    
    best_score = 10000000.
    best_param = { **param }
    para = { **param }
    key1 = list(param1.keys())[0]
    
    if param2 is not None:
        #min_child_weight search
        key2 = list(param2.keys())[0]
        for parama in param1[key1]:
            for paramb in param2[key2]:
                para[key1] = parama
                para[key2] = paramb
                est = estimator(objective=huber_approx_obj, **para)
                est.fit(X_train, y_train)
                y_pred = est.predict(X_val)
                current_score = score(y_pred, y_val)
                #print('The current score: ', current_score)
                #print('The current parameter: {} = {}, {} = {}'.format(key1, parama, key2, paramb))
                if (current_score < best_score):
                    best_score = current_score
                    best_param[key1] = parama
                    best_param[key2] = paramb
        #print('The best score: ', best_score)
        #print('The best parameter: {} = {}, {} = {}'.format(key1, best_param[key1], key2, best_param[key2]))
    else:
        #max_depth search
        for parama in param1[key1]:
            para[key1] = parama
            est = estimator(objective=huber_approx_obj, **para)
            est.fit(X_train, y_train)
            y_pred = est.predict(X_val)
            current_score = score(y_pred, y_val)
            print('The current score: ', current_score)
            print('The current parameter: {} = {}'.format(key1, parama))
            if (current_score < best_score):
                best_score = current_score
                best_param[key1] = parama
        #print('The best score: ', best_score)
        #print('The best parameter: {} = {}'.format(key1, best_param[key1]))
    return best_score, best_param


def train_xgbr(X, y, param, param1, param2, model_path='./model', test_size=0.2, estimator = XGBRegressor, score = mean_absolute_error):
    '''
    训练最佳参数模型
   
    Inputs: X, y, param, param1, param2: 参考参数搜索函数search_best_param()
                - 去除名字与日期 （日期可以进一步做特征工程，但当前版本暂不考虑
            test_size: 测试集比例
            model_path: 模型保存路径
            estimator: 参数搜索用模型
            score: 参数搜索用评分metric
             *注意：实际模型用Huber loss进行优化，相对普遍的square loss function对异常值比较不敏感，表现更加robust
    Output: 保存训练好的xgb模型至路径
    '''
    
    X = X.drop(['name', 'date'], axis=1)
    # CV dataset split w/o shuffling (not sure if suffuling is better anot)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=None)    
    
    #find best param
    best_score, best_param = search_best_param(X_train, y_train, X_val, y_val, param, param1, param2, estimator = XGBRegressor, score = mean_absolute_error)
    
    #initilize and train on training data
    best_xgbr = XGBRegressor(objective=huber_approx_obj, **best_param)
    best_xgbr.fit(X, y)
    
    # output found best_param and trained model
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    try:
        best_xgbr.save_model(os.path.join(model_path, 'xgb_model.json'))
    except:
        print("error saving the model")
    
    return best_xgbr



def make_prediction(X_test, model_path, output_path):
    '''
    读取模型及测试集，进行预测
    Inputs: test: 测试集路径
            model: 模型路径
    Outputs: pred: 预测结果，并保存pred.csv至路径output_path 
    '''
    
    X_test = X_test.drop(['name', 'date'], axis=1)
    
    model_xgb = xgb.Booster()
    model_xgb.load_model(os.path.join(model_path, 'xgb_model.json'))
    
    pred = model_xgb.predict(xgb.DMatrix(X_test))
    
    try:
        pd.DataFrame(pred).to_csv(os.path.join(output_path, 'pred.csv'), encoding='gbk', index=False)
    except:
        print('Fail to output pred.csv make sure path is correct')
        
    return pred