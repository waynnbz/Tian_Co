import pandas as pd
import numpy as np
import pickle
import json
import os
from scipy.spatial.distance import cosine
from mlxtend.plotting import heatmap
from sklearn import preprocessing
import matplotlib.pyplot as plt
# %matplotlib inline

from config import TARGET_COLS, CONST_COLS

def preprocess(data, model_path, cos_dist_threshold=0.1):
    '''
    最低限度数据预处理: 
        - 更改目标列名字， 方便数据操作
        - 去除泄露数据(cosine dist > 0.1)的列，无效常数列(从EDA中人工检索到的)
        - 转换时间数据datatype
        - 简单去除含有异常值的列 （可以进一步细节处理，暂不考虑
        - 没有做填充处理
            - 因为数据缺失占比较低(max missing in a col < 0.05)
            - 且XGB模型有学习处理缺失数据的机制(Sparsity-aware Split Finding)
    Input: cos_dist_threshold: 特征与目标cosine距离相似度的阀值
            model_path: 含有训练集预处理的cache：
                - leak_cols: 数据泄露项列表
                - abn_cols: 异常值项列表
    '''
    
    # rename key functional cols
    data.rename(columns=TARGET_COLS, inplace=True)
    
    # finding data leak
    leak_cols_path = os.path.join(model_path, 'leak_cols.pkl')
    if os.path.isfile(leak_cols_path):
        with open(leak_cols_path, 'rb') as f:
            leak_cols = pickle.load(f)
    else:
        leak_cols = []
        for c in data.columns:
            try:
                if c != 'y':
                    dist = 1 - cosine(data.y, data[c]) #具体结果参照EDA
                    if dist > cos_dist_threshold:
                        leak_cols.append(c)
            except:
                # to mute warning for known non-numeric feature
                non_num_col = ['name']
                if c not in non_num_col:
                    print(f'Unable to calculate cosine dist for {c}, check if its numeric')
        with open(leak_cols_path, 'wb') as f:
            pickle.dump(leak_cols, f)
        

    # drop features 
#     print(leak_cols)
    data.drop(leak_cols, axis=1, inplace=True)
    data.drop(CONST_COLS, axis=1, inplace=True)

    # convert dates to pd.datetime
    data.date = pd.to_datetime(data.date, format='%Y%m%d')
    
    # seperate out label and train cols
    y = data.y
    X = data.drop(['y'], axis=1)
    
    # take out feature cols with infinite values
    abn_cols_path = os.path.join(model_path, 'abn_cols.pkl')
    if os.path.isfile(abn_cols_path):
        with open(abn_cols_path, 'rb') as f:
            abn_cols = pickle.load(f)
    else:
        std = X.describe().loc['std']
        abn_cols = std[std.isnull()].index.tolist() #具体结果参照EDA, 可能包括有价值特征，暂不考虑
        with open(abn_cols_path, 'wb') as f:
            pickle.dump(abn_cols, f)
    
    X = X.drop(abn_cols, axis=1)
    
    # filling missing values/NaN
    # X = X.fillna(-999)
    
    return X, y


def load_data(input_path, model_path):
    
    try:
        df = pd.read_csv(input_path, encoding='gbk')
    except:
        print("Fail to load the data, check {input_path}")
        df = pd.read_csv(input_path, encoding='gbk') #just raises error and exits
        
    X, y = preprocess(df, model_path)

    return X, y


def plot_corr(X, y, output_path='./output/plot', plot_it=False):
    '''
    绘制Pearson特征相关性热力图
    Input: X, y: 预处理过的数据集 并在计算相关性前进行归一化
            output_path: 保存路径
            plot_it: 是否显示
    Output: corr_matrix.png: 热力图
            col_dict: 图上index对应的特征名
    '''
    
    X = X.drop(['name', 'date'], axis=1)
    X['y'] = y
    
    # Pearson correlation coefficience matrix
    cm = X.corr(method='pearson')

    col_dict = {k: v for k, v in enumerate(X.columns)}
    hm = heatmap(np.array(cm), row_names = list(col_dict.keys()), column_names = list(col_dict.keys()), figsize = (40, 40))
    plt.title('Features Correlation Heatmap', fontsize = 20)
    
    # output plot and col_dict
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    plt.savefig(os.path.join(output_path, 'corr_matrix.png'), dpi=300)
    with open(os.path.join(output_path, 'col_dict.json'), 'w') as f:
        json.dump(col_dict, f)
        
    if plot_it:
        plt.show()



# Pseduo Huber loss 
def huber_approx_obj(y_true, y_pred):
    z = y_pred - y_true
    delta = 1
    scale = 1 + (z/delta)**2
    scale_sqrt = np.sqrt(scale)
    grad = z/scale_sqrt
    hess = 1/(scale*scale_sqrt)
    return grad, hess


#Quantitative Scoring using MAPE
def MAPE(gt, pred):
    mape = []

    for g, p in zip(gt, pred):
        mape.append(max(0, 1 - abs((g-p)/g)))

    return np.mean(mape)


def evaluate(pred, gt, data, output_path, window_size=10, top=50, ranks=5):
    '''
    Input: pred: 模型预测的目标值
            gt: 真实的目标值
            data: 包含日期的测试集X_test
            window_size: 目标窗口天数
            top: 
            rank: 分阶总数
    Output: temp: 组合了预测及真实和一些标签的临时df
            scores: 根据当天真实y值分成#rank组之后，其每组对应股票的预测值的平均值
    '''
    
    date_split = data.date.unique()[-window_size]
    row_counts = (data.date >= date_split).sum()
    
    temp = pd.DataFrame()
    temp['pred'] = pred[-row_counts:]
    temp['gt'] = gt[-row_counts:].tolist()
    temp['date'] = data.date[-row_counts:].tolist()
    temp['name'] = data.name[-row_counts:].tolist()
    
    scores = {}

    for d in temp.date.unique():

        top_index = temp.groupby(['date'])['pred'].nlargest(top)[d].index

        for i in range(ranks):
            group_size = int(top/ranks) 
            sub_index = top_index[i*group_size : (i+1)*group_size]
            scores.setdefault(i+1, []).append(temp.iloc[sub_index]['gt'].mean())
        
    scores = pd.DataFrame.from_dict(scores)
    scores.set_index(temp.date.unique(), inplace=True)
    
    ax = scores.plot(figsize=(12,8), grid=True)
    fig = ax.get_figure()
    fig.savefig(os.path.join(output_path, 'evaluation_plot.png'))
        
    return temp, scores