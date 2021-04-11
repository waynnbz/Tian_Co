运行 `pip install -r requirements.txt` 安装所有依赖包 （建议创建的新环境下安装）
***
## 文件结构：

- main.py 
  - 最高层封装，支持click cli指令直接运行。具体操作记录在后面
- config.py
  - 全球变量
    - 数据变量
      - TARGET_COL 目标特征名字的map
      - CONST_COL 这是不变的常数项
      - 其他变量在preprocess函数中自动搜索，并保存至model_path，以便清洗未知测试集，保持一致性。每次训练会重新生成相应的列表。
    - 模型参数
      - INITIAL_PARAM 起始的基本参数
      - PARAM_1, PARAM_2 主要搜索的超参范围
    - 评估绘图变量
      - WIN_SIZE：目标窗口天数，x轴
      - TOP： int,当天收益最高的股票采样数量，
      - RANK: 评估分阶的组数
- utils.py
  - 所有帮助函数，都已注释，详细可以参考具体函数文档
- model.py
  - 主要包括XGB模型
    - grind-search最佳参数
    - 训练模型
    - 用模型预测


# CLI指令
CLI运行python main.py的全部选项如下：
- '--train':
  - 文件路径：训练数据；如提供，将重新训练模型，并保存/覆盖至当前模型路径
- '--test':
  - 文件路径：测试数据; 如未提供，将从默认路径使用样本测试集：./data/sample.test.csv
- '--model':
  - 文件夹路径：模型的读取/保存路径；默认路径：./model
- --output':
  - 文件夹路径：存储预测结果 pred.csv以及评测图evaluation_score.png；默认路径：./output
- '--plot':
  - Flag 是否绘制训练集的特征相关性，并保存其热力图corr_matrix.png至output_path

### 示例：（注：未提供选项将使用默认值
- 情况1- 提供测试数据，读取指定路径已训练模型进行预测：
  `python main.py --test <测试文件路径> --model <模型文件夹路径>`

- 情况2- 提供训练数据及测试数据，重新训练模型并存去指定路径，进行预测存去指定路径，并绘制相关性热力图：
 `python main.py --train <训练文件路径> --test <测试文件路径> --model <模型文件夹路径> --output <结果存储文件夹路径> --plot`

***
 ## EDA
 - EDA python notebook 包括在目录中供参考