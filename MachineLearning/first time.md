# first time

## 1. baseline相关

`Label` `0/1/2` —>下跌/不变/上涨？

启用GPU:   model = *clf*(*iterations*=500, ***task_type*="GPU"**, **params)

baseline的XY对：无时间序列相关特征，仅独立元素

​	x: `['date', 'sym', 'n_close', 'amount_delta', 'n_midprice', 'n_bid1', 'n_bsize1', 'n_bid2', 'n_bsize2', 'n_bid3', 'n_bsize3', 'n_bid4', 'n_bsize4', 'n_bid5', 'n_bsize5', 'n_ask1', 'n_asize1', 'n_ask2', 'n_asize2', 'n_ask3', 'n_asize3', 'n_ask4', 'n_asize4', 'n_ask5', 'n_asize5', 'wap1', 'hour', 'minute']`

​	y: `['label_5','label_10','label_20','label_40','label_60']`

修改代码输出，单独保存预测结果以节省读取csv时间

```python
train_df_result = pd.DataFrame()
test_df_result = pd.DataFrame()
for label in ['label_5','label_10','label_20','label_40','label_60']:
    print(f'=================== {label} ===================')
    cat_oof, cat_test = cv_model(CatBoostClassifier, train_df[cols], train_df[label], test_df[cols], 'cat')
    train_df_result[label] = np.argmax(cat_oof, axis=1)
    test_df_result[label] = np.argmax(cat_test, axis=1)
```



## 2. Advanced

> （1）**当前时间特征**：围绕买卖价格和买卖量进行构建，暂时只构建买一卖一和买二卖二相关特征，进行优化时可以加上其余买卖信息；
>
> （2）**历史平移特征**：通过历史平移获取上个阶段的信息；
>
> （3）**差分特征**：可以帮助获取相邻阶段的增长差异，描述数据的涨减变化情况。在此基础上还可以构建相邻数据比值变化、二阶差分等；
>
> （4）**窗口统计特征**：窗口统计可以构建不同的窗口大小，然后基于窗口范围进统计均值、最大值、最小值、中位数、方差的信息，可以反映最近阶段数据的变化情况。

- 修改代码启用GPU

  - lightgbm需要重新编译

    - 放弃使用

  - xgboost 使用pip重新安装

    - >  The `py-xgboost-gpu` is currently not available on Windows. If you are using Windows, please use `pip` to install XGBoost with GPU support.

    - 失败，无法使用

- 观察结果提升

## 3. TODO

超参

特征工程（`auto-ml`）

尝试`lstm`模型

