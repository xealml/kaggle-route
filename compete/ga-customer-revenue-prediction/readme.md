- [compete url](https://www.kaggle.com/c/ga-customer-revenue-prediction/)
- [basic baseline solution](https://www.kaggle.com/sudalairajkumar/simple-exploration-baseline-ga-customer-revenue)
- [feature-engineer and stacking solution](https://www.kaggle.com/augustmarvel/base-model-v2-user-level-solution)
- [Exploring the Consumer Patterns + ML Pipeline](https://www.kaggle.com/kabure/exploring-the-consumer-patterns-ml-pipeline)
    - better pipeline
    - handle missing value
    - handle data value
    - best data plot ever
- [Winning solution (link to kernel inside)](https://www.kaggle.com/c/ga-customer-revenue-prediction/discussion/82614)
    - winning solution 
- 总结
    - 数据
        1. 表格型数据
        2. 对于很多类别列,直接使用label encoding,针对类别encoding也可以使用上面提供到的min max等扩充特征
        3. 对于数值型列,使用min max mean 等直接扩充特征
    - 模型
        1. kaggle画风为树模型直接刚,stacking 直接用