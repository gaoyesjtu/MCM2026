Q1结构：

1.Q1建模：数据读入与建模分析，主要使用了带有多重复杂淘汰规则限制的MILP求解，在consistency上达到了100%。

2.Q1可视化（两个文件）：初步的可视化操作，有待细化

3.certainty_week:以空间收缩率作为certainty的度量，计算单位为week；结果为一个csv

4.certainty_person:以频率作为rank的个人certainty；以MAD作为percent的个人certainty;结果分为两个csv保存

5.各种csv文件：实验结果；各种png文件：初步的可视化图片