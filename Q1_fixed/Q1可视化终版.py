import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 加载归一化后的数据
df = pd.read_csv('dwts_harmonized_fan_votes.csv')

# 设置绘图风格
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

# 图 A：全局表现对比 (评委 vs 粉丝)
# 颜色区分不同规则，形状区分是否被淘汰
sns.scatterplot(data=df, x='judge_percent', y='harmonized_fan_percent',
                hue='rule_type', style='is_eliminated', alpha=0.5, ax=axes[0, 0])
axes[0, 0].plot([0, 0.4], [0, 0.4], 'r--', alpha=0.6, label='Equality Line')
axes[0, 0].set_title('Global Performance: Judge % vs Harmonized Fan %', fontsize=14)

# 图 B：谁是历史上的“观众缘”之王？
# 计算 Fan Bias = 粉丝投票% - 评委评分%
df['fan_bias'] = df['harmonized_fan_percent'] - df['judge_percent']
top_bias = df.groupby('celebrity_name')['fan_bias'].mean().sort_values(ascending=False).head(15).reset_index()
sns.barplot(data=top_bias, x='fan_bias', y='celebrity_name', palette='magma', ax=axes[0, 1])
axes[0, 1].set_title('Top 15 "Fan Favorites" (Highest Avg Fan Bias)', fontsize=14)

# 图 C：规则演变对粉丝影响力的改变
# 将赛季划分为三个时代
df['era'] = df['season'].apply(lambda x: 'Early (S1-2)' if x <= 2 else ('Modern (S28+)' if x >= 28 else 'Classic (S3-27)'))
sns.boxplot(data=df, x='era', y='fan_bias', palette='Set2', ax=axes[1, 0])
axes[1, 0].set_title('Bias Distribution by Show Era', fontsize=14)

# 图 D：代表性冠军的人气曲线
# 选取几个代表性冠军进行追踪
sample_stars = ['Kelly Monaco', 'Robert Irwin', 'Bindi Irwin', 'Charli D\'Amelio']
trajectory = df[df['celebrity_name'].isin(sample_stars)].sort_values(['season', 'week'])
sns.lineplot(data=trajectory, x='week', y='harmonized_fan_percent', hue='celebrity_name', marker='o', ax=axes[1, 1])
axes[1, 1].set_title('Popularity Trajectory of Selected Winners', fontsize=14)

plt.tight_layout()
plt.savefig('percent可视化.png')