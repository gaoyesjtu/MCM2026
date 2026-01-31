import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 加载推理数据
df = pd.read_csv('fan_estimate_with_rank.csv')

# 1. 数据归一化处理 (1.0 代表当周最佳)
df['norm_judge'] = 0.0
df['norm_fan'] = 0.0

# 百分比制标准化
mask_p = df['rule_type'] == 'Percentage'
df.loc[mask_p, 'norm_judge'] = df.loc[mask_p, 'judge_percent'] / df.loc[mask_p, 'judge_percent'].max()
df.loc[mask_p, 'norm_fan'] = df.loc[mask_p, 'predicted_fan_vote'] / df.loc[mask_p, 'predicted_fan_vote'].max()

# 排名制标准化 (Rank 1 -> 1.0, Max Rank -> 0.0)
mask_r = df['rule_type'].str.contains('Rank')
for s, g in df[mask_r].groupby('season'):
    idx = g.index
    max_rj, max_rf = g['judge_rank'].max(), g['predicted_fan_vote'].max()
    df.loc[idx, 'norm_judge'] = (max_rj - g['judge_rank'] + 1) / max_rj
    df.loc[idx, 'norm_fan'] = (max_rf - g['predicted_fan_vote'] + 1) / max_rf

# 计算偏差
df['fan_bias'] = df['norm_fan'] - df['norm_judge']

# 2. 绘图
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# A. 评委 vs 粉丝 (分布)
sns.scatterplot(data=df, x='norm_judge', y='norm_fan', hue='is_eliminated',
                palette={True: '#e74c3c', False: '#3498db'}, alpha=0.4, ax=axes[0,0])
axes[0,0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[0,0].set_title('Normalized Performance: Judge vs. Inferred Fan', fontsize=14, fontweight='bold')
axes[0,0].set_xlabel('Normalized Judge Score (1.0 = Best)')
axes[0,0].set_ylabel('Normalized Fan Vote (1.0 = Best)')

# B. 偏差密度分析 (展示民意分歧)
sns.kdeplot(data=df, x='fan_bias', hue='rule_type', fill=True, ax=axes[0,1])
axes[0,1].axvline(0, color='red', linestyle='--')
axes[0,1].set_title('Fan Preference Bias (Fan Score - Judge Score)', fontsize=14, fontweight='bold')
axes[0,1].set_xlabel('Bias Score (>0 means Fans liked them more than Judges)')

# C. 跨赛季争议波动
sample_seasons = [1, 5, 10, 20, 28, 34]
subset = df[df['season'].isin(sample_seasons)]
sns.boxplot(data=subset, x='season', y='fan_bias', palette='Set2', ax=axes[1,0])
axes[1,0].set_title('Bias Volatility Over Sampled Seasons', fontsize=14, fontweight='bold')

# D. 十大“民选黑马”决赛选手
finalists = df[df['results'].str.contains('1st|2nd|3rd', na=False, case=False)]
top_horses = finalists.sort_values('fan_bias', ascending=False).head(12)
sns.barplot(data=top_horses, x='fan_bias', y='celebrity_name', hue='season', palette='magma', ax=axes[1,1])
axes[1,1].set_title('Top 12 "Fan Favorites" Finalists (Highest Positive Bias)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('带rank的可视化.png')