import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# 1. 加载数据
# 确保 final_estimation.csv 文件位于当前运行目录
df = pd.read_csv('final_estimation.csv')
df = df.dropna(subset=['harmonized_fan_percent', 'judge_percent'])

# 2. 计算核心逻辑：进度加权偏差 (Progress-Weighted Bias)
# 计算每季的相对进度 (0.1 - 1.0)，解决赛季长度不一导致的权重不均问题
df['fan_bias'] = df['harmonized_fan_percent'] - df['judge_percent']
max_weeks = df.groupby('season')['week'].transform('max')
df['relative_progress'] = df['week'] / max_weeks

def calc_hero_score(group):
    # 使用进度平方作为权重 (Weights = Progress^2)
    # 理由：选手在后期表现出的“民意韧性”比前期更有统计价值
    weights = group['relative_progress'] ** 2
    return np.average(group['fan_bias'], weights=weights)

# 聚合选手得分并筛选前十名
hero_stats = df.groupby(['celebrity_name', 'season']).apply(lambda x: calc_hero_score(x)).reset_index(name='hero_score')
top_10_heroes = hero_stats.sort_values(by='hero_score', ascending=False).head(10)

# 3. 设置用户定制配色
# 定义提供的色值序列（从浅到深）
user_palette_hex = ["#91CDC8", "#6FB9D0", "#5499BD", "#3981AF", "#386195", "#324C63"]
# 创建颜色映射：排名越高（得分越高）颜色越深
cmap_user = LinearSegmentedColormap.from_list("user_palette", user_palette_hex)
# 获取 10 个颜色，并反转映射以确保最高分(index 0)对应最深色
bar_colors = [cmap_user(val) for val in np.linspace(1, 0, 10)]

# 4. 可视化绘制
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 2, figsize=(22, 10))

# --- A 图: 全局份额对比分布 (原逻辑) ---
sns.scatterplot(
    data=df,
    x='judge_percent',
    y='harmonized_fan_percent',
    hue='is_eliminated',
    palette={True: '#e74c3c', False: '#3498db'},
    alpha=0.5,
    s=120,
    edgecolor='w',
    ax=axes[0]
)

# 绘制 y=x 对角参考线 (平等线)
max_val = max(df['judge_percent'].max(), df['harmonized_fan_percent'].max())
axes[0].plot([0, max_val], [0, max_val], color='#2c3e50', linestyle='--', linewidth=2, alpha=0.7, label='Equality Line')

axes[0].set_title('A. Normalized Score Distribution: Judges vs. Fans\n(Points above dashed line indicate Fan Favorites)',
                 fontsize=18, fontweight='bold', pad=20)
axes[0].set_xlabel('Judge Score Share (Professional Opinion)', fontsize=14)
axes[0].set_ylabel('Inferred Fan Vote Share (Public Sentiment)', fontsize=14)
axes[0].legend(title='Status (Eliminated in that week)')

# --- B 图: 历史十大“民选黑马”排行榜 ---
sns.barplot(
    data=top_10_heroes,
    x='hero_score',
    y='celebrity_name',
    palette=bar_colors,
    ax=axes[1]
)

# 动态添加赛季标签
for i, p in enumerate(axes[1].patches):
    season_val = top_10_heroes.iloc[i]['season']
    axes[1].annotate(f' S{int(season_val)} ',
                    (p.get_width(), p.get_y() + p.get_height()/2),
                    ha='left', va='center', fontsize=12, fontweight='bold', color='#333', xytext=(5, 0),
                    textcoords='offset points')

# 优化文字说明
axes[1].set_title('B. Top 10 "Fan Favorites" by Progress-Weighted Bias\n(Quantifying Public Support Divergence from Judge Expectations)',
                 fontsize=18, fontweight='bold', pad=20)
axes[1].set_xlabel('Weighted Bias Score (Higher = Stronger Late-Season Resistance)', fontsize=14)
axes[1].set_ylabel('', fontsize=14)

plt.tight_layout()
plt.savefig('DWTS_Analysis_Final_AB.png', dpi=300)
plt.show()

print("可视化文件已保存为: DWTS_Analysis_Final_AB.png")