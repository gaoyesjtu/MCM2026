import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. 设置文件路径
file_estimation = 'Q2_new/final_estimation.csv'
file_comparison = 'Q2_new/rank_percent_comparison.csv'
file_discrepancies = 'Q2_new/rank_percent_discrepancies.csv'

# 2. 加载数据
df_est = pd.read_csv(file_estimation)
df_comp = pd.read_csv(file_comparison)
df_disc = pd.read_csv(file_discrepancies)

# 确保输出目录存在（如果需要）
# os.makedirs('plots', exist_ok=True)

# 3. 数据预处理：获取分歧周次中被淘汰选手的评委得分
def get_judge_score(row, col_name):
    name = row[col_name]
    s, w = row['season'], row['week']
    # 从原始估计文件中匹配该选手在该周的总评委得分
    score_series = df_est[(df_est['season'] == s) & 
                          (df_est['week'] == w) & 
                          (df_est['celebrity_name'] == name)]['total_judge_score']
    return score_series.values[0] if not score_series.empty else 0

# 计算分歧周次中两种方法淘汰选手的技术分
df_disc['rank_js'] = df_disc.apply(lambda r: get_judge_score(r, 'rank_elim'), axis=1)
df_disc['perc_js'] = df_disc.apply(lambda r: get_judge_score(r, 'perc_elim'), axis=1)

# --- 图表 1: 与纯粉丝投票意愿的一致性对比 ---
fan_match_data = pd.DataFrame({
    'Method': ['Rank Method', 'Percentage Method'],
    'Agreement Count': [
        (df_disc['rank_elim'] == df_disc['fan_elim']).sum(),
        (df_disc['perc_elim'] == df_disc['fan_elim']).sum()
    ]
})

plt.figure(figsize=(8, 6))
sns.set_style("whitegrid")
ax1 = sns.barplot(x='Method', y='Agreement Count', data=fan_match_data, palette='coolwarm')
plt.title('Agreement with Pure Fan Vote in Discrepant Weeks (N=100)', fontsize=14)
plt.ylabel('Number of Matches', fontsize=12)
# 添加数值标签
for p in ax1.patches:
    ax1.annotate(format(p.get_height(), '.0f'), 
                 (p.get_x() + p.get_width() / 2., p.get_height()), 
                 ha = 'center', va = 'center', xytext = (0, 9), 
                 textcoords = 'offset points', fontsize=11)
plt.savefig('fan_agreement_plot.png', dpi=300)
plt.show()

# --- 图表 2: 被淘汰选手的平均技术得分对比 ---
avg_scores = pd.DataFrame({
    'Method': ['Rank Method', 'Percentage Method'],
    'Avg Judge Score': [df_disc['rank_js'].mean(), df_disc['perc_js'].mean()]
})

plt.figure(figsize=(8, 6))
ax2 = sns.barplot(x='Method', y='Avg Judge Score', data=avg_scores, palette='viridis')
plt.title('Average Technical Score (JS) of Eliminated Contestants', fontsize=14)
plt.ylabel('Mean Total Judge Score', fontsize=12)
plt.ylim(0, 30)
for p in ax2.patches:
    ax2.annotate(format(p.get_height(), '.2f'), 
                 (p.get_x() + p.get_width() / 2., p.get_height()), 
                 ha = 'center', va = 'center', xytext = (0, 9), 
                 textcoords = 'offset points', fontsize=11)
plt.savefig('avg_js_comparison_plot.png', dpi=300)
plt.show()

# --- 图表 3: 被淘汰选手技术得分的分布情况 (Boxplot) ---
# 转换为长格式以便绘图
df_melted = pd.melt(df_disc[['rank_js', 'perc_js']], var_name='Method', value_name='Judge Score')
df_melted['Method'] = df_melted['Method'].map({'rank_js': 'Rank Method', 'perc_js': 'Percentage Method'})

plt.figure(figsize=(8, 6))
sns.boxplot(x='Method', y='Judge Score', data=df_melted, palette='Set2')
plt.title('Distribution of Technical Scores for Eliminated Candidates', fontsize=14)
plt.ylabel('Total Judge Score', fontsize=12)
plt.savefig('js_distribution_plot.png', dpi=300)
plt.show()

print("Successfully saved!")