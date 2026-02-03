import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. 定义并设置您要求的 RGB 色卡 ---
# RGB 145,205,200、 RGB 111, 185, 208 、 RGB 84,153,189 、 RGB 57,129,175
COLORS_RGB = [
    (145/255, 205/255, 200/255), 
    (111/255, 185/255, 208/255), 
    (84/255, 153/255, 189/255), 
    (57/255, 129/255, 175/255)
]

# 设置全局绘图风格
sns.set_style("whitegrid")
# 定义一个调色板，后续会自动按顺序取用
MY_PALETTE = sns.color_palette(COLORS_RGB)

# --- 2. 设置文件路径 ---
file_estimation = 'Q2_new/final_estimation.csv'
file_comparison = 'Q2_new/rank_percent_comparison.csv'
file_discrepancies = 'Q2_new/rank_percent_discrepancies.csv'

# 加载数据
try:
    df_est = pd.read_csv(file_estimation)
    df_comp = pd.read_csv(file_comparison)
    df_disc = pd.read_csv(file_discrepancies)
except FileNotFoundError as e:
    print(f"错误：未找到数据文件 ({e.filename})，请确保 Q2_new 文件夹及文件存在。")
    exit()

# 3. 数据预处理
def get_judge_score(row, col_name):
    name = row[col_name]
    s, w = row['season'], row['week']
    score_series = df_est[(df_est['season'] == s) & 
                          (df_est['week'] == w) & 
                          (df_est['celebrity_name'] == name)]['total_judge_score']
    return score_series.values[0] if not score_series.empty else 0

df_disc['rank_js'] = df_disc.apply(lambda r: get_judge_score(r, 'rank_elim'), axis=1)
df_disc['perc_js'] = df_disc.apply(lambda r: get_judge_score(r, 'perc_elim'), axis=1)

# --- 图表 1: 与纯粉丝投票意愿的一致性对比 ---
fan_match_data = pd.DataFrame({
    'Method': ['Rank Combination', 'Percentage Combination'],
    'Agreement Count': [
        (df_disc['rank_elim'] == df_disc['fan_elim']).sum(),
        (df_disc['perc_elim'] == df_disc['fan_elim']).sum()
    ]
})

plt.figure(figsize=(8, 6))
# 使用您要求的颜色，并通过 hue 消除警告
ax1 = sns.barplot(x='Method', y='Agreement Count', data=fan_match_data, 
                  hue='Method', palette=MY_PALETTE, legend=False)
plt.title('Agreement with Fan Votes Only in Discrepant Weeks', fontsize=14)
plt.ylabel('Number of Matches', fontsize=12)

for p in ax1.patches:
    ax1.annotate(format(p.get_height(), '.0f'), 
                 (p.get_x() + p.get_width() / 2., p.get_height()), 
                 ha = 'center', va = 'center', xytext = (0, 9), 
                 textcoords = 'offset points', fontsize=11)
plt.savefig('fan_agreement_plot.png', dpi=300)
plt.close()

# --- 图表 2: 被淘汰选手的平均技术得分对比 ---
avg_scores = pd.DataFrame({
    'Method': ['Rank Combination', 'Percentage Combination'],
    'Avg Judge Score': [df_disc['rank_js'].mean(), df_disc['perc_js'].mean()]
})

plt.figure(figsize=(8, 6))
ax2 = sns.barplot(x='Method', y='Avg Judge Score', data=avg_scores, 
                  hue='Method', palette=MY_PALETTE, legend=False)
plt.title('Average Judge Score  of Eliminated Contestants', fontsize=14)
plt.ylabel('Mean Judge Score', fontsize=12)
plt.ylim(0, 30)

for p in ax2.patches:
    ax2.annotate(format(p.get_height(), '.2f'), 
                 (p.get_x() + p.get_width() / 2., p.get_height()), 
                 ha = 'center', va = 'center', xytext = (0, 9), 
                 textcoords = 'offset points', fontsize=11)
plt.savefig('avg_js_comparison_plot.png', dpi=300)
plt.close()

# --- 图表 3: 被淘汰选手技术得分的分布情况 (Boxplot) ---
df_melted = pd.melt(df_disc[['rank_js', 'perc_js']], var_name='Method', value_name='Judge Score')
df_melted['Method'] = df_melted['Method'].map({'rank_js': 'Rank Method', 'perc_js': 'Percentage Method'})

plt.figure(figsize=(8, 6))
sns.boxplot(x='Method', y='Judge Score', data=df_melted, 
            hue='Method', palette=MY_PALETTE, legend=False)
plt.title('Distribution of Judge Scores for Eliminated Candidates', fontsize=14)
plt.ylabel('Total Judge Score', fontsize=12)
plt.savefig('js_distribution_plot.png', dpi=300)
plt.close()

print("Successfully saved all plots with your custom RGB color card!")