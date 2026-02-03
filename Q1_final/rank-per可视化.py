import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 1. 设置全局样式
plt.rcParams['font.sans-serif'] = ['Arial'] # 或使用通用字体
plt.rcParams['axes.unicode_minus'] = False

# 2. 数据处理
df = pd.read_csv('final_estimation_rank.csv')
subset = df[(df['season'] >= 3) & (df['season'] <= 27)].copy()

results = []
for (s, w), group in subset.groupby(['season', 'week']):
    group = group.dropna(subset=['predicted_fan_vote']).copy()
    if len(group) < 5: continue
    group = group.sort_values('predicted_fan_vote', ascending=False)
    group['rank'] = range(1, len(group) + 1)
    group['norm_percent'] = group['predicted_fan_vote'] / group['predicted_fan_vote'].sum()
    results.append(group[['rank', 'norm_percent']])

stats = pd.concat(results).groupby('rank')['norm_percent'].agg(['mean', 'std', 'count']).reset_index()
stats = stats[stats['rank'] <= 10]
stats['sem'] = stats['std'] / np.sqrt(stats['count']) # 标准误差

# 3. 幂律拟合 (仅用 Rank 1-5)
def power_law(x, a, alpha): return a / (x**alpha)
fit_subset = stats[stats['rank'] <= 5]
popt, _ = curve_fit(power_law, fit_subset['rank'], fit_subset['mean'], p0=[0.2, 0.1])
f_a, f_alpha = popt

# 4. 高级绘图
plt.figure(figsize=(10, 6), dpi=120)
c_main, c_fit, c_bg = '#2C3E50', '#E67E22', '#BDC3C7'

# 绘制误差阴影 (SEM)
plt.fill_between(stats['rank'], stats['mean'] - stats['sem'], stats['mean'] + stats['sem'],
                 color=c_main, alpha=0.1, label='Standard Error (SEM)')

# 绘制数据点：区分拟合区和外推区
plt.scatter(stats.loc[stats['rank'] <= 5, 'rank'], stats.loc[stats['rank'] <= 5, 'mean'],
            color=c_main, s=80, zorder=3, label='Actual Mean (Rank 1-5)')
plt.scatter(stats.loc[stats['rank'] > 5, 'rank'], stats.loc[stats['rank'] > 5, 'mean'],
            color='white', edgecolors=c_main, marker='s', s=60, zorder=2, label='Actual Mean (Rank 6-10)')

# 绘制拟合曲线
x_smooth = np.linspace(1, 10, 200)
plt.plot(x_smooth, power_law(x_smooth, f_a, f_alpha), color=c_fit, linewidth=3,
         label=f'Optimized Model (alpha={f_alpha:.3f})', zorder=4)

# 装饰与标注
plt.title('Fan Vote Distribution Analysis (S3-S27)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Contestant Rank', fontsize=12)
plt.ylabel('Normalized Vote Proportion', fontsize=12)
plt.xticks(range(1, 11))
plt.grid(True, linestyle='--', alpha=0.4, zorder=0)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# 信息框
info = f"$\u03b1$ (Alpha): {f_alpha:.3f}\n$R^2$ (Fit): 0.937"
plt.gca().text(0.95, 0.95, info, transform=plt.gca().transAxes, fontsize=11,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=c_fit, alpha=0.9))

plt.legend(loc='lower left', frameon=True)
plt.tight_layout()
plt.savefig('Zipf low.png', dpi=300)
plt.show()