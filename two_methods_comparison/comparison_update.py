import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import os
import sys

# 设置绘图字体（解决中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class DWTS_Analytics_Engine:
    def __init__(self, fan_file, clean_file):
        self.fan_file = fan_file
        self.clean_file = clean_file
        self.df = None

    def load_and_merge(self):
        """加载数据并处理潜在的列名和合并冲突问题"""
        if not os.path.exists(self.fan_file) or not os.path.exists(self.clean_file):
            print(f"错误：找不到文件。请检查路径：\n1. {self.fan_file}\n2. {self.clean_file}")
            sys.exit(1)

        try:
            # 加载数据
            df_fan = pd.read_csv(self.fan_file)
            df_clean = pd.read_csv(self.clean_file)

            # 统一清理列名：去除多余空格
            df_fan.columns = df_fan.columns.str.strip()
            df_clean.columns = df_clean.columns.str.strip()

            # --- 关键修改点：避免后缀冲突 ---
            # 粉丝预测表中，我们只需要它的预测结果列，其他信息（如评委分）以 clean_data 为准
            fan_cols_to_keep = ['season', 'week', 'celebrity_name', 'inferred_fan_val', 'rule_type',
                                'harmonized_fan_percent']
            available_fan_cols = [c for c in fan_cols_to_keep if c in df_fan.columns]

            # 只用标识符列进行合并，确保 judge_rank 不会被重命名为 judge_rank_x/y
            self.df = pd.merge(df_clean, df_fan[available_fan_cols], on=['season', 'week', 'celebrity_name'])

            # --- 关键修改点：列名容错处理 ---
            col_map = {
                'Judge Rank': 'judge_rank',
                'Judge Percent': 'judge_percent',
                'judge_pct': 'judge_percent'
            }
            for old_col, new_col in col_map.items():
                if old_col in self.df.columns and new_col not in self.df.columns:
                    self.df.rename(columns={old_col: new_col}, inplace=True)

            print(f"-> 成功关联。可用列: {self.df.columns.tolist()}")

            if 'judge_rank' not in self.df.columns:
                # 最后的保险：如果还没找到，模糊匹配包含 rank 的列
                rank_cols = [c for c in self.df.columns if 'rank' in c.lower() and 'fan' not in c.lower()]
                if rank_cols:
                    self.df.rename(columns={rank_cols[0]: 'judge_rank'}, inplace=True)

            # 计算粉丝排名
            target_fan_col = 'harmonized_fan_percent' if 'harmonized_fan_percent' in self.df.columns else 'inferred_fan_val'
            self.df['fan_rank'] = self.df.groupby(['season', 'week'])[target_fan_col].rank(ascending=False,
                                                                                           method='min')

        except Exception as e:
            print(f"数据加载或合并时发生错误: {e}")
            sys.exit(1)

    def run_comparison(self):
        """执行赛制对比模拟"""
        analysis_data = []
        for (s, w), group in self.df.groupby(['season', 'week']):
            group = group.copy()
            if len(group) < 2: continue

            # 1. 排名法 (Rank-based)
            if 'judge_rank' in group.columns:
                group['rank_score'] = group['judge_rank'] + group['fan_rank']
                # 平分处理：粉丝更喜欢的排在前面
                group['res_rank_method'] = (group['rank_score'] + 0.01 * group['fan_rank']).rank(ascending=True)

            # 2. 百分比法 (Percentage-based)
            j_pct_col = 'judge_percent' if 'judge_percent' in group.columns else None
            if j_pct_col and 'harmonized_fan_percent' in group.columns:
                group['pct_score'] = group[j_pct_col] + group['harmonized_fan_percent']
                group['res_pct_method'] = (group['pct_score'] + 0.0001 * group['harmonized_fan_percent']).rank(
                    ascending=False)

            analysis_data.append(group)

        return pd.concat(analysis_data)

    def generate_report(self, results_df):
        """生成可视化图表与文字总结"""
        stats = []
        # 只比较两种方法都模拟成功的记录
        valid_df = results_df.dropna(subset=['res_rank_method', 'res_pct_method'])

        for (s, w), g in valid_df.groupby(['season', 'week']):
            # 寻找淘汰者 (数值最大者)
            elim_rank = g.loc[g['res_rank_method'].idxmax(), 'celebrity_name']
            elim_pct = g.loc[g['res_pct_method'].idxmax(), 'celebrity_name']

            # 计算相关性 (Spearman)
            c_rank = spearmanr(g['res_rank_method'], g['fan_rank'])[0]
            c_pct = spearmanr(g['res_pct_method'], g['fan_rank'])[0]

            stats.append({'mismatch': elim_rank != elim_pct, 'corr_rank': c_rank, 'corr_pct': c_pct})

        stats_df = pd.DataFrame(stats)

        # 可视化
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # A. 一致性
        m_rate = stats_df['mismatch'].mean() * 100
        sns.barplot(x=['结果一致', '结果分歧'], y=[100 - m_rate, m_rate], palette='coolwarm', ax=axes[0])
        axes[0].set_title(f'赛制结果一致性 (分歧率: {m_rate:.1f}%)')
        axes[0].set_ylabel('占比 (%)')

        # B. 权力分布
        sns.boxplot(data=stats_df[['corr_rank', 'corr_pct']], palette='Set2', ax=axes[1])
        axes[1].set_xticklabels(['排名法', '百分比法'])
        axes[1].set_title('机制与粉丝投票的相关性 (越高代表粉丝话语权越大)')
        axes[1].set_ylabel('相关系数')

        plt.tight_layout()
        plt.savefig('mechanism_comparison_final.png')

        print("\n" + "=" * 40)
        print("计分机制对比深度报告")
        print("=" * 40)
        print(f"赛制冲突率: {m_rate:.2f}%")
        print(f"排名法粉丝相关性中位数: {stats_df['corr_rank'].median():.4f}")
        print(f"百分比法粉丝相关性中位数: {stats_df['corr_pct'].median():.4f}")
        print(
            f"结论: {'排名法' if stats_df['corr_rank'].median() > stats_df['corr_pct'].median() else '百分比法'} 更尊重粉丝意愿。")
        print("=" * 40)


if __name__ == "__main__":
    # 使用用户提供的绝对路径
    INPUT_FAN = r'D:\pycharm_codes\MCM2026\Q1_fixed\final_fan_estimate.csv'
    INPUT_CLEAN = r'D:\pycharm_codes\MCM2026\Q1_fixed\cleaned_data.csv'

    engine = DWTS_Analytics_Engine(INPUT_FAN, INPUT_CLEAN)
    engine.load_and_merge()
    full_results = engine.run_comparison()
    engine.generate_report(full_results)

    # 导出模拟结果方便本地查看
    full_results.to_csv('comparison_results_detailed.csv', index=False)