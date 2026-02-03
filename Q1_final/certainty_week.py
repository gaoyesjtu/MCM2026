import pandas as pd
import numpy as np
import re
import itertools
import math
from tqdm import tqdm  # 用于显示进度


class DWTS_Certainty_Analysis_Engine:
    def __init__(self, input_file):
        try:
            self.df = pd.read_csv(input_file)
        except FileNotFoundError:
            print(f"错误：未找到输入文件 {input_file}")
            import sys
            sys.exit(1)

    def get_rule_type(self, season):
        if season <= 2:
            return 'Rank_Standard'
        elif season >= 28:
            return 'Rank_Save'
        else:
            return 'Percentage'

    def parse_final_rank(self, res_str):
        res_str = str(res_str).lower()
        if 'withdrew' in res_str or 'withdrawn' in res_str:
            return 'Withdrew'
        match = re.search(r'(\d+)', res_str)
        if match:
            return int(match.group(1))
        return None

    def is_valid_logic(self, total_scores, elim_indices, rule_type, f_ranks=None):
        """
        核心判定逻辑：完全沿用 Q1 的淘汰与决赛准则
        total_scores: 评委分 + 预测观众分 (数值越小排名越高/表现越好)
        """
        n = len(total_scores)

        # 1. 决赛逻辑：如果最终名次已知，总分必须严格符合名次顺序
        if f_ranks is not None:
            # 按名次升序排列索引 (1st, 2nd, 3rd...)
            sorted_idx = np.argsort(f_ranks)
            for i in range(len(sorted_idx) - 1):
                # 前一名的总分分值 必须小于等于 后一名的总分分值
                if total_scores[sorted_idx[i]] > total_scores[sorted_idx[i + 1]]:
                    return False
            return True

        # 2. 淘汰逻辑：如果没有淘汰发生，该排列视为有效
        if not elim_indices:
            return True

        survivors = [i for i in range(n) if i not in elim_indices]
        k_elim = len(elim_indices)

        # 情况 A: 标准赛制 或 多人淘汰 (所有幸存者优于淘汰者)
        if rule_type == 'Rank_Standard' or (rule_type == 'Rank_Save' and k_elim >= 2):
            for s in survivors:
                for e in elim_indices:
                    if total_scores[s] > total_scores[e]:
                        return False
            return True

        # 情况 B: 现代赛制单人淘汰 (Bottom 2 逻辑)
        elif rule_type == 'Rank_Save' and k_elim == 1:
            e = elim_indices[0]
            # 统计有多少人总分比淘汰者更差 (总分数值更大)
            # 淘汰者必须在 Bottom 2，意味着比他差的人数 <= 1 (0人代表他是倒数第一，1人代表他是倒数第二)
            worse_than_e = sum(1 for i in range(n) if total_scores[i] > total_scores[e])
            return worse_than_e <= 1

        return True

    def calculate_week_certainty(self, rj, pj, elim_indices, rule_type, f_ranks=None):
        """
        计算解空间的收缩比 (Contraction Ratio)
        """
        n = len(rj)
        samples = 100000  # 采样数

        if rule_type != 'Percentage':
            # --- Rank 模式 ---
            if n <= 9:
                # 全遍历：绝对准确
                total_perms = math.factorial(n)
                valid_count = 0
                for p in itertools.permutations(range(1, n + 1)):
                    if self.is_valid_logic(rj + np.array(p), elim_indices, rule_type, f_ranks):
                        valid_count += 1
                ratio = valid_count / total_perms
            else:
                # 大规模蒙特卡洛采样
                valid_count = 0
                for _ in range(samples):
                    v = np.random.permutation(np.arange(1, n + 1))
                    if self.is_valid_logic(rj + v, elim_indices, rule_type, f_ranks):
                        valid_count += 1
                ratio = valid_count / samples
        else:
            # --- Percentage 模式 ---
            # 使用 Dirichlet 分布在单纯形上均匀采样体积
            valid_count = 0
            for _ in range(samples):
                # 生成总和为 100 的随机观众得票百分比
                v_percent = np.random.dirichlet([1.0] * n) * 100
                # 判定 (Percentage 模式下 total = 评委% + 观众%)
                # 注意：Percentage 模式下数值越大表现越好，所以逻辑需反向或调整判定器
                total_percentage = pj + v_percent
                # 为复用逻辑，这里简单处理：将百分比转为负值(越大越好 变 越小越好)
                if self.is_valid_logic(-total_percentage, elim_indices, 'Rank_Standard', f_ranks):
                    valid_count += 1
            ratio = valid_count / samples

        # 确定性 = 1 - 可行空间占比
        # ratio 越小，说明限制越死，Certainty 越高
        return 1.0 - ratio

    def execute(self, output_file):
        results = []
        # 按赛季和周分组
        groups = list(self.df.groupby(['season', 'week']))

        print("正在分析解空间收缩比 (Certainty)...")
        for (s, w), group in tqdm(groups):
            rule = self.get_rule_type(s)
            group = group.copy()
            group['f_placement'] = group['results'].apply(self.parse_final_rank)

            # 过滤退赛者
            active = group[group['f_placement'] != 'Withdrew'].copy()
            if active.empty:
                continue

            rj = active['judge_rank'].values
            pj = active['judge_percent'].values * 100
            elim = np.where(active['is_eliminated'])[0].tolist()

            # 判断是否为该季最后一周 (决赛)
            is_last = (w == self.df[self.df['season'] == s]['week'].max())
            f_ranks = active['f_placement'].values if is_last else None

            # 计算确定性
            cert = self.calculate_week_certainty(rj, pj, elim, rule, f_ranks)

            # 将结果写回
            for _, row in group.iterrows():
                row_data = row.to_dict()
                row_data['rule_type'] = rule
                row_data['space_contraction_certainty'] = cert
                row_data['feasible_ratio'] = 1.0 - cert  # 辅助参考：可行解占总空间的比例
                results.append(row_data)

        output_df = pd.DataFrame(results)
        output_df.to_csv(output_file, index=False)
        print(f"\n分析完成！结果已保存至: {output_file}")


if __name__ == "__main__":
    # 请确保目录下有 cleaned_data.csv
    engine = DWTS_Certainty_Analysis_Engine('cleaned_data.csv')
    engine.execute('certainty_week.csv')