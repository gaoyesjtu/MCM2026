import pandas as pd
import numpy as np
import re
import itertools
import math
from tqdm import tqdm


class DWTS_MAD_Certainty_Engine:
    def __init__(self, input_file):
        try:
            # 输入文件需包含预测出的 'predicted_fan_vote' 列
            self.df = pd.read_csv(input_file)
        except FileNotFoundError:
            print(f"错误：未找到文件 {input_file}")
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
        if 'withdrew' in res_str or 'withdrawn' in res_str: return 'Withdrew'
        match = re.search(r'(\d+)', res_str)
        return int(match.group(1)) if match else None

    # --- 逻辑判定器：严格保留 Q1 核心规则 ---
    def is_valid_rank(self, total_scores, elim_indices, rule_type, f_ranks):
        n = len(total_scores)
        if f_ranks is not None:
            idx = np.argsort(f_ranks)
            for i in range(len(idx) - 1):
                if total_scores[idx[i]] > total_scores[idx[i + 1]]: return False
            return True
        if not elim_indices: return True
        survivors = [i for i in range(n) if i not in elim_indices]
        if rule_type == 'Rank_Standard' or len(elim_indices) >= 2:
            for s in survivors:
                for e in elim_indices:
                    if total_scores[s] > total_scores[e]: return False
        elif rule_type == 'Rank_Save':
            e = elim_indices[0]
            worse_than_e = sum(1 for i in range(n) if total_scores[i] > total_scores[e])
            return worse_than_e <= 1
        return True

    def is_valid_percent(self, total_scores, elim_indices, f_ranks):
        n = len(total_scores)
        if f_ranks is not None:
            idx = np.argsort(f_ranks)
            for i in range(len(idx) - 1):
                if total_scores[idx[i]] < total_scores[idx[i + 1]]: return False
            return True
        if not elim_indices: return True
        survivors = [i for i in range(n) if i not in elim_indices]
        for s in survivors:
            for e in elim_indices:
                if total_scores[s] < total_scores[e]: return False
        return True

    def analyze_week_refined(self, rj, pj, elim_indices, rule_type, f_ranks, preds):
        n = len(rj)
        samples = 100000
        valid_samples = []

        if rule_type != 'Percentage':
            # --- Rank 模式：保留命中频次逻辑 ---
            if n <= 9:
                for p in itertools.permutations(range(1, n + 1)):
                    p_arr = np.array(p)
                    if self.is_valid_rank(rj + p_arr, elim_indices, rule_type, f_ranks):
                        valid_samples.append(p_arr)
            else:
                for _ in range(samples):
                    p_arr = np.random.permutation(np.arange(1, n + 1))
                    if self.is_valid_rank(rj + p_arr, elim_indices, rule_type, f_ranks):
                        valid_samples.append(p_arr)

            if not valid_samples: return [0.0] * n, 1.0
            valid_samples = np.array(valid_samples)
            m_valid = len(valid_samples)
            week_cert = 1.0 - (m_valid / (math.factorial(n) if n <= 9 else samples))
            # 排名模式下的确定性 (Hit Rate)
            indiv_certs = [np.mean(np.isclose(valid_samples[:, i], preds[i], atol=0.1)) for i in range(n)]

        else:
            # --- Percentage 模式：直接使用 MAD (平均绝对偏差) ---
            v_pool = np.random.dirichlet([1.0] * n, samples) * 100
            for i in range(samples):
                if self.is_valid_percent(pj + v_pool[i], elim_indices, f_ranks):
                    valid_samples.append(v_pool[i])

            if not valid_samples: return [99.0] * n, 1.0  # 逻辑冲突时赋予极大偏差

            valid_samples = np.array(valid_samples)
            m_valid = len(valid_samples)
            week_cert = 1.0 - (m_valid / samples)

            indiv_certs = []
            for i in range(n):
                obs_vals = valid_samples[:, i]
                p_val = preds[i]

                # 计算 MAD: 平均绝对偏离程度
                # 这种度量方式直接反映了“受制约的剧本”离你的“预测点”有多远
                mad = np.mean(np.abs(obs_vals - p_val))
                indiv_certs.append(mad)

        return indiv_certs, week_cert

    def execute(self, rank_out, percent_out):
        rank_res, percent_res = [], []
        groups = list(self.df.groupby(['season', 'week']))

        print("开始基于 MAD 实验度量预测确定性...")
        for (s, w), group in tqdm(groups):
            rule = self.get_rule_type(s)
            group = group.copy()
            group['f_placement'] = group['results'].apply(self.parse_final_rank)
            active = group[group['f_placement'] != 'Withdrew'].copy()
            if active.empty: continue

            rj, pj = active['judge_rank'].values, active['judge_percent'].values * 100
            elim = np.where(active['is_eliminated'])[0].tolist()
            is_last = (w == self.df[self.df['season'] == s]['week'].max())
            f_ranks = active['f_placement'].values if is_last else None
            preds = active['predicted_fan_vote'].values

            indiv_certs, week_cert = self.analyze_week_refined(rj, pj, elim, rule, f_ranks, preds)

            active['certainty_metric'] = indiv_certs
            active['week_certainty'] = week_cert

            if rule == 'Percentage':
                percent_res.append(active)
            else:
                rank_res.append(active)

        if rank_res: pd.concat(rank_res).to_csv(rank_out, index=False)
        if percent_res: pd.concat(percent_res).to_csv(percent_out, index=False)
        print(f"\n分析完成！")


if __name__ == "__main__":
    # 输入应为 Q1 生成的包含 predicted_fan_vote 的文件
    engine = DWTS_MAD_Certainty_Engine('final_estimation_rank.csv')
    engine.execute('final_percent_certainty.csv', 'final_rank_certainty.csv')