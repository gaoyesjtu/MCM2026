import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds, minimize
import re
import sys


class DWTS_Final_Inference_Engine:
    def __init__(self, input_file):
        try:
            self.df = pd.read_csv(input_file)
        except FileNotFoundError:
            print(f"错误：未找到输入文件 {input_file}")
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
        match = re.search(r'(\d+)(?:st|nd|rd|th)\s+place', res_str)
        if match:
            return int(match.group(1))
        return None

    def solve_rank_milp(self, rj, elim_indices, rule_type, final_ranks=None):
        """
        修正版：正确处理 Rank_Save 模式下的单人淘汰 (Bottom 2) 约束
        """
        n = len(rj)
        n_x = n * n
        k_elim = len(elim_indices)

        # 修正：确保在单人淘汰的 Rank_Save 模式下启用辅助变量 y
        use_y = (rule_type == 'Rank_Save' and k_elim == 1 and final_ranks is None)
        n_vars = n_x + (k_elim * n if use_y else 0)

        c = np.zeros(n_vars)
        for i in range(n):
            for j in range(1, n + 1):
                c[i * n + (j - 1)] = (j - rj[i]) ** 2

        A_rows, b_l, b_u = [], [], []

        # 基础指派约束 (每个人一个排名，每个排名一个人)
        for i in range(n):
            row = np.zeros(n_vars)
            for j in range(1, n + 1): row[i * n + j - 1] = 1
            A_rows.append(row);
            b_l.append(1);
            b_u.append(1)
        for j in range(1, n + 1):
            row = np.zeros(n_vars)
            for i in range(n): row[i * n + j - 1] = 1
            A_rows.append(row);
            b_l.append(1);
            b_u.append(1)

        def get_v_expr(idx):
            expr = np.zeros(n_vars)
            for j in range(1, n + 1): expr[idx * n + (j - 1)] = j
            return expr, float(rj[idx])

        # --- 业务逻辑约束 ---
        if final_ranks is not None:
            # 决赛排名已知逻辑
            sorted_idx = sorted(range(n), key=lambda x: final_ranks[x])
            for idx in range(len(sorted_idx) - 1):
                h, l = sorted_idx[idx], sorted_idx[idx + 1]
                expr_h, base_h = get_v_expr(h)
                expr_l, base_l_val = get_v_expr(l)
                A_rows.append(expr_h - expr_l)
                b_l.append(-np.inf);
                b_u.append(base_l_val - base_h)

        elif k_elim > 0:
            survivors = [i for i in range(n) if i not in elim_indices]

            # 情况 A: 标准赛制 或 多人淘汰 (认为总分表现最差的全部出局)
            if rule_type == 'Rank_Standard' or (rule_type == 'Rank_Save' and k_elim >= 2):
                for e in elim_indices:
                    for s in survivors:
                        expr_e, base_e = get_v_expr(e)
                        expr_s, base_s = get_v_expr(s)
                        A_rows.append(expr_s - expr_e)
                        b_l.append(-np.inf);
                        b_u.append(base_e - base_s)

            # 情况 B: 现代赛制单人淘汰 (修正的核心点：Bottom 2 逻辑)
            elif rule_type == 'Rank_Save' and k_elim == 1:
                M = n + 10  # 大常数
                for e_idx, e in enumerate(elim_indices):
                    y_start = n_x + e_idx * n
                    for i in range(n):
                        if i == e: continue
                        expr_e, base_e = get_v_expr(e)
                        expr_i, base_i = get_v_expr(i)
                        # v_i + rj_i - (v_e + rj_e) <= M * y_i
                        # 即：v_i - v_e - M * y_i <= rj_e - rj_i
                        row = expr_i - expr_e
                        row[y_start + i] = -M
                        A_rows.append(row)
                        b_l.append(-np.inf);
                        b_u.append(base_e - base_i)

                    # 统计比淘汰者表现更差（总分数值更大）的人数不能超过1人 (即淘汰者在 Bottom 2)
                    row_y = np.zeros(n_vars)
                    for i in range(n):
                        if i != e: row_y[y_start + i] = 1
                    A_rows.append(row_y)
                    b_l.append(0);
                    b_u.append(1)

        # 求解 MILP
        res = milp(c, constraints=LinearConstraint(A_rows, b_l, b_u),
                   integrality=np.ones(n_vars), bounds=Bounds(0, 1))

        if res.success:
            v = np.zeros(n)
            for i in range(n):
                for j in range(1, n + 1):
                    if res.x[i * n + (j - 1)] > 0.5: v[i] = j
            return v

        # --- 随机搜索保底 ---
        #print(1)
        for _ in range(2000):
            v_guess = np.random.permutation(np.arange(1, n + 1))
            total = rj + v_guess
            is_valid = True

            if final_ranks is not None:
                s_idx = np.argsort(final_ranks)
                for k in range(len(s_idx) - 1):
                    if total[s_idx[k]] > total[s_idx[k + 1]]:
                        is_valid = False;
                        break
            elif k_elim > 0:
                if rule_type == 'Rank_Standard' or (rule_type == 'Rank_Save' and k_elim >= 2):
                    survivors = [i for i in range(n) if i not in elim_indices]
                    if any(total[s] > total[e] for s in survivors for e in elim_indices):
                        is_valid = False
                else:  # S28+ 单人淘汰
                    sorted_t = sorted(total)
                    limit = sorted_t[-2] if n >= 2 else sorted_t[-1]
                    if all(total[e] < limit for e in elim_indices):
                        is_valid = False

            if is_valid: return v_guess

        return np.argsort(np.argsort(rj)) + 1

    def solve_percentage(self, pj, elim_indices, final_ranks=None):
        n = len(pj)

        def obj(x):
            return np.sum((x - pj) ** 2)

        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 100}]
        if final_ranks is not None:
            idx = np.argsort(final_ranks)
            for k in range(len(idx) - 1):
                h, l = idx[k], idx[k + 1]
                cons.append({'type': 'ineq', 'fun': lambda x, h=h, l=l: (pj[h] + x[h]) - (pj[l] + x[l])})
        elif elim_indices:
            survivors = [i for i in range(n) if i not in elim_indices]
            for e in elim_indices:
                for s in survivors:
                    cons.append({'type': 'ineq', 'fun': lambda x, e=e, s=s: (pj[s] + x[s]) - (pj[e] + x[e])})

        res = minimize(obj, pj, method='SLSQP', bounds=[(0, 100)] * n, constraints=cons)
        return res.x if res.success else pj

    def execute(self, output_file):
        results = []
        for (s, w), group in self.df.groupby(['season', 'week']):
            rule = self.get_rule_type(s)
            group = group.copy()
            group['f_placement'] = group['results'].apply(self.parse_final_rank)
            active = group[group['f_placement'] != 'Withdrew'].copy()
            if active.empty: continue

            names = active['celebrity_name'].tolist()
            rj = active['judge_rank'].values
            pj = active['judge_percent'].values * 100
            elim = np.where(active['is_eliminated'])[0].tolist()
            is_last = (w == self.df[self.df['season'] == s]['week'].max())
            f_ranks = active['f_placement'].values if is_last else None

            if rule == 'Percentage':
                votes = self.solve_percentage(pj, elim, f_ranks)
            else:
                votes = self.solve_rank_milp(rj, elim, rule, f_ranks)

            vote_map = dict(zip(names, votes))
            for _, row in group.iterrows():
                row_data = row.to_dict()
                row_data['rule_type'] = rule
                row_data['predicted_fan_vote'] = vote_map.get(row['celebrity_name'], None)
                results.append(row_data)

        pd.DataFrame(results).to_csv(output_file, index=False)
        print(f"成功：修正版已启用。结果保存至 {output_file}")


if __name__ == "__main__":
    engine = DWTS_Final_Inference_Engine('cleaned_data.csv')
    engine.execute('final_estimation_rank.csv')