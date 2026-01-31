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
        """核心业务判断：定义不同时代的计分规则"""
        if season <= 2:
            return 'Rank_Standard'
        elif season >= 28:
            return 'Rank_Save'
        else:
            return 'Percentage'

    def parse_final_rank(self, res_str):
        """解析决赛排名及退赛状态"""
        res_str = str(res_str).lower()
        if 'withdrew' in res_str or 'withdrawn' in res_str:
            return 'Withdrew'
        match = re.search(r'(\d+)(?:st|nd|rd|th)\s+place', res_str)
        if match:
            return int(match.group(1))
        return None

    def solve_rank_milp(self, rj, elim_indices, rule_type, final_ranks=None):
        """
        修正版 MILP 求解器：支持平分随机判定
        """
        n = len(rj)
        n_x = n * n
        n_vars = n_x

        k_elim = len(elim_indices)
        use_y = (rule_type == 'Rank_Save' and k_elim > 0 and final_ranks is None)
        if use_y:
            n_vars += k_elim * n

        # 目标函数：最小化与评委排名的距离平方
        c = np.zeros(n_vars)
        for i in range(n):
            for j in range(1, n + 1):
                c[i * n + (j - 1)] = (j - rj[i]) ** 2

        A_rows, b_l, b_u = [], [], []

        # 约束 1: 基础指派约束
        for i in range(n):
            row = np.zeros(n_vars);
            [row.__setitem__(i * n + j - 1, 1) for j in range(1, n + 1)]
            A_rows.append(row);
            b_l.append(1);
            b_u.append(1)
        for j in range(1, n + 1):
            row = np.zeros(n_vars);
            [row.__setitem__(i * n + j - 1, 1) for i in range(n)]
            A_rows.append(row);
            b_l.append(1);
            b_u.append(1)

        # 辅助函数：获取总分 (移除 0.01 * rj 的平分判定权重)
        def get_v_expr(idx):
            expr = np.zeros(n_vars)
            for j in range(1, n + 1):
                expr[idx * n + (j - 1)] = j
            base_val = float(rj[idx])
            return expr, base_val

        # 约束 2: 业务逻辑判断
        if final_ranks is not None:
            # 决赛名次链
            sorted_idx = sorted(range(n), key=lambda x: final_ranks[x])
            for idx in range(len(sorted_idx) - 1):
                h, l = sorted_idx[idx], sorted_idx[idx + 1]
                expr_h, base_h = get_v_expr(h)
                expr_l, base_l_val = get_v_expr(l)
                A_rows.append(expr_h - expr_l)
                # 修改点：允许平分，移除 0.001 偏移量
                b_l.append(-np.inf);
                b_u.append(base_l_val - base_h)

        elif k_elim > 0:
            survivors = [i for i in range(n) if i not in elim_indices]
            if rule_type == 'Rank_Standard':
                # 标准淘汰
                for e in elim_indices:
                    for s in survivors:
                        expr_e, base_e = get_v_expr(e)
                        expr_s, base_s = get_v_expr(s)
                        A_rows.append(expr_s - expr_e)
                        # 修改点：允许平分 (幸存者总分 <= 淘汰者总分)
                        b_l.append(-np.inf);
                        b_u.append(base_e - base_s)

            elif rule_type == 'Rank_Save':
                # S28+ 救人逻辑
                M = n + 10
                for e_idx_in_list, e in enumerate(elim_indices):
                    y_start = n_x + e_idx_in_list * n
                    for i in range(n):
                        if i == e: continue
                        expr_e, base_e = get_v_expr(e)
                        expr_i, base_i = get_v_expr(i)
                        A_rows.append(expr_i - expr_e - M * np.eye(1, n_vars, y_start + i).flatten())
                        # 修改点：允许平分
                        b_l.append(-np.inf);
                        b_u.append(base_e - base_i)

                    row_y = np.zeros(n_vars)
                    for i in range(n):
                        if i != e: row_y[y_start + i] = 1
                    A_rows.append(row_y)
                    b_l.append(n - (k_elim + 1));
                    b_u.append(n - 1)

        # 求解
        res = milp(c, constraints=LinearConstraint(A_rows, b_l, b_u),
                   integrality=np.ones(n_vars), bounds=Bounds(0, 1))

        if res.success:
            v = np.zeros(n)
            for i in range(n):
                for j in range(1, n + 1):
                    if res.x[i * n + (j - 1)] > 0.5: v[i] = j
            return v
        return np.argsort(np.argsort(rj)) + 1

    def solve_percentage(self, pj, elim_indices, final_ranks=None):
        """百分比制求解：允许平分随机判定"""
        n = len(pj)

        def obj(x):
            return np.sum((x - pj) ** 2)

        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 100}]

        # 修改点：移除所有 0.01 * pj 的优先级干扰
        if final_ranks is not None:
            idx = np.argsort(final_ranks)
            for k in range(len(idx) - 1):
                h, l = idx[k], idx[k + 1]
                # 修改点：使用 >= (即偏移量为0)
                cons.append({'type': 'ineq', 'fun': lambda x, h=h, l=l: (pj[h] + x[h]) - (pj[l] + x[l])})
        elif elim_indices:
            survivors = [i for i in range(n) if i not in elim_indices]
            for e in elim_indices:
                for s in survivors:
                    # 修改点：使用 >=
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
        print(f"成功：计算完成（已放宽平分逻辑），结果保存至 {output_file}")


if __name__ == "__main__":
    engine = DWTS_Final_Inference_Engine('cleaned_data.csv')
    engine.execute('predicted_fan_votes_full.csv')