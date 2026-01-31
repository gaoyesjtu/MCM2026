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
        修正版 MILP 求解器：支持多人淘汰与平局判定
        """
        n = len(rj)
        n_x = n * n
        n_vars = n_x

        # 确定需要标记“谁比淘汰者强”的辅助变量 y (用于 S28+ Save 逻辑)
        # 如果是 k 人淘汰，我们需要保证淘汰者处于 Bottom k+1
        k_elim = len(elim_indices)
        use_y = (rule_type == 'Rank_Save' and k_elim > 0 and final_ranks is None)
        if use_y:
            # 为每一对 (淘汰者 e, 其他人 i) 建立关系
            # y_{e,i} = 1 表示 i 的表现优于 e
            n_vars += k_elim * n

        # 目标函数：最小化与评委排名的距离平方
        c = np.zeros(n_vars)
        for i in range(n):
            for j in range(1, n + 1):
                c[i * n + (j - 1)] = (j - rj[i]) ** 2

        A_rows, b_l, b_u = [], [], []

        # 约束 1: 基础指派约束 (每人一个排名，每个排名一位选手)
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

        # 辅助函数：获取虚拟总分 (包含平局判定逻辑)
        # V = (rj + rf) + 0.01 * rj。V 越大越差。
        def get_v_expr(idx):
            expr = np.zeros(n_vars)
            for j in range(1, n + 1):
                expr[idx * n + (j - 1)] = j
            base_val = rj[idx] + 0.01 * rj[idx]
            return expr, base_val

        # 约束 2: 业务逻辑判断
        if final_ranks is not None:
            # 决赛名次链：名次更好的人虚拟总分必须更小
            sorted_idx = sorted(range(n), key=lambda x: final_ranks[x])
            for idx in range(len(sorted_idx) - 1):
                h, l = sorted_idx[idx], sorted_idx[idx + 1]
                expr_h, base_h = get_v_expr(h)
                expr_l, base_l_val = get_v_expr(l)
                A_rows.append(expr_h - expr_l)
                b_l.append(-np.inf);
                b_u.append(base_l_val - base_h - 0.001)

        elif k_elim > 0:
            survivors = [i for i in range(n) if i not in elim_indices]
            if rule_type == 'Rank_Standard':
                # 标准淘汰：所有晋级者的虚拟总分必须优于（小于）淘汰者
                for e in elim_indices:
                    for s in survivors:
                        expr_e, base_e = get_v_expr(e)
                        expr_s, base_s = get_v_expr(s)
                        A_rows.append(expr_s - expr_e)
                        b_l.append(-np.inf);
                        b_u.append(base_e - base_s - 0.001)

            elif rule_type == 'Rank_Save':
                # S28+ 救人逻辑：每个淘汰者 e 必须处于 Bottom k+1
                # 即：全场表现比 e 差的人数必须 <= k
                M = n + 10  # 大 M 法常数
                for e_idx_in_list, e in enumerate(elim_indices):
                    y_start = n_x + e_idx_in_list * n
                    for i in range(n):
                        if i == e: continue
                        expr_e, base_e = get_v_expr(e)
                        expr_i, base_i = get_v_expr(i)
                        # V_e - V_i <= M * y_ei -> 如果 y_ei=0, 则 e 必须比 i 差（V_e > V_i）
                        # 也就是 y_ei 标记了那些比 e 表现更好（V更小）的人
                        A_rows.append(expr_i - expr_e - M * np.eye(1, n_vars, y_start + i).flatten())
                        b_l.append(-np.inf);
                        b_u.append(base_e - base_i - 0.001)

                    # 表现比 e 更好的人数必须 >= n - (k+1)
                    # 或者说表现比 e 差的人数 y_ei 为 0 的数量有限
                    row_y = np.zeros(n_vars)
                    for i in range(n):
                        if i != e: row_y[y_start + i] = 1
                    A_rows.append(row_y)
                    # 允许最多只有 k 个人的 V 比 e 更大（更差）
                    # 换言之，至少要有 n - (k+1) 个人的 V 比 e 更小（更好）
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
        return np.argsort(np.argsort(rj)) + 1  # 降级兜底

    def solve_percentage(self, pj, elim_indices, final_ranks=None):
        """百分比制求解：统一使用 0-100 步长进行内部计算"""
        n = len(pj)

        def obj(x):
            return np.sum((x - pj) ** 2)

        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 100}]

        # 同样引入 0.01 * judge_score 级别的平局判定扰动
        if final_ranks is not None:
            idx = np.argsort(final_ranks)
            for k in range(len(idx) - 1):
                h, l = idx[k], idx[k + 1]
                # 虚拟总分判别
                cons.append({'type': 'ineq',
                             'fun': lambda x, h=h, l=l: (pj[h] + x[h] + 0.01 * pj[h]) - (pj[l] + x[l] + 0.01 * pj[l])})
        elif elim_indices:
            survivors = [i for i in range(n) if i not in elim_indices]
            for e in elim_indices:
                for s in survivors:
                    cons.append({'type': 'ineq', 'fun': lambda x, e=e, s=s: (pj[s] + x[s] + 0.01 * pj[s]) - (
                                pj[e] + x[e] + 0.01 * pj[e])})

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
                # 保存原始预测值 (Rank 为 1-N，Percentage 为 0-100)
                row_data['predicted_fan_vote'] = vote_map.get(row['celebrity_name'], None)
                results.append(row_data)

        pd.DataFrame(results).to_csv(output_file, index=False)
        print(f"成功：计算完成，所有赛制约束已闭环，结果保存至 {output_file}")


if __name__ == "__main__":
    engine = DWTS_Final_Inference_Engine('cleaned_data.csv')
    engine.execute('predicted_fan_votes_full.csv')