import pandas as pd
import numpy as np
from scipy.optimize import minimize


# 1. 核心推理引擎
class DWTSInferenceEngine:
    def solve_percentage(self, pj, elim_indices, final_ranks=None):
        n = len(pj)

        def objective(x):
            return np.sum((x - pj) ** 2)

        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        if final_ranks is not None:
            sorted_idx = np.argsort(final_ranks)
            for k in range(len(sorted_idx) - 1):
                h, l = sorted_idx[k], sorted_idx[k + 1]
                cons.append({'type': 'ineq', 'fun': lambda x, h=h, l=l: (pj[h] + x[h]) - (pj[l] + x[l]) + 1e-5})
        elif elim_indices:
            survivors = [i for i in range(n) if i not in elim_indices]
            for e in elim_indices:
                for s in survivors:
                    cons.append({'type': 'ineq', 'fun': lambda x, s=s, e=e: (pj[s] + x[s]) - (pj[e] + x[e])})
        res = minimize(objective, pj, method='SLSQP', bounds=[(0, 1)] * n, constraints=cons)
        return res.x if res.success else pj

    def solve_rank(self, rj, elim_indices, rule_type, final_ranks=None):
        n = len(rj)
        for _ in range(2000):
            v = np.random.permutation(np.arange(1, n + 1))
            total = rj + v
            valid = True
            if final_ranks is not None:
                s_idx = np.argsort(final_ranks)
                for k in range(len(s_idx) - 1):
                    h, l = s_idx[k], s_idx[k + 1]
                    if total[h] > total[l] or (total[h] == total[l] and rj[h] > rj[l]):
                        valid = False;
                        break
            elif elim_indices:
                sorted_t = sorted(total)
                if rule_type == 'Rank_Standard':
                    thresh = sorted_t[-len(elim_indices)]
                    if any(total[e] < thresh for e in elim_indices): valid = False
                else:
                    limit = sorted_t[-2] if n >= 2 else sorted_t[-1]
                    if all(total[e] < limit for e in elim_indices): valid = False
            if valid: return v
        return np.argsort(np.argsort(rj)) + 1


# 2. 执行与映射
df = pd.read_csv('cleaned_data.csv')
engine = DWTSInferenceEngine()
ALPHA = 0.16  # 基于 S3-27 拟合的衰减率

results = []
for (s, w), group in df.groupby(['season', 'week']):
    rule = 'Rank_Standard' if s <= 2 else ('Rank_Save' if s >= 28 else 'Percentage')
    group = group.copy()
    active = group[~group['results'].str.contains('withdrew', case=False, na=False)].copy()
    if active.empty: continue

    pj, rj = active['judge_percent'].values, active['judge_rank'].values
    elim = np.where(active['is_eliminated'])[0].tolist()
    is_finale = (w == df[df['season'] == s]['week'].max())
    f_ranks_limit = active['results'].str.extract(r'(\d+)').astype(float).values if is_finale else None

    # 原始推理
    raw_val = engine.solve_percentage(pj, elim, f_ranks_limit) if rule == 'Percentage' \
        else engine.solve_rank(rj, elim, rule, f_ranks_limit)

    active['inferred_fan_val'] = raw_val
    active['rule_type'] = rule

    # 【核心转换步骤】
    if rule == 'Percentage':
        active['harmonized_fan_percent'] = active['inferred_fan_val']
    else:
        # 排名转百分比 (基于幂律分布)
        ranks = active['inferred_fan_val'].astype(float)
        weights = 1.0 / (ranks ** ALPHA)
        active['harmonized_fan_percent'] = weights / weights.sum()

    results.append(active)

# 3. 输出保存
final_df = pd.concat(results)
final_df.to_csv('final_fan_estimate.csv', index=False)