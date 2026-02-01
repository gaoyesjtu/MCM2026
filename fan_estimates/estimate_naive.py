import pandas as pd
import numpy as np
import scipy.optimize as opt
import re


df = pd.read_csv('data/cleaned_data.csv')

# ==========================================
# 1. Estimation Modeling Functions
# ==========================================

def solve_fan_votes_percent(sub_df):
    """
    Estimates fan votes for 'Percent Method' using optimization.
    Constraints: 
    1. Sum of fan votes = 1
    2. Fan votes >= 0.001
    3. Eliminated contestant total score <= Survivor total scores
       (Judge% + Fan%)_Elim <= (Judge% + Fan%)_Safe
       => Fan_Elim - Fan_Safe <= Judge_Safe - Judge_Elim
    """
    contestants = sub_df['celebrity_name'].tolist()
    judge_percents = sub_df['judge_percent'].values
    is_eliminated = sub_df['is_eliminated'].values
    n = len(contestants)
    
    # Check if anyone is eliminated this week
    if sum(is_eliminated) == 0:
        # No elimination, assume uniform fan votes (1/n) or neutral
        return np.full(n, 1.0/n)
    
    # Indices
    elim_indices = np.where(is_eliminated)[0]
    safe_indices = np.where(~is_eliminated)[0]
    
    # Objective: Minimize sum of squared differences from Uniform Distribution
    # We want the "least surprising" fan vote that explains the result.
    # Uniform = 1/n
    
    def objective(fan_votes):
        return np.sum((fan_votes - 1.0/n)**2)
    
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, # Sum to 1
    ]
    
    # Add inequality constraints: (Judge_Safe - Judge_Elim) >= (Fan_Elim - Fan_Safe)
    # i.e., Fan_Safe - Fan_Elim + (Judge_Safe - Judge_Elim) >= 0
    # Wait, condition is Total_Elim <= Total_Safe
    # J_E + F_E <= J_S + F_S
    # => F_S - F_E + (J_S - J_E) >= 0
    
    for e_idx in elim_indices:
        for s_idx in safe_indices:
            j_diff = judge_percents[s_idx] - judge_percents[e_idx]
            # Constraint: fan_votes[s_idx] - fan_votes[e_idx] + j_diff >= 0
            constraints.append({
                'type': 'ineq', 
                'fun': lambda x, s=s_idx, e=e_idx, j=j_diff: x[s] - x[e] + j
            })
            
    bounds = [(0.001, 1.0) for _ in range(n)]
    
    # Initial guess: Uniform
    x0 = np.full(n, 1.0/n)
    
    result = opt.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if result.success:
        return result.x
    else:
        # Fallback if impossible (e.g. controversy where rules were bent or judges saved)
        # Return result anyway or uniform
        return result.x

def solve_fan_votes_rank(sub_df):
    """
    Estimates fan votes for 'Rank Method'.
    Returns estimated Ranks.
    Rank 1 = Best.
    Eliminated = Highest Sum of Ranks.
    """
    contestants = sub_df['celebrity_name'].tolist()
    judge_ranks = sub_df['judge_rank'].values
    is_eliminated = sub_df['is_eliminated'].values
    n = len(contestants)
    
    if sum(is_eliminated) == 0:
        # Return average ranks
        return np.arange(1, n+1) # Placeholder, actually order doesn't matter
    
    elim_indices = np.where(is_eliminated)[0]
    safe_indices = np.where(~is_eliminated)[0]
    
    # Heuristic:
    # We want to assign Fan Ranks (permutation of 1..n) such that:
    # (J_E + F_E) >= (J_S + F_S)
    
    # Let's try to assign the "Worst" fan ranks to the eliminated contestants to satisfy the condition,
    # and "Best" fan ranks to those with low judge scores who survived.
    
    # This is a matching problem. Let's do a simple simulation/shuffle to find a valid one 
    # closest to random/neutral.
    
    # Simplified Logic:
    # 1. Create a base 'fan score'
    # 2. If Eliminated, give them bad fan score.
    # 3. If Safe but Bad Judge Score, give them Good Fan Score.
    
    # Let's try 100 random permutations and pick the one that satisfies constraints
    # and has least correlation with Judge Ranks (assuming independence).
    
    best_perm = None
    min_violations = float('inf')
    
    # Deterministic fallback:
    # Sort survivors by Judge Rank (Descending: Worst judges first).
    # Give them the Best Fan Ranks (1, 2, 3...) to save them.
    # Give Eliminated the Worst Fan Ranks.
    
    # Create indices array
    indices = np.arange(n)
    
    # Candidate Fan Ranks
    possible_ranks = np.arange(1, n+1)
    
    # Strategy:
    # Assign 'Worst' available ranks to Eliminated.
    # Assign 'Best' available ranks to Survivors with 'Worst' Judge Ranks (they need saving).
    
    fan_ranks = np.zeros(n)
    
    # 1. Assign Worst Ranks to Eliminated
    # If multiple eliminated, sort by judge rank? Doesn't matter much.
    for i, e_idx in enumerate(elim_indices):
        fan_ranks[e_idx] = n - i # Assign n, n-1...
        
    # Remove used ranks
    used_ranks = set(fan_ranks[elim_indices])
    remaining_ranks = sorted([r for r in possible_ranks if r not in used_ranks])
    
    # 2. Assign Remaining Ranks to Survivors
    # Sort Survivors by Judge Rank Descending (Worst score first). 
    # They need the best fan votes (smallest rank numbers) to survive.
    survivors_sorted = sorted(safe_indices, key=lambda idx: judge_ranks[idx], reverse=True)
    
    for i, s_idx in enumerate(survivors_sorted):
        fan_ranks[s_idx] = remaining_ranks[i]
        
    return fan_ranks

# ==========================================
# 2. Main Loop
# ==========================================
estimated_data = []

for (season, week), group in df.groupby(['season', 'week']):
    # Determine Method
    # Rank: 1-2, 28-34
    # Percent: 3-27
    if season <= 2 or season >= 28:
        method = 'rank'
        est_votes = solve_fan_votes_rank(group)
        # Normalize to 0-1 scale for consistency.md in output? Or keep as rank.
        # Let's keep as rank but also store a 'normalized_score'
        group['estimated_fan_metric'] = est_votes # This is Rank (Lower is better)
        # For Rank: High Metric = Bad. For Percent: High Metric = Good. 
        # Invert Rank for consistency.md later: (N+1 - Rank) / N
        n = len(group)
        group['estimated_fan_support'] = (n + 1 - est_votes) / n
    else:
        method = 'percent'
        est_votes = solve_fan_votes_percent(group)
        group['estimated_fan_metric'] = est_votes # This is Percent (Higher is better)
        group['estimated_fan_support'] = est_votes # Already higher is better

    group['scoring_method'] = method
    estimated_data.append(group)

df_estimated = pd.concat(estimated_data)

# Show result snippet
print(df_estimated[['season', 'week', 'celebrity_name', 'judge_rank', 'results', 'estimated_fan_support']].head(10))

# Save for user
df_estimated.to_csv('fan_estimates/estimated_fan_votes.csv', index=False)
print("Saved estimated_fan_votes.csv")