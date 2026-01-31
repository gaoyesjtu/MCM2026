import pandas as pd
import numpy as np
import scipy.optimize as opt
import re
import matplotlib.pyplot as plt


# ==========================================
# 1. Optimization Algorithms
# ==========================================

# Algorithm A: Percent Method (Judge Prior) - For S3-27
def solve_fan_votes_judge_prior(sub_df):
    contestants = sub_df['celebrity_name'].tolist()
    judge_percents = sub_df['judge_percent'].values
    is_eliminated = sub_df['is_eliminated'].values
    n = len(contestants)
    
    priors = judge_percents.copy()
    priors = np.maximum(priors, 0.001)
    priors = priors / np.sum(priors)
    
    def objective(fan_votes):
        return np.sum( ((fan_votes - priors)**2) / priors )
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    
    elim_indices = np.where(is_eliminated)[0]
    safe_indices = np.where(~is_eliminated)[0]
    
    for e_idx in elim_indices:
        for s_idx in safe_indices:
            j_diff = judge_percents[s_idx] - judge_percents[e_idx]
            constraints.append({'type': 'ineq', 'fun': lambda x, s=s_idx, e=e_idx, j=j_diff: x[s] - x[e] + j})
            
    bounds = [(0.001, 1.0) for _ in range(n)]
    x0 = priors 
    
    result = opt.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if result.success:
        return result.x
    else:
        return priors

# Algorithm B: Rank Method (Naive) - For S1-2, S28-34
def solve_fan_votes_rank_method(sub_df):
    contestants = sub_df['celebrity_name'].tolist()
    judge_ranks = sub_df['judge_rank'].values
    is_eliminated = sub_df['is_eliminated'].values
    n = len(contestants)
    
    if sum(is_eliminated) == 0:
        return np.arange(1, n+1)
    
    elim_indices = np.where(is_eliminated)[0]
    safe_indices = np.where(~is_eliminated)[0]
    
    fan_ranks = np.zeros(n)
    
    # 1. Punish Eliminated
    for i, e_idx in enumerate(elim_indices):
        fan_ranks[e_idx] = n - i 
        
    used_ranks = set(fan_ranks[elim_indices])
    possible_ranks = np.arange(1, n+1)
    remaining_ranks = sorted([r for r in possible_ranks if r not in used_ranks])
    
    # 2. Save Survivors (Prioritize those with bad judge scores)
    survivors_sorted = sorted(safe_indices, key=lambda idx: judge_ranks[idx], reverse=True)
    
    for i, s_idx in enumerate(survivors_sorted):
        fan_ranks[s_idx] = remaining_ranks[i]
        
    return fan_ranks

# ==========================================
# 2. Execution Pipeline
# ==========================================

# Load Data
df = pd.read_csv('data/cleaned_data.csv')

# --- Phase 1: Process S3-27 (Percent Method) to learn distribution ---
percent_results = []
percent_seasons_df = df[(df['season'] >= 3) & (df['season'] <= 27)]

for (season, week), group in percent_seasons_df.groupby(['season', 'week']):
    if len(group) <= 1: continue
    
    est_percents = solve_fan_votes_judge_prior(group)
    
    group['estimated_fan_metric'] = est_percents
    group['normalized_fan_support'] = est_percents # Percent is already normalized
    group['scoring_system'] = 'percent'
    group['fan_judge_deviation'] = est_percents - group['judge_percent']
    
    percent_results.append(group)

df_percent = pd.concat(percent_results)

# --- Phase 2: Learn Distribution Map (N, Rank) -> Mean Percent ---
# Create a lookup table
distribution_map = {} # Key: (n_contestants, rank_position), Value: mean_percent

# Group by week to get N and sorted percents
for (season, week), group in df_percent.groupby(['season', 'week']):
    n = len(group)
    percents = sorted(group['estimated_fan_metric'].values, reverse=True) # Highest first
    
    for rank_idx, val in enumerate(percents):
        rank = rank_idx + 1 # 1-based rank
        key = (n, rank)
        if key not in distribution_map:
            distribution_map[key] = []
        distribution_map[key].append(val)

# Average them
final_map = {}
for key, val_list in distribution_map.items():
    final_map[key] = np.mean(val_list)

# Fallback function for missing keys
def get_expected_percent(n, rank):
    # If exact (n, rank) exists, use it
    if (n, rank) in final_map:
        return final_map[(n, rank)]
    
    # If not, try to find closest N
    available_ns = sorted(list(set([k[0] for k in final_map.keys()])))
    if not available_ns: return 1.0/n # Should not happen
    
    closest_n = min(available_ns, key=lambda x: abs(x-n))
    
    # Map rank to closest_n relative rank
    # rank/n approx new_rank/closest_n
    new_rank = max(1, min(closest_n, int(round(rank * closest_n / n))))
    
    if (closest_n, new_rank) in final_map:
        return final_map[(closest_n, new_rank)]
    
    return 1.0/n # Ultimate fallback

# --- Phase 3: Process S1-2, S28-34 (Rank Method) and Apply Mapping ---
rank_results = []
rank_seasons_df = df[(df['season'] <= 2) | (df['season'] >= 28)]

for (season, week), group in rank_seasons_df.groupby(['season', 'week']):
    if len(group) <= 1: continue
    n = len(group)
    
    est_ranks = solve_fan_votes_rank_method(group)
    
    group['estimated_fan_metric'] = est_ranks # This is Rank (1, 2, 3...)
    group['scoring_system'] = 'rank'
    
    # Apply Mapping: Convert Rank to Expected Percent
    # We map estimate_rank X -> expected percent
    expected_percents = [get_expected_percent(n, r) for r in est_ranks]
    
    group['normalized_fan_support'] = expected_percents # Use the learned percent
    
    # Deviation (Tricky for rank, maybe use normalized - judge_percent?)
    # Judge percent exists in 'group' even for rank seasons (calculated in clean step)
    group['fan_judge_deviation'] = group['normalized_fan_support'] - group['judge_percent']
    
    rank_results.append(group)

df_rank = pd.concat(rank_results)

# --- Phase 4: Merge and Save ---
df_final = pd.concat([df_percent, df_rank])
df_final = df_final.sort_values(['season', 'week', 'celebrity_name'])

output_file = 'merged_fan_estimates_mapped.csv'
df_final.to_csv(output_file, index=False)

# Visualization of the Mapping Logic
# Plot the learned distribution for a few sample Ns
sample_ns = [4, 8, 12]
plt.figure(figsize=(10, 6))
for n in sample_ns:
    ranks = range(1, n+1)
    percents = [get_expected_percent(n, r) for r in ranks]
    plt.plot(ranks, percents, marker='o', label=f'N={n} Contestants')

plt.title('Learned Fan Vote Distribution (from Seasons 3-27)')
plt.xlabel('Fan Vote Rank (1=Highest)')
plt.ylabel('Expected Fan Vote %')
plt.grid(True)
plt.legend()
plt.savefig('fan_vote_distribution_curve.png')

print("Mapping complete. Curve saved.")
print(df_final[['season', 'week', 'celebrity_name', 'scoring_system', 'estimated_fan_metric', 'normalized_fan_support']].head())