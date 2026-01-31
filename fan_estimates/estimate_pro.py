import pandas as pd
import numpy as np
import scipy.optimize as opt
import re
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('data/cleaned_data.csv')

# ==========================================
# 1. Enhanced Model: Judge Score as Prior
# ==========================================

def solve_fan_votes_judge_prior(sub_df):
    contestants = sub_df['celebrity_name'].tolist()
    judge_percents = sub_df['judge_percent'].values
    is_eliminated = sub_df['is_eliminated'].values
    n = len(contestants)
    
    # --- CHANGE IS HERE ---
    # Prior P_i is simply the Judge's Percentage
    # Logic: Fans agree with judges unless proven otherwise
    priors = judge_percents.copy()
    
    # Safety: Ensure priors are not zero (though judge scores > 0 usually)
    priors = np.maximum(priors, 0.001) 
    priors = priors / np.sum(priors) # Re-normalize just in case
    
    # Objective: Minimize Chi-Squared distance from Prior (Judges)
    # min sum( (F_i - P_i)^2 / P_i )
    def objective(fan_votes):
        return np.sum( ((fan_votes - priors)**2) / priors )
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    
    elim_indices = np.where(is_eliminated)[0]
    safe_indices = np.where(~is_eliminated)[0]
    
    # Constraints: F_s - F_e >= J_e - J_s
    # Note: J_e and J_s are from 'judge_percents'
    for e_idx in elim_indices:
        for s_idx in safe_indices:
            j_diff = judge_percents[s_idx] - judge_percents[e_idx]
            constraints.append({'type': 'ineq', 'fun': lambda x, s=s_idx, e=e_idx, j=j_diff: x[s] - x[e] + j})
            
    bounds = [(0.001, 1.0) for _ in range(n)]
    x0 = priors # Start at the prior
    
    result = opt.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x, priors

# ==========================================
# 2. Execution (Seasons 3-27: Percent Method)
# ==========================================
percent_seasons = df[(df['season'] >= 3) & (df['season'] <= 27)].copy()
results_list = []

for (season, week), group in percent_seasons.groupby(['season', 'week']):
    if len(group) <= 1: continue
    
    # Run Model
    est_votes, priors = solve_fan_votes_judge_prior(group)
    
    group['estimated_fan_percent'] = est_votes
    group['judge_prior'] = priors # This should be almost same as judge_percent
    
    # Calculate Deviation: How much did fans disagree with judges?
    # Simple difference
    group['fan_judge_diff'] = group['estimated_fan_percent'] - group['judge_percent']
    
    results_list.append(group)

df_results = pd.concat(results_list)

# ==========================================
# 3. Results & Visualization
# ==========================================

# A. Print Sample Output
print("Model Results Sample (Judge Prior):")
print(df_results[['season', 'week', 'celebrity_name', 'judge_percent', 'estimated_fan_percent', 'fan_judge_diff', 'is_eliminated']].head(10))

# B. Identify Largest Deviations (Controversies)
# Where Fan Vote >>> Judge Score (Underdog saved by fans)
print("\nTop 5 Fan Favorites (Highest Positive Deviation from Judge Score):")
print(df_results.sort_values('fan_judge_diff', ascending=False)[['season', 'week', 'celebrity_name', 'judge_percent', 'estimated_fan_percent', 'fan_judge_diff', 'results']].head(5))

# Where Fan Vote <<< Judge Score (Shocking Eliminations?)
# Note: If eliminated, model forces F to be low.
print("\nTop 5 'Shocking' Low Fan Support (Negative Deviation):")
print(df_results.sort_values('fan_judge_diff', ascending=True)[['season', 'week', 'celebrity_name', 'judge_percent', 'estimated_fan_percent', 'fan_judge_diff', 'results']].head(5))

# C. Scatter Plot
plt.figure(figsize=(10, 8))
# Color by 'is_eliminated' to see where they fall
sns.scatterplot(x='judge_percent', y='estimated_fan_percent', hue='is_eliminated', data=df_results, alpha=0.6, palette={True: 'red', False: 'blue'})
# Add diagonal line (y=x)
plt.plot([0, 0.4], [0, 0.4], 'k--', lw=1)
plt.title('Judge Score (Prior) vs Estimated Fan Vote (Posterior)')
plt.xlabel('Judge Score % (Prior)')
plt.ylabel('Estimated Fan Vote %')
plt.annotate('Fans Agreed with Judges', xy=(0.2, 0.2), xytext=(0.25, 0.15), arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate('Fans Saved Low Scorer', xy=(0.1, 0.25), xytext=(0.05, 0.3), arrowprops=dict(facecolor='black', shrink=0.05))
plt.savefig('fan_estimates/judge_prior_scatter.png')

# Save CSV
df_results.to_csv('fan_estimates/fan_votes_judge_prior.csv', index=False)
print("\nSaved results to 'fan_votes_judge_prior.csv'")