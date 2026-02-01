import pandas as pd
import numpy as np

# Load the datasets
# Note the typo in the filename provided by the user's file list

fan_estimates = pd.read_csv(r'Q1_fixed\estimate_fan_final.csv')

# Data Processing and Comparison Logic

# Ensure we are working with the fan estimates dataframe
df = fan_estimates.copy()

# 1. Normalize Fan Support to be sure (it should sum to 1 per week)
# But first, let's just inspect if it does.
fan_sum = df.groupby(['season', 'week'])['predicted_fan_vote'].sum()
# print(fan_sum.describe()) # Should be close to 1

# 2. Define calculation logic
def calculate_outcomes(group):
    # --- Rank Method ---
    # Judge Rank: Higher Score = Better Rank (1 is best)
    # Method 'min' assigns the same rank to ties (e.g. 1, 1, 3)
    group['calc_judge_rank'] = group['total_judge_score'].rank(ascending=False, method='min')
    
    # Fan Rank: Higher Support = Better Rank (1 is best)
    group['calc_fan_rank'] = group['predicted_fan_vote'].rank(ascending=False, method='min')
    
    # Combined Rank
    group['total_rank_score'] = (group['calc_judge_rank'] + group['calc_fan_rank']) + group['calc_judge_rank'] * 0.01

    
    # Eliminated by Rank: Highest Score is worst.
    # We rank the 'total_rank_score' descending. Rank 1 here means "Highest Score" (Worst).
    # To handle ties for elimination (multiple people with same worst score), we might have multiple losers.
    # For this analysis, we will mark the person with the Max Score as the potential elimination.
    max_rank_score = group['total_rank_score'].max()
    group['eliminated_by_rank'] = group['total_rank_score'] == max_rank_score
    
    # --- Percent Method ---
    # Judge Percent
    total_judge_score = group['total_judge_score'].sum()
    if total_judge_score == 0:
        group['calc_judge_pct'] = 0
    else:
        group['calc_judge_pct'] = group['total_judge_score'] / total_judge_score
        

    total_fan_support = group['predicted_fan_vote'].sum()
    if total_fan_support == 0:
        group['calc_fan_pct'] = 0
    else:
        group['calc_fan_pct'] = group['predicted_fan_vote'] / total_fan_support
        
    group['total_pct_score'] = (group['calc_judge_pct'] + group['calc_fan_pct']) * group['calc_judge_pct'] * 0.01
    
    # Eliminated by Percent: Lowest Score is worst.
    min_pct_score = group['total_pct_score'].min()
    group['eliminated_by_percent'] = group['total_pct_score'] == min_pct_score
    
    return group

# Apply to each season and week
results_df = df.groupby(['season', 'week']).apply(calculate_outcomes)

# Drop the grouping index added by apply
results_df = results_df.reset_index(drop=True)

# 3. Analyze Differences
# Filter for rows where they are identified as eliminated by at least one method
potential_losers = results_df[results_df['eliminated_by_rank'] | results_df['eliminated_by_percent']].copy()

# Group by week to find disagreements
disagreement_stats = []
season_stats = []

per_fa, rk_fa =0, 0

for (season, week), group in results_df.groupby(['season', 'week']):
    # Get losers for each method
    losers_rank = group[group['eliminated_by_rank']]['celebrity_name'].tolist()
    losers_pct = group[group['eliminated_by_percent']]['celebrity_name'].tolist()
    
    # Check if the sets of losers are different
    # Note: Sets might differ if one method has a tie and the other doesn't, or if they point to completely different people.
    
    set_rank = set(losers_rank)
    set_pct = set(losers_pct)
    
    match = (set_rank == set_pct)
    
    if not match:
        # Disagreement found
        # Who would Rank eliminate that Percent wouldn't?
        rank_only = set_rank - set_pct
        # Who would Percent eliminate that Rank wouldn't?
        pct_only = set_pct - set_rank
        

        for r_person in rank_only:
            for p_person in pct_only:
                # Get metrics
                r_metrics = group[group['celebrity_name'] == r_person].iloc[0]
                p_metrics = group[group['celebrity_name'] == p_person].iloc[0]
                

                r_fan_rank = r_metrics['calc_fan_pct']  # 只被 rank 制度淘汰的人
                p_fan_rank = p_metrics['calc_fan_pct']  # 只被 percentage 制度淘汰的人
                
                favored_method = 'Indeterminate'
                
                if r_fan_rank > p_fan_rank: 
                    favored_method = 'Percent Method' # Percent kept the popular one, eliminated the unpopular one
                    per_fa += 1
                else:
                    favored_method = 'Rank Method' # Rank kept the popular one, eliminated the unpopular one
                    rk_fa += 1

                print(f'r_fan_pct:{r_fan_rank}  p_fan_pct:{p_fan_rank} favored_method:{favored_method}')
                
                disagreement_stats.append({
                    'season': season,
                    'week': week,
                    'rank_loser': r_person,
                    'rank_loser_fan_rank': r_fan_rank,
                    'pct_loser': p_person,
                    'pct_loser_fan_rank': p_fan_rank,
                    'favored_method': favored_method
                })

disagreement_df = pd.DataFrame(disagreement_stats)

print("Number of Disagreements:", len(disagreement_df))
if len(disagreement_df) > 0:
    #print(disagreement_df['favored_method'].value_counts())
    print(f'Percent Method:{per_fa} Rank Method:{rk_fa}')
    disagreement_df.to_csv('two_methods_comparison/diff_outcome.csv')
else:
    print("No disagreements found.")