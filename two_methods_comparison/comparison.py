import pandas as pd
import numpy as np

# Load the datasets
# Note the typo in the filename provided by the user's file list
try:
    fan_estimates = pd.read_csv('fan_estimates/fianal_fan_estimate.csv')
except FileNotFoundError:
    # Fallback in case the typo was corrected in the environment but I'm unaware
    fan_estimates = pd.read_csv('fan_estimates/final_fan_estimate.csv')

cleaned_data = pd.read_csv('data/cleaned_data.csv')

# Data Processing and Comparison Logic

# Ensure we are working with the fan estimates dataframe
df = fan_estimates.copy()

# 1. Normalize Fan Support to be sure (it should sum to 1 per week)
# But first, let's just inspect if it does.
fan_sum = df.groupby(['season', 'week'])['normalized_fan_support'].sum()
# print(fan_sum.describe()) # Should be close to 1

# 2. Define calculation logic
def calculate_outcomes(group):
    # --- Rank Method ---
    # Judge Rank: Higher Score = Better Rank (1 is best)
    # Method 'min' assigns the same rank to ties (e.g. 1, 1, 3)
    group['calc_judge_rank'] = group['total_judge_score'].rank(ascending=False, method='min')
    
    # Fan Rank: Higher Support = Better Rank (1 is best)
    group['calc_fan_rank'] = group['normalized_fan_support'].rank(ascending=False, method='min')
    
    # Combined Rank
    group['total_rank_score'] = group['calc_judge_rank'] + group['calc_fan_rank']
    
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
        
    # Fan Percent (normalized_fan_support is already a ratio, typically)
    # We re-normalize just in case to be strictly comparable to judge pct
    total_fan_support = group['normalized_fan_support'].sum()
    if total_fan_support == 0:
        group['calc_fan_pct'] = 0
    else:
        group['calc_fan_pct'] = group['normalized_fan_support'] / total_fan_support
        
    # Combined Percent (50/50 Split implies just adding the ratios if they are both 0-1)
    # The problem description says "Judge scores... / sum of total judge scores" + "Fan votes / Total Fan Votes"
    # So it's a 50/50 weight implicitly.
    group['total_pct_score'] = group['calc_judge_pct'] + group['calc_fan_pct']
    
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
        
        # Analyze Bias
        # Get the Fan Rank of the people involved
        # If Rank eliminates X (Fan Rank High) and Percent saves X
        # And Percent eliminates Y (Fan Rank Low) and Rank saves Y
        # Then Percent favored the Fan Vote (by keeping the High Fan Rank person).
        
        # We need to look at the specific people in the difference sets
        for r_person in rank_only:
            for p_person in pct_only:
                # Get metrics
                r_metrics = group[group['celebrity_name'] == r_person].iloc[0]
                p_metrics = group[group['celebrity_name'] == p_person].iloc[0]
                
                # Check Fan Ranks
                # Lower Fan Rank number = Higher Votes
                r_fan_rank = r_metrics['calc_fan_rank'] 
                p_fan_rank = p_metrics['calc_fan_rank']
                
                favored_method = 'Indeterminate'
                
                # If Rank Method eliminated someone with BETTER fan support (Lower Rank Num) 
                # than the person eliminated by Percent Method...
                # Then Percent Method saved the popular person. Percent favors Fans.
                if r_fan_rank < p_fan_rank: 
                    favored_method = 'Percent Method' # Percent kept the popular one, eliminated the unpopular one
                elif p_fan_rank < r_fan_rank:
                    favored_method = 'Rank Method' # Rank kept the popular one, eliminated the unpopular one
                
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
    print(disagreement_df['favored_method'].value_counts())
    print("\nSample Disagreements:")
    disagreement_df.to_csv('two_methods_comparison/diff_outcome.csv')
else:
    print("No disagreements found.")