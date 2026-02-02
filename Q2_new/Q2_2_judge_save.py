import pandas as pd
import numpy as np

df = pd.read_csv('Q2_new/final_estimation.csv')

def find_elim_modified(group, method, season):

    has_save = 1
    
    if method == 'rank':

        group['score'] = group['judge_rank'] + group['predicted_fan_vote']
        #Tie-break: predicted_fan_vote (higher is worse)
        sort_cols = ['score', 'predicted_fan_vote']
        asc = [False, False]    # control ascending / descending
    elif method == 'percent':

        group['score'] = group['judge_percent'] + group['harmonized_fan_percent']
        # Tie-break: fan votes percent
        sort_cols = ['score', 'judge_percent']
        asc = [True, True]
    elif method == 'fan':
  
        group['score'] = group['harmonized_fan_percent']
        sort_cols = ['score', 'total_judge_score']
        asc = [True, True]
    
    sorted_group = group.sort_values(by=sort_cols, ascending=asc)
    
    if has_save and len(group) >= 2:
        # Identify bottom 2 based on the method's logic
        bottom_two = sorted_group.head(2)
        # Judges eliminate the one with the LOWER technical score
        return bottom_two.sort_values(by='total_judge_score', ascending=True).iloc[0]['celebrity_name']
    else:
        return sorted_group.iloc[0]['celebrity_name']

all_data = []
for (s, w), group in df.groupby(['season', 'week']):
    
    if s > 27:
        break

    active = group[group['total_judge_score'] > 0].copy()
    if len(active) < 2: continue
    
    r = find_elim_modified(active.copy(), 'rank', s)
    p = find_elim_modified(active.copy(), 'percent', s)
    f = find_elim_modified(active.copy(), 'fan', s)
    
    eliminated_this_week = active[active['results'].str.contains(f'Eliminated Week {w}$', na=False, case=False)]['celebrity_name'].tolist()
    actual = ", ".join(eliminated_this_week)
    
    all_data.append({
        'season': s, 'week': w,
        'rank_elim_judgesave': r, 'perc_elim_judgesave': p, 'fan_elim_judgesave': f,
        'actual': actual, 'is_ignored': (actual == "")
    })

results_df = pd.DataFrame(all_data)
# Filter: ignore weeks with empty actual
valid_df = results_df[~results_df['is_ignored']].copy()
# Discrepant weeks
diff_df = valid_df[valid_df['rank_elim_judgesave'] != valid_df['perc_elim_judgesave']].copy()


r_matches = (diff_df['rank_elim_judgesave'] == diff_df['fan_elim_judgesave']).sum()
p_matches = (diff_df['perc_elim_judgesave'] == diff_df['fan_elim_judgesave']).sum()

print(f"Valid Weeks: {len(valid_df)}")
print(f"Discrepant Weeks: {len(diff_df)}")
print(f"Matches - Rank with Fan: {r_matches}")
print(f"Matches - Percent with Fan: {p_matches}")

results_df.to_csv('Q2_new/rank_percent_judgesave_comparison.csv', index=False)
diff_df.to_csv('Q2_new/rank_percent_judgesave_discrepancies.csv', index=False)

'''
Origin:
Valid Weeks: 208
Discrepant Weeks: 88
Matches - Rank with Fan: 6
Matches - Percent with Fan: 75


Judge Save:
Valid Weeks: 208
Discrepant Weeks: 43
Matches - Rank with Fan: 2
Matches - Percent with Fan: 39

'''