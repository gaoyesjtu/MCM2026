import pandas as pd

df_est = pd.read_csv('Q2_new/final_estimation.csv')
df_no_save = pd.read_csv('Q2_new/rank_percent_comparison.csv')
df_with_save = pd.read_csv('Q2_new/rank_percent_judgesave_comparison.csv')

# 1. Recalculate the correct 'actual' column from final_estimation.csv
# Logic: actual should only have the person if results is 'Eliminated Week s' for week s
actual_list = []
for (s, w), group in df_est.groupby(['season', 'week']):
    # Only pick those whose result string exactly matches current week elimination
    # The format is typically "Eliminated Week X"
    eliminated = group[group['results'].str.contains(f'Eliminated Week {w}$', na=False, case=False)]['celebrity_name'].tolist()
    actual_str = ", ".join(eliminated) if eliminated else ""
    actual_list.append({'season': s, 'week': w, 'actual_fixed': actual_str})

df_actual_fixed = pd.DataFrame(actual_list)

# 2. Merge the documents
# Filter S1-27 first for efficiency
df_no_save_sub = df_no_save[df_no_save['season'] <= 27][['season', 'week', 'rank_elim', 'perc_elim', 'fan_elim']]
df_with_save_sub = df_with_save[df_with_save['season'] <= 27][['season', 'week', 'rank_elim_judgesave', 'perc_elim_judgesave', 'fan_elim_judgesave']]

# Merge them
merged = pd.merge(df_no_save_sub, df_with_save_sub, on=['season', 'week'], how='inner')

# Merge with the fixed actual column
merged = pd.merge(merged, df_actual_fixed, on=['season', 'week'], how='left')

# Rename actual_fixed to actual
merged = merged.rename(columns={'actual_fixed': 'actual'})

# Select and order columns as requested:
# season, week, rank_elim, perc_elim, fan_elim, rank_elim_judgesave, perc_elim_judgesave, fan_elim_judgesave, actual
cols = ['season', 'week', 'rank_elim', 'perc_elim', 'fan_elim', 
        'rank_elim_judgesave', 'perc_elim_judgesave', 'fan_elim_judgesave', 'actual']
merged = merged[cols]

# Save the final file
merged.to_csv('Q2_new/S1_S27_combined.csv', index=False)
print('Merge done!')
