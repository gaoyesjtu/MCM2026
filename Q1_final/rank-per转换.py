import pandas as pd
import numpy as np


def harmonize_votes(input_file, output_file, alpha=0.1):
    """
    读取CSV文件并进行排名到百分比的统化转换
    :param input_file: 上传的 CSV 文件路径
    :param output_file: 结果保存路径
    :param alpha: 幂律分布衰减系数 (默认 0.16)
    """
    # 1. 加载数据
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
        return

    results_list = []

    # 2. 按赛季和周次分组处理，确保每一集内部的投票百分比总和为 1
    for (s, w), group in df.groupby(['season', 'week']):
        group = group.copy()

        # 识别该周使用的赛制规则
        # 假设同一周的规则是一致的，取第一行即可
        rule = group['rule_type'].iloc[0] if 'rule_type' in group.columns else 'Percentage'

        # 仅处理有预测值的行
        mask = group['predicted_fan_vote'].notna()
        valid_votes = group.loc[mask, 'predicted_fan_vote']

        if not valid_votes.empty:
            if rule == 'Percentage':
                # 比例制：直接归一化
                total = valid_votes.sum()
                group.loc[mask, 'harmonized_fan_percent'] = valid_votes / total if total > 0 else 1.0 / len(valid_votes)
            else:
                # 排名制 (Rank_Standard 或 Rank_Save)
                # 应用公式: Weight = 1 / (Rank ^ Alpha)
                weights = 1 / (valid_votes ** alpha)
                total_weight = weights.sum()
                group.loc[mask, 'harmonized_fan_percent'] = weights / total_weight if total_weight > 0 else 1.0 / len(
                    valid_votes)
        else:
            group['harmonized_fan_percent'] = np.nan

        results_list.append(group)

    # 3. 合并并保存
    final_df = pd.concat(results_list)
    final_df.to_csv(output_file, index=False)
    print(f"处理完成！结果已保存至: {output_file}")

    # 打印前 10 行预览
    print("\n--- 转换结果预览 ---")
    preview_cols = ['season', 'week', 'celebrity_name', 'rule_type', 'predicted_fan_vote', 'harmonized_fan_percent']
    print(final_df[preview_cols].head(10))


if __name__ == "__main__":
    # 执行转换
    harmonize_votes(
        input_file='final_estimation_rank.csv',
        output_file='final_estimation.csv',
        alpha=0.1
    )