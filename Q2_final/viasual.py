import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

# --- 1. 定义颜色 ---
# 将 RGB 值转换为 matplotlib 所需的 0-1 范围
c_lightest = (145/255, 205/255, 200/255)
c_medium_light = (111/255, 185/255, 208/255)
c_medium_dark = (84/255, 153/255, 189/255)
c_darkest = (57/255, 129/255, 175/255)

# 设置通用字体和线宽参数
font_title_size = 14
font_label_size = 11
font_content_size = 9
line_width = 2

# --- 2. 创建画布 ---
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.axis('off')  # 关闭坐标轴

# --- 3. 绘制辅助函数 ---
def draw_box(x, y, width, height, color, label, title_color=c_darkest, content_text=None):
    # 绘制带圆角的矩形背景
    box = patches.FancyBboxPatch((x, y), width, height,
                                 boxstyle="round,pad=0.1,rounding_size=0.2",
                                 ec=c_darkest, fc=color, lw=line_width)
    ax.add_patch(box)
    
    # 添加标题
    cx = x + width / 2
    cy = y + height - 0.4
    ax.text(cx, cy, label, ha='center', va='center', 
            fontsize=font_label_size, fontweight='bold', color=title_color)
    
    # 添加内容文本（如果有）
    if content_text:
        ax.text(cx, y + height/2 - 0.3, content_text, ha='center', va='top',
                fontsize=font_content_size, color=title_color, linespacing=1.5)
    return box

def draw_arrow(x_start, y_start, x_end, y_end):
    # 绘制带箭头的连接线
    arrow = patches.FancyArrowPatch(
        (x_start, y_start), (x_end, y_end),
        arrowstyle='-|>,head_width=0.4,head_length=0.8',
        connectionstyle="arc3,rad=0", # 直线
        color=c_darkest, lw=line_width
    )
    ax.add_patch(arrow)

# --- 4. 绘制图表元素 ---

# --- A. 输入层 (Input) - 左侧 ---
# 绘制一个大的背景框表示数据源 S1-S34
input_group_box = patches.Rectangle((0.5, 3.5), 2.5, 5, linewidth=1, edgecolor=c_lightest, facecolor='none', linestyle='--')
ax.add_patch(input_group_box)
ax.text(1.75, 8.7, "INPUT SOURCE\n(Subjects S1 - S34)", ha='center', fontsize=font_label_size, color=c_darkest)

input_box_h = 1.5
input_box_w = 2.2
input_x = 0.65

# Judge Scores 输入框
box_judge = draw_box(input_x, 6.5, input_box_w, input_box_h, c_lightest, "Judge Scores", 
                     content_text="(Quantitative/\nQualitative Data)")
# Fan Votes 输入框
box_fan = draw_box(input_x, 4.0, input_box_w, input_box_h, c_lightest, "Fan Votes", 
                   content_text="(Popularity Count)")


# --- B. 处理层 (Methods) - 中间 ---
method_box_h = 1.8
method_box_w = 2.5
method_x = 4.5

# Method 1: Rank Combination
box_m1 = draw_box(method_x, 7.5, method_box_w, method_box_h, c_medium_light, "Method 1:\nRank Combination",
                  content_text="Merge individual ranks\nfrom both sources")

# Method 2: Percent Combination
box_m2 = draw_box(method_x, 4.1, method_box_w, method_box_h, c_medium_light, "Method 2:\nPercent Combination",
                  content_text="Weighted average\nof normalized scores")

# Method 3: Fan votes only
box_m3 = draw_box(method_x, 0.7, method_box_w, method_box_h, c_medium_light, "Method 3:\nFan Votes Only",
                  content_text="Ranking based solely\non fan votes")


# --- C. 输出层 (Results) - 右侧 ---
result_box_h = 2.8
result_box_w = 2.2
result_x = 8.5

# 模拟不同的排名结果数据，展示差异性
res1_text = "1. S12\n2. S05\n3. S31\n...\n34. S09"
res2_text = "1. S05\n2. S12\n3. S18\n...\n34. S22"
res3_text = "1. S31\n2. S05\n3. S02\n...\n34. S12"

# Result 1 Box
box_r1 = draw_box(result_x, 7.0, result_box_w, result_box_h, c_medium_dark, "Result A\n(Rank Comb.)",
                  content_text=res1_text)
# Result 2 Box
box_r2 = draw_box(result_x, 3.6, result_box_w, result_box_h, c_medium_dark, "Result B\n(Percent Comb.)",
                  content_text=res2_text)
# Result 3 Box
box_r3 = draw_box(result_x, 0.2, result_box_w, result_box_h, c_medium_dark, "Result C\n(Fan Votes Only)",
                  content_text=res3_text)


# --- 5. 绘制连接箭头 ---

# 定义连接点的Y坐标偏移量，使箭头从盒子侧面中心发出
judge_y_mid = 6.5 + input_box_h / 2
fan_y_mid = 4.0 + input_box_h / 2
m1_y_mid = 7.5 + method_box_h / 2
m2_y_mid = 4.1 + method_box_h / 2
m3_y_mid = 0.7 + method_box_h / 2
r1_y_mid = 7.0 + result_box_h / 2
r2_y_mid = 3.6 + result_box_h / 2
r3_y_mid = 0.2 + result_box_h / 2

input_right_x = input_x + input_box_w + 0.1
method_left_x = method_x - 0.1
method_right_x = method_x + method_box_w + 0.1
result_left_x = result_x - 0.1

# --> Inputs to Method 1 (Rank Combination)
draw_arrow(input_right_x, judge_y_mid, method_left_x, m1_y_mid + 0.4) # Judge -> M1
draw_arrow(input_right_x, fan_y_mid, method_left_x, m1_y_mid - 0.4)   # Fan -> M1

# --> Inputs to Method 2 (Percent Combination)
draw_arrow(input_right_x, judge_y_mid, method_left_x, m2_y_mid + 0.4) # Judge -> M2
draw_arrow(input_right_x, fan_y_mid, method_left_x, m2_y_mid - 0.4)   # Fan -> M2

# --> Inputs to Method 3 (Fan Votes Only) - 关键：只有 Fan Votes 连接
# (Judge Score 不连接到这里)
draw_arrow(input_right_x, fan_y_mid, method_left_x, m3_y_mid)         # Fan -> M3

# --> Methods to Results
draw_arrow(method_right_x, m1_y_mid, result_left_x, r1_y_mid)
draw_arrow(method_right_x, m2_y_mid, result_left_x, r2_y_mid)
draw_arrow(method_right_x, m3_y_mid, result_left_x, r3_y_mid)

# --- 6. 添加区域标题 ---
ax.text(1.75, 9.5, "INPUTS", ha='center', fontsize=font_title_size, fontweight='bold', color=c_darkest)
ax.text(5.75, 9.5, "RANKING METHODS", ha='center', fontsize=font_title_size, fontweight='bold', color=c_darkest)
ax.text(9.6, 9.5, "OUTPUT RESULTS (Rankings)", ha='center', fontsize=font_title_size, fontweight='bold', color=c_darkest)

# 调整布局并保存
plt.tight_layout()
# 保存为高分辨率 PNG，适合论文插入
plt.savefig('ranking_methods_diagram_paper.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()