import os
import pandas as pd
import matplotlib.pyplot as plt

results_path = './results'


def IoU_hist(root):
    root = './results'
    file_paths = [os.path.join(root, i) for i in os.listdir(root)]
    dfs = []
    labels = []  # 存储文件基本名称

    for file_path in file_paths:
        df = pd.read_csv(file_path, sep='\s+')
        dfs.append(df)
        print(df)
        labels.append(os.path.basename(file_path).split('.')[0])

    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(16, 9))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 每个文件对应的柱状图颜色
    width = 0.3  # 柱状图的宽度
    xticks_labels = dfs[0]['methods']  # x轴标签

    x = [val for val in range(len(dfs[0]))]  # 初始化x轴位置

    for i, df in enumerate(dfs):
        iou_values = df['iou']
        ax.bar(x, iou_values, width, align='center', color=colors[i], label=f'{labels[i]} (IoU)')
        x = [val + width for val in x.copy()]  # 创建新的列表以保持正确的横坐标位置

    ax.set_xticks([val + width for val in range(len(dfs[0]))])
    ax.set_xticklabels(xticks_labels, rotation=45, ha='right')
    ax.set_xlabel('Methods')
    ax.set_ylabel('IoU Scores')
    ax.set_title('IoU Performance Comparison')
    ax.legend()
    ax.set_ylim(0, 100)  # 设置y轴范围为0到100
    for i, df in enumerate(dfs):
        values = df['dice']
        max_value = values.max()
        second_max_value = values.sort_values(ascending=False).iloc[1]
        min_value = values.min()
        max_index = values[values == max_value].index[0]
        second_max_index = values[values == second_max_value].index[0]
        min_index = values[values == min_value].index[0]
        file_name = labels[i]
        ax.annotate(f'Max: {max_value}\n({file_name})', xy=(max_index, max_value), xytext=(5, 5),
                    textcoords='offset points', color='red')
        ax.annotate(f'Second Max: {second_max_value}\n({file_name})', xy=(second_max_index, second_max_value),
                    xytext=(5, -15), textcoords='offset points', color='blue')
        ax.annotate(f'Min: {min_value}\n({file_name})', xy=(min_index, min_value), xytext=(5, -35),
                    textcoords='offset points', color='green')
    plt.tight_layout()
    plt.show()

def Dice_hist(root):
    file_paths = [os.path.join(root, i) for i in os.listdir(root)]
    dfs = []
    labels = []  # 存储文件基本名称

    for file_path in file_paths:
        df = pd.read_csv(file_path, sep='\s+')
        dfs.append(df)
        print(df)
        labels.append(os.path.basename(file_path).split('.')[0])

    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(16, 9))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 每个文件对应的柱状图颜色
    width = 0.3  # 柱状图的宽度
    xticks_labels = dfs[0]['methods']  # x轴标签

    x = [val for val in range(len(dfs[0]))]  # 初始化x轴位置

    for i, df in enumerate(dfs):
        iou_values = df['dice']
        ax.bar(x, iou_values, width, align='center', color=colors[i], label=f'{labels[i]} (dice)')
        x = [val + width for val in x.copy()]  # 创建新的列表以保持正确的横坐标位置

    ax.set_xticks([val + width for val in range(len(dfs[0]))])
    ax.set_xticklabels(xticks_labels, rotation=45, ha='right')
    ax.set_xlabel('Methods')
    ax.set_ylabel('Dice Scores')
    ax.set_title('Dice Performance Comparison')
    ax.legend()
    ax.set_ylim(0, 100)  # 设置y轴范围为0到100

    # 标记每个柱状图的最大值和最小值
    # 标记每个柱状图的最大值、第二大值和最小值
    for i, df in enumerate(dfs):
        values = df['dice']
        max_value = values.max()
        second_max_value = values.sort_values(ascending=False).iloc[1]
        min_value = values.min()
        max_index = values[values == max_value].index[0]
        second_max_index = values[values == second_max_value].index[0]
        min_index = values[values == min_value].index[0]
        file_name = labels[i]
        ax.annotate(f'Max: {max_value}\n({file_name})', xy=(max_index, max_value), xytext=(5, 5),
                    textcoords='offset points', color='red')
        ax.annotate(f'Second Max: {second_max_value}\n({file_name})', xy=(second_max_index, second_max_value),
                    xytext=(5, -15), textcoords='offset points', color='blue')
        ax.annotate(f'Min: {min_value}\n({file_name})', xy=(min_index, min_value), xytext=(5, -35),
                    textcoords='offset points', color='green')
    plt.tight_layout()
    plt.show()

def dice_line(root):
    file_paths = [os.path.join(root, i) for i in os.listdir(root)]
    dfs = []
    labels = []  # 存储文件基本名称

    for file_path in file_paths:
        df = pd.read_csv(file_path, sep='\s+')
        dfs.append(df)
        labels.append(os.path.basename(file_path).split('.')[0])

    # 绘制折线图
    fig, ax = plt.subplots(figsize=(16, 9))
    line_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 每个文件对应的折线图颜色
    xticks_labels = dfs[0]['methods']  # x轴标签

    num_methods = len(dfs[0])
    x = range(num_methods)  # 初始化x轴位置

    for i, df in enumerate(dfs):
        dice_values = df['dice']
        ax.plot(x, dice_values, color=line_colors[i], marker='o', linestyle='-', label=f'{labels[i]} (Dice)')

    ax.set_xticks(x)
    ax.set_xticklabels(xticks_labels, rotation=45, ha='right')
    ax.set_xlabel('Methods')
    ax.set_ylabel('Dice Scores')
    ax.set_title('Dice Performance Comparison')
    ax.legend()
    ax.set_ylim(50, 100)  # 设置y轴范围为0到100
    for i, df in enumerate(dfs):
        values = df['dice']
        max_value = values.max()
        second_max_value = values.sort_values(ascending=False).iloc[1]
        min_value = values.min()
        max_index = values[values == max_value].index[0]
        second_max_index = values[values == second_max_value].index[0]
        min_index = values[values == min_value].index[0]
        file_name = labels[i]
        ax.annotate(f'Max: {max_value}\n({file_name})', xy=(max_index, max_value), xytext=(5, 5),
                    textcoords='offset points', color='red')
        ax.annotate(f'Second Max: {second_max_value}\n({file_name})', xy=(second_max_index, second_max_value),
                    xytext=(5, -15), textcoords='offset points', color='blue')
        ax.annotate(f'Min: {min_value}\n({file_name})', xy=(min_index, min_value), xytext=(5, -35),
                    textcoords='offset points', color='green')
    plt.tight_layout()
    plt.show()

def iou_line(root):
    file_paths = [os.path.join(root, i) for i in os.listdir(root)]
    dfs = []
    labels = []  # 存储文件基本名称

    for file_path in file_paths:
        df = pd.read_csv(file_path, sep='\s+')
        dfs.append(df)
        labels.append(os.path.basename(file_path).split('.')[0])

    # 绘制折线图
    fig, ax = plt.subplots(figsize=(16, 9))
    line_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 每个文件对应的折线图颜色
    xticks_labels = dfs[0]['methods']  # x轴标签

    num_methods = len(dfs[0])
    x = range(num_methods)  # 初始化x轴位置

    for i, df in enumerate(dfs):
        iou_values = df['iou']
        ax.plot(x, iou_values, color=line_colors[i], marker='o', linestyle='-', label=f'{labels[i]} (iou)')

    ax.set_xticks(x)
    ax.set_xticklabels(xticks_labels, rotation=45, ha='right')
    ax.set_xlabel('Methods')
    ax.set_ylabel('iou Scores')
    ax.set_title('iou Performance Comparison')
    ax.legend()
    ax.set_ylim(50, 100)  # 设置y轴范围为0到100
    for i, df in enumerate(dfs):
        values = df['dice']
        max_value = values.max()
        second_max_value = values.sort_values(ascending=False).iloc[1]
        min_value = values.min()
        max_index = values[values == max_value].index[0]
        second_max_index = values[values == second_max_value].index[0]
        min_index = values[values == min_value].index[0]
        file_name = labels[i]
        ax.annotate(f'Max: {max_value}\n({file_name})', xy=(max_index, max_value), xytext=(5, 5),
                    textcoords='offset points', color='red')
        ax.annotate(f'Second Max: {second_max_value}\n({file_name})', xy=(second_max_index, second_max_value),
                    xytext=(5, -15), textcoords='offset points', color='blue')
        ax.annotate(f'Min: {min_value}\n({file_name})', xy=(min_index, min_value), xytext=(5, -35),
                    textcoords='offset points', color='green')

    plt.tight_layout()
    plt.show()

def specific_iou_line(root,key):
    file_paths = [os.path.join(root, i) for i in os.listdir(root)]
    dfs = []
    labels = []  # 存储文件基本名称

    for file_path in file_paths:
        df = pd.read_csv(file_path, sep='\s+')
        if df['methods'].str.contains(key).any():
            df_filtered = df[df['methods'].str.contains(key)]
            dfs.append(df_filtered)
            labels.append(os.path.basename(file_path).split('.')[0])

    # print(dfs)
    # 绘制折线图
    fig, ax = plt.subplots(figsize=(16, 9))
    line_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 每个文件对应的折线图颜色
    xticks_labels = dfs[0]['methods']  # x轴标签

    num_methods = len(dfs[0])
    x = range(num_methods)  # 初始化x轴位置

    for i, df in enumerate(dfs):
        iou_values = df['iou']
        ax.plot(x, iou_values, color=line_colors[i], marker='o', linestyle='-', label=f'{labels[i]} (iou)')

    ax.set_xticks(x)
    ax.set_xticklabels(xticks_labels, rotation=45, ha='right')
    ax.set_xlabel('Methods')
    ax.set_ylabel('iou Scores')
    ax.set_title('iou Performance Comparison')
    ax.legend()
    ax.set_ylim(50, 100)  # 设置y轴范围为0到100
    for i, df in enumerate(dfs):
        values = df['iou']
        max_value = values.max()
        second_max_value = values.sort_values(ascending=False).iloc[1]
        min_value = values.min()
        max_index = values[values == max_value].index[0]
        second_max_index = values[values == second_max_value].index[0]
        min_index = values[values == min_value].index[0]
        file_name = labels[i]
        ax.annotate(f'Max: {max_value}\n({file_name})', xy=(max_index, max_value), xytext=(5, 5),
                    textcoords='offset points', color='red')
        ax.annotate(f'Second Max: {second_max_value}\n({file_name})', xy=(second_max_index, second_max_value),
                    xytext=(5, -15), textcoords='offset points', color='blue')
        ax.annotate(f'Min: {min_value}\n({file_name})', xy=(min_index, min_value), xytext=(5, -35),
                    textcoords='offset points', color='green')

    plt.tight_layout()
    plt.show()

def specific_metrics_line(root,key):
    file_paths = [os.path.join(root, i) for i in os.listdir(root)]
    dfs = []
    labels = []  # 存储文件基本名称

    for file_path in file_paths:
        df = pd.read_csv(file_path, sep='\s+')
        if df['methods'].str.contains(key).any():
            df_filtered = df[df['methods'].str.contains(key)]
            dfs.append(df_filtered)
            labels.append(os.path.basename(file_path).split('.')[0])

    # print(dfs)
    # 绘制折线图
    fig, axs = plt.subplots(2, 1, figsize=(16, 16))

    # 绘制 IoU 折线图
    ax_iou = axs[0]
    line_colors_iou = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 每个文件对应的折线图颜色
    xticks_labels = dfs[0]['methods']  # x轴标签

    num_methods = len(dfs[0])
    x = range(num_methods)  # 初始化x轴位置

    for i, df in enumerate(dfs):
        iou_values = df['iou']
        ax_iou.plot(x, iou_values, color=line_colors_iou[i], marker='o', linestyle='-', label=f'{labels[i]} (IoU)')

    ax_iou.set_xticks(x)
    ax_iou.set_xticklabels(xticks_labels, rotation=45, ha='right')
    ax_iou.set_xlabel('Methods')
    ax_iou.set_ylabel('IoU Scores')
    ax_iou.set_title('IoU Performance Comparison')
    ax_iou.legend()
    ax_iou.set_ylim(50, 100)  # 设置y轴范围为50到100
    for i, df in enumerate(dfs):
        values = df['iou']
        max_value = values.max()
        second_max_value = values.sort_values(ascending=False).iloc[1]
        min_value = values.min()
        max_index = values[values == max_value].index[0]
        second_max_index = values[values == second_max_value].index[0]
        min_index = values[values == min_value].index[0]
        file_name = labels[i]
        ax_iou.annotate(f'Max: {max_value}\n({file_name})', xy=(max_index, max_value), xytext=(5, 0+i*30),
                        textcoords='offset points', color='red', arrowprops=dict(arrowstyle='->', color='red'))
        ax_iou.annotate(f'Second Max: {second_max_value}\n({file_name})', xy=(second_max_index, second_max_value),
                        xytext=(5, -15+i*30), textcoords='offset points', color='blue',
                        arrowprops=dict(arrowstyle='->', color='blue'))
        ax_iou.annotate(f'Min: {min_value}\n({file_name})', xy=(min_index, min_value), xytext=(5, -35+i*30),
                        textcoords='offset points', color='green', arrowprops=dict(arrowstyle='->', color='green'))

    # 绘制 Dice 折线图
    ax_dice = axs[1]
    line_colors_dice = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 每个文件对应的折线图颜色

    for i, df in enumerate(dfs):
        dice_values = df['dice']
        ax_dice.plot(x, dice_values, color=line_colors_dice[i], marker='o', linestyle='-', label=f'{labels[i]} (Dice)')

    ax_dice.set_xticks(x)
    ax_dice.set_xticklabels(xticks_labels, rotation=45, ha='right')
    ax_dice.set_xlabel('Methods')
    ax_dice.set_ylabel('Dice Scores')
    ax_dice.set_title('Dice Performance Comparison')
    ax_dice.legend()
    ax_dice.set_ylim(50, 100)  # 设置y轴范围为50到100
    for i, df in enumerate(dfs):
        values = df['dice']
        max_value = values.max()
        second_max_value = values.sort_values(ascending=False).iloc[1]
        min_value = values.min()
        max_index = values[values == max_value].index[0]
        second_max_index = values[values == second_max_value].index[0]
        min_index = values[values == min_value].index[0]
        file_name = labels[i]
        ax_dice.annotate(f'Max: {max_value}\n({file_name})', xy=(max_index, max_value), xytext=(5, 0+i*30),
                        textcoords='offset points', color='red', arrowprops=dict(arrowstyle='->', color='red'))
        ax_dice.annotate(f'Second Max: {second_max_value}\n({file_name})', xy=(second_max_index, second_max_value),
                        xytext=(5, -15+i*30), textcoords='offset points', color='blue',
                        arrowprops=dict(arrowstyle='->', color='blue'))
        ax_dice.annotate(f'Min: {min_value}\n({file_name})', xy=(min_index, min_value), xytext=(5, -35+i*30),
                        textcoords='offset points', color='green', arrowprops=dict(arrowstyle='->', color='green'))


    plt.tight_layout()
    plt.show()
# IoU_hist(results_path)
# Dice_hist(results_path)
# dice_line(results_path)
specific_metrics_line(results_path,'')