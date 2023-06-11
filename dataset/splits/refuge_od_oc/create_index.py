# 读取路径REFUGE下的images和ground_truths
# 然后形成一个index.txt，其内容格式如下
# images/V0301.jpg ground_truths/g0001.bmp
import os

data_path = r'F:/DL-Data/eyes/glaucoma_OD_OC/REFUGE/REFUGE'  # 数据路径
images_dir = os.path.join(data_path, 'images')  # 图像文件夹路径
ground_truths_dir = os.path.join(data_path, 'ground_truths')  # ground_truths文件夹路径
index_file = os.path.join(data_path, 'index.txt')  # index.txt文件路径

# 获取images文件夹下的文件列表
image_files = [file for file in os.listdir(images_dir) if file.endswith('.jpg')]

# 获取ground_truths文件夹下对应的文件列表
ground_truth_files = [file for file in os.listdir(ground_truths_dir) if file.endswith('.bmp')]

# 确保两个文件列表长度相同
if len(image_files) != len(ground_truth_files):
    print("Error: Number of image files and ground truth files mismatch!")
    exit(1)

# 创建index.txt文件并写入内容
with open(index_file, 'w') as f:
    for image_file, ground_truth_file in zip(image_files, ground_truth_files):
        image_path = os.path.join('images', image_file)
        ground_truth_path = os.path.join('ground_truths', ground_truth_file)
        f.write(f"{image_path} {ground_truth_path}\n")

print("Index file created successfully!")
