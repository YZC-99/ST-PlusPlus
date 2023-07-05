# 读取路径REFUGE下的images和ground_truths
# 然后形成一个index.txt，其内容格式如下
# images/V0301.jpg ground_truths/g0001.bmp
import os

data_path = r'../data/fundus_datasets/od_oc/WACV/REFUGE_cross_new/ORIGA'  # 数据路径
images_dir = os.path.join(data_path, '650image')  # 图像文件夹路径
save_path = '../dataset/splits/sup/ORIGA650_labeled.txt'
ground_truths_dir = os.path.join(data_path, 'my_gts')  # ground_truths文件夹路径

# 获取images文件夹下的文件列表
image_files = [file for file in os.listdir(images_dir) if file.endswith('.jpg')]

# 获取ground_truths文件夹下对应的文件列表
ground_truth_files = [file for file in os.listdir(ground_truths_dir) if file.endswith('.jpg')]

# 确保两个文件列表长度相同
if len(image_files) != len(ground_truth_files):
    print("Error: Number of image files and ground truth files mismatch!")
    exit(1)

# 创建index.txt文件并写入内容
with open(save_path, 'w') as f:
    for image_file in image_files:
        image_path = os.path.join('ORIGA/650image', image_file)
        # ground_truth_path = os.path.join('ORIGA/my_gts', ground_truth_file)
        ground_truth_path = image_path.replace('650image','650mask').replace('jpg','mat')
        f.write(f"{image_path} {ground_truth_path}\n")

print("Index file created successfully!")
