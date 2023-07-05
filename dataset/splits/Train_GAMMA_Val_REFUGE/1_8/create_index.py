# 读取路径REFUGE下的images和ground_truths
# 然后形成一个index.txt，其内容格式如下
# images/V0301.jpg ground_truths/g0001.bmp
import os

data_path = r"/media/ls/data/yzc/fundus_datasets/od_oc/WACV/REFUGE_cross_new/"
 # 数据路径
images_dir = os.path.join(data_path, 'images')  # 图像文件夹路径
ground_truths_dir = os.path.join(data_path, 'ground_truths')  # ground_truths文件夹路径
index_file =  'all.txt'  # index.txt文件路径

# 获取images文件夹下的文件列表
image_files = [file for file in os.listdir(images_dir) if file.endswith(('.JPG','jpg'))]


# 创建index.txt文件并写入内容
with open(index_file, 'w') as f:
    for image_file in image_files:
        image_path = os.path.join('images', image_file)
        ground_truth_path = image_path.replace('images','ground_truths').replace('.jpg','.bmp')
        f.write(f"{image_path} {ground_truth_path}\n")

print("Index file created successfully!")
