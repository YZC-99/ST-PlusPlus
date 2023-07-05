## 1.强增强策略解析
下面的简称为：
- **CJ: ColorJitter**
- **RG: RandomGrayscale**
- **B: blur**
- **CO: cutout**
---
### 关于为何要在未标记数据上使用强增强的几点解释：
[《ST++: Make Self-training Work Better for Semi-supervised Semantic Segmentation》](https://ieeexplore.ieee.org/document/9880151)
- #### 缓解在噪声标签上的过拟合现象
```markdown
strong data augmentations (SDA) on unlabeled images to alleviate overfitting noisy labels
```
- #### 解耦教师网络和学生网络的预测相似性
```markdown
decouple similar predictions between the teacher and student
```
---
### 1.1 transforms.ColorJitter
是PyTorch中的一个图像增强类，用于对图像进行颜色调整。
```python
img = transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)(img)
```
参数说明：
- **brightness（float或元组)**:亮度调整的范围。默认值为0，表示不调整亮度。如果指定一个元组，则范围为(lower, upper)，表示亮度在[1 - lower, 1 + upper]之间进行随机调整。
- **contrast（float或元组)**:对比度调整的范围。默认值为0，表示不调整对比度。如果指定一个元组，则范围为(lower, upper)，表示对比度在[1 - lower, 1 + upper]之间进行随机调整。
- **saturation（float或元组)**:饱和度调整的范围。默认值为0，表示不调整饱和度。如果指定一个元组，则范围为(lower, upper)，表示饱和度在[1 - lower, 1 + upper]之间进行随机调整。
- **hue（float或元组)**:色调调整的范围。默认值为0，表示不调整色调。如果指定一个元组，则范围为(-hue, hue)，表示色调在[-hue, hue]之间进行随机调整。

### 1.2 transforms.RandomGrayscale
是PyTorch中的一个图像增强类，用于以一定的概率将图像转换为灰度图像。
```python
img = transforms.RandomGrayscale(p=0.2)(img)
```
参数说明：
- p=0.2代表有百分之20的概率将图片转为灰度图片

### 1.3 blur
用于对输入的图像应用高斯模糊。
```python
def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img
```
参数说明：
- **img**：输入的图像，可以是PIL.Image.Image对象或者具有相同接口的图像对象。
- **p**：高斯模糊应用的概率，默认为0.5。如果random.random()生成的随机数小于概率p，则应用高斯模糊；否则，直接返回原始图像。

函数逻辑：
- 通过random.random()生成一个随机数，用于确定是否应用高斯模糊。
- 如果随机数小于概率p，则执行以下步骤：
    - 生成一个随机的高斯模糊半径sigma，范围为0.1到2.0之间。
    - 使用ImageFilter.GaussianBlur滤波器将图像进行高斯模糊，半径为sigma。
- 返回处理后的图像。

### 1.4 cutout
用于在图像中应用随机的Cutout效果。
```python
def cutout(img, mask, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3,
           ratio_2=1/0.3, value_min=0, value_max=255, pixel_level=True):
    if random.random() < p:
        img = np.array(img)
        mask = np.array(mask)

        img_h, img_w, img_c = img.shape

        while True:
            size = np.random.uniform(size_min, size_max) * img_h * img_w
            ratio = np.random.uniform(ratio_1, ratio_2)
            erase_w = int(np.sqrt(size / ratio))
            erase_h = int(np.sqrt(size * ratio))
            x = np.random.randint(0, img_w)
            y = np.random.randint(0, img_h)

            if x + erase_w <= img_w and y + erase_h <= img_h:
                break

        if pixel_level:
            value = np.random.uniform(value_min, value_max, (erase_h, erase_w, img_c))
        else:
            value = np.random.uniform(value_min, value_max)

        img[y:y + erase_h, x:x + erase_w] = value
        mask[y:y + erase_h, x:x + erase_w] = 255

        img = Image.fromarray(img.astype(np.uint8))
        mask = Image.fromarray(mask.astype(np.uint8))

    return img, mask
```
参数说明：
- **img**：输入的图像，可以是PIL.Image.Image对象或者具有相同接口的图像对象。
- **mask**：与输入图像对应的掩膜图像，用于指示哪些区域需要进行Cutout操作。可以是PIL.Image.Image对象或者具有相同接口的图像对象。
- **p**：应用Cutout的概率，默认为0.5。如果random.random()生成的随机数小于概率p，则应用Cutout；否则，直接返回原始图像和掩膜。
- **size_min和size_max**：Cutout操作的区域大小的范围。默认范围为0.02到0.4，这两个参数用于计算Cutout操作的区域面积。
- **ratio_1和ratio_2**：Cutout操作的区域的宽高比范围。默认为0.3和1/0.3，这两个参数用于计算Cutout操作的区域宽高。
- **value_min和value_max**：Cutout操作填充区域的像素值范围。默认范围为0到255，这两个参数用于生成填充像素值。
- **pixel_level**：是否对图像进行像素级别的Cutout操作。默认为True，表示对图像的每个通道进行独立的像素级别Cutout操作。如果为False，则对整个区域进行统一的填充。
