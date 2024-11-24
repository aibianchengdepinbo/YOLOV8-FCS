import cv2
import os
import random

def visualize_labels(image_dir, label_dir, output_dir):
    image_files = os.listdir(image_dir)
    label_files = os.listdir(label_dir)
    with open('class_neu-det.txt', 'r') as f:
        classes = f.read().splitlines()
    # class_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in classes]
    for label_file in label_files:
        if label_file.endswith(".txt"):
            # 读取标签文件
            label_path = os.path.join(label_dir, label_file)
            with open(label_path, "r") as f:
                lines = f.readlines()

            # 读取对应的图像文件
            image_file = label_file.replace(".txt", ".jpg")
            image_path = os.path.join(image_dir, image_file)
            if not os.path.exists(image_path):
                # 如果对应的图像文件不存在，跳过该文件
                continue

            # 绘制边界框
            image = cv2.imread(image_path)
            height, width, _ = image.shape
            for line in lines:
                line = line.strip().split()
                class_index = int(line[0])
                name = classes[class_index]
                x = float(line[1])
                y = float(line[2])
                w = float(line[3])
                h = float(line[4])

                # 将边界框坐标转换为图像坐标
                x1 = int((x - w / 2) * width)
                y1 = int((y - h / 2) * height)
                x2 = int((x + w / 2) * width)
                y2 = int((y + h / 2) * height)
                # color = class_colors[class_index]
                # 绘制边界框
                cv2.rectangle(image, (x1, y1), (x2, y2), (10, 249, 72), 2, lineType=cv2.LINE_AA)
                cv2.putText(image, str(name), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, lineType=cv2.LINE_AA)

            # 保存可视化结果到文件夹
            output_file = os.path.join(output_dir, image_file)
            cv2.imwrite(output_file, image)

# 图像文件夹路径
image_dir = r"D:\NEU-DET\detect images"
# 标签文件夹路径
label_dir = r"D:\NEU-DET\detect labels"
# 输出文件夹路径
output_dir = r"D:\NEU-DET\groundtruth result"

# 创建输出文件夹
os.makedirs(output_dir, exist_ok=True)

# 调用函数进行可视化并保存结果
visualize_labels(image_dir, label_dir, output_dir)