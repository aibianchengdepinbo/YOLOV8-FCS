import matplotlib.pyplot as plt

img_path = 'inclusion_1_layer14_my/0.png'
mask_path = 'D:/NEU-DET/images/inclusion_1.jpg'

img = plt.imread(img_path)
mask = plt.imread(mask_path)

# 叠加显示img, mask
plt.imshow(img)
plt.imshow(mask, alpha=0.4, cmap='rainbow')  # alpha设置透明度, cmap可以选择颜色

plt.imshow()