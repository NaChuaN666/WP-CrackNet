import os
import numpy as np
import matplotlib.pyplot as plt
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

# 源文件夹路径和目标文件夹路径
# source_folder = 'E:/学习资料/科研/复现/AEFT_crf/dict'  # 替换为你的源文件夹路径
# image_folder = 'E:/学习资料/科研/复现/AEFT_crf/test'  # 替换为你的图像文件夹路径
# target_folder = 'E:/学习资料/科研/复现/AEFT_crf/label'  # 替换为你的目标文件夹路径

source_folder = './results/0306_deepcrack_bs4_model9_grid/cam_multi'  # 替换为你的源文件夹路径
image_folder = './data/Deepcrack/train_img_300_p256'  # 替换为你的图像文件夹路径
target_folder = './results/0306_deepcrack_bs4_model9_grid/cam2crf_multi'  # 替换为你的目标文件夹路径


# 创建目标文件夹（如果不存在）
os.makedirs(target_folder, exist_ok=True)

# CRF 参数
alpha = 0.5  # 背景得分的幂运算参数
t = 10  # CRF 推理的迭代次数
scale_factor = 1  # 缩放因子
labels = 2  # 类别数量

# CRF 推理函数
def crf_inference_ysh(img, probs, t=10, scale_factor=1, labels=1):
    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
    
    # 确保 img 的数据类型是 uint8
    
    img_uint8 = (img * 255).astype(np.uint8)
    
    print_option = False
    if print_option:
        print(img_uint8.shape) # (H,W,C)
        exit()

    d.addPairwiseBilateral(sxy=80/scale_factor, srgb=13, rgbim=img_uint8, compat=10)

    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))

# 自适应引导 CRF 推理函数
def crf_inference_adaptive_ysh(img, probs, confidence_map, t=10, scale_factor=1, labels=1):
    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)
    d.setUnaryEnergy(unary)

    # 添加 Gaussian pairwise term (不变)
    d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)

    # 确保 img 是 uint8
    img_uint8 = (img * 255).astype(np.uint8)

    # 计算 confidence map 的全局均值
    confidence_mean = np.mean(confidence_map)

    # 动态调整 bilateral compat
    base_compat = 10
    alpha = 2.0  # 调整幅度因子
    adaptive_compat = base_compat * (1 + alpha * (confidence_mean - 0.5))

    # 防止compat过小或过大
    adaptive_compat = np.clip(adaptive_compat, 5, 20)

    d.addPairwiseBilateral(sxy=80/scale_factor, srgb=13, rgbim=img_uint8, compat=int(adaptive_compat))

    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))

if __name__ == '__main__':
    # 遍历源文件夹中的所有 .npy 文件
    for filename in os.listdir(source_folder):
        if filename.endswith('.npy'):
            # 构建完整的 .npy 文件路径
            file_path = os.path.join(source_folder, filename)
            
            # 读取 .npy 文件，允许加载包含 Python 对象的数组
            cam_dict = np.load(file_path, allow_pickle=True)
            print(f"Processing file: {file_path}")

            # 检查 cam_dict.item()[1] 是否存在
            if 1 in cam_dict.item():
                cam_dict = cam_dict.item()[1]
                v = cam_dict.reshape(1, 65536)
                
                # 计算背景得分
                bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
                
                # 合并背景得分和CAM值
                bgcam_score = np.concatenate((bg_score, v), axis=0)
                
                # 构建对应的 .png 文件路径
                image_filename = filename.replace('.npy', '.png')
                image_path = os.path.join(image_folder, image_filename)
                
                # 检查图像文件是否存在
                if not os.path.exists(image_path):
                    # 构建对应的 .jpg 文件路径
                    image_filename = filename.replace('.npy', '.jpg')
                    image_path = os.path.join(image_folder, image_filename)
                
                if not os.path.exists(image_path):
                    print(f"Image file not found: {image_path}")
                    continue
                
                # 读取原始图像
                img = plt.imread(image_path)
                


                # 进行CRF推理
                crf_score = crf_inference_ysh(img, bgcam_score, t=t, scale_factor=scale_factor, labels=labels)
                
                # 构建目标文件路径
                target_file_path = os.path.join(target_folder, filename.replace('.npy', '.png'))
                
                # 使用 matplotlib 保存图像
                plt.imsave(target_file_path, crf_score.argmax(axis=0), cmap='gray')  # 保存为灰度图像
            else:
                print(f"Key 1 not found in {file_path}, skipping this file.")

    print("转换完成！")