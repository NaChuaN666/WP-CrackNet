import os
import os.path as osp
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import multiprocessing
import argparse
import pdb
import torch
from img import crf_inference_ysh

################################################################################
# Evaluate the performance by computing mIoU.
# It assumes that every CAM or CRF dict file is already infered and saved.  
# For CAM, threshold will be searched in range [0.01, 0.80].
#
# If you want to evaluate CAM performance...
# python evaluation.py --name [exp_name] --task cam --dict_dir dict
#
# Or if you want to evaluate CRF performance of certain alpha (let, a1)...
# python evaluation.py --name [exp_name] --task crf --dict_dir crf/a1
#
# For AFF evaluation, go to evaluation_aff.py
################################################################################


categories = ['road', 'crack']
image_folder = './data/Deepcrack/images_all'  # 替换为你的图像文件夹路径

def do_python_eval(predict_folder, gt_folder, name_list, num_cls, task, threshold, printlog=False):
    TP = []
    P = []
    T = []
    # for i in range(num_cls):
    #     TP.append(torch.multiprocessing.Value('i', 0, lock=True))
    #     P.append(torch.multiprocessing.Value('i', 0, lock=True))
    #     T.append(torch.multiprocessing.Value('i', 0, lock=True))

    # 初始化普通的 Python 列表
    TP = [0] * num_cls  # True Positives
    P = [0] * num_cls   # Predicted Positives
    T = [0] * num_cls   # Actual Positives

    for i in range(num_cls):
        TP[i] += 1  # 根据需要更新值
        P[i] += 1
        T[i] += 1

    
    def compare(start,step,TP,P,T,task,threshold):
        for idx in range(start,len(name_list),step):
            name = name_list[idx]

            if task=='cam':
                predict_file = os.path.join(predict_folder,'%s.npy'%name)
                # print(predict_file)
                predict_dict = np.load(predict_file, allow_pickle=True).item()
                # for pre_key in predict_dict:
                    # print(pre_key)
                # print("predict_dict is {}".format(predict_dict))
                h, w = list(predict_dict.values())[0].shape
                # 3 or 2 ??!
                tensor = np.zeros((2,h,w),np.float32)
                for key in predict_dict.keys():
                    # tensor[key+1] = predict_dict[key]
                    # for crack:
                    tensor[key] = predict_dict[key]
                    
                tensor[0,:,:] = threshold 
                predict = np.argmax(tensor, axis=0).astype(np.uint8)

            gt_file = os.path.join(gt_folder,'%s.png'%name)
            gt = np.array(Image.open(gt_file))
            gt = gt/ 255
            # print("gt.max is :", np.max(gt))
            # exit()
            cal = gt<255 # Reject object boundary
            mask = (predict==gt) * cal
      
            for i in range(num_cls):
                # P[i].acquire()
                # P[i].value += np.sum((predict==i)*cal)
                # P[i].release()
                # T[i].acquire()
                # T[i].value += np.sum((gt==i)*cal)
                # T[i].release()
                # TP[i].acquire()
                # TP[i].value += np.sum((gt==i)*mask)
                # TP[i].release()

                # debug by nachuan:
                P[i] += np.sum((predict == i) * cal)  # 预测为 i 类的像素总数
                T[i] += np.sum((gt == i) * cal)       # 实际的 i 类像素总数
                TP[i] += np.sum((gt == i) * mask)     # 预测为 i 类且预测正确的像素数
                
    # p_list = []
    # for i in range(8):
    #     p = torch.multiprocessing.Process(target=compare, args=(i,8,TP,P,T,task,threshold))
    #     p.start()
    #     p_list.append(p)
    # for p in p_list:
    #     p.join()

    for i in range(8):  # 原来的任务划分循环保留
        compare(i, 8, TP, P, T, task, threshold)
   
    precision = []
    recall = []
    # F = []
    IoU = []
    for i in range(num_cls):
        # precision.append(TP[i].value/(P[i].value+1e-10))
        # recall.append(TP[i].value/(T[i].value+1e-10))
        # IoU.append(TP[i].value/(T[i].value+P[i].value-TP[i].value+1e-10))
        # debug by nachuan:
        precision.append(TP[i]/(P[i]+1e-10))
        recall.append(TP[i]/(T[i]+1e-10))
        IoU.append(TP[i]/(T[i]+P[i]-TP[i]+1e-10))

        # T_TP.append(T[i].value/(TP[i].value+1e-10))
        # P_TP.append(P[i].value/(TP[i].value+1e-10))
        # FP_ALL.append((P[i].value-TP[i].value)/(T[i].value + P[i].value - TP[i].value + 1e-10))
        # FN_ALL.append((T[i].value-TP[i].value)/(T[i].value + P[i].value - TP[i].value + 1e-10))
    
    loglist = {}
    # for i in range(num_cls):
    # #     loglist[categories[i]] = precision[i] * 100
    # for i in range(num_cls):
    #     F.append(2*precision[i]*recall[i]/(precision[i]+recall[i]))

    miou = np.mean(np.array(IoU))
    mp = np.mean(np.array(precision))
    mr = np.mean(np.array(recall))
    # mf = np.mean(np.array(F))
    
    loglist['mIoU'] = miou * 100
    loglist['mP'] = mp
    loglist['mR'] = mr
    loglist['IoU_0'] = IoU[0]
    loglist['IoU_1'] = IoU[1]
    # loglist['mF'] = mf

    if printlog:
        for i in range(num_cls):
            if i%2 != 1:
                print('%11s:%7.3f%%'%(categories[i],precision[i]*100),end='\t')
            else:
                print('%11s:%7.3f%%'%(categories[i],precision[i]*100))
        print('\n======================================================')
        print('%11s:%7.3f%%'%('mIoU',miou*100))
    return loglist


def do_python_eval_CRF(predict_folder, gt_folder, name_list, num_cls, task, threshold, printlog=False):
    # CRF 参数
    alpha = 0.5  # 背景得分的幂运算参数
    t_crf = 10  # CRF 推理的迭代次数
    scale_factor = 1  # 缩放因子
    labels = 2  # 类别数量

    TP = []
    P = []
    T = []
   
    # 初始化普通的 Python 列表
    TP = [0] * num_cls  # True Positives
    P = [0] * num_cls   # Predicted Positives
    T = [0] * num_cls   # Actual Positives

    for i in range(num_cls):
        TP[i] += 1  # 根据需要更新值
        P[i] += 1
        T[i] += 1
    
    def compare(start,step,TP,P,T,task,threshold):
        for idx in range(start,len(name_list),step):
            name = name_list[idx]

            if task=='cam':
                predict_file = os.path.join(predict_folder,'%s.npy'%name)
                # print(predict_file)
                # 读取 .npy 文件，允许加载包含 Python 对象的数组
                # predict_dict = np.load(predict_file, allow_pickle=True).item()
                predict_dict = np.load(predict_file, allow_pickle=True)

                # 检查 cam_dict.item()[1] 是否存在
                if 1 in predict_dict.item():
                    predict_dict = predict_dict.item()[1]
                    v = predict_dict.reshape(1, 65536)
                else:
                    print(f"Key 1 not found in {predict_file}, skipping this file.")

                # 计算背景得分
                bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)

                # 合并背景得分和CAM值
                bgcam_score = np.concatenate((bg_score, v), axis=0)

                # 构建对应的 .png 文件路径
                image_file = os.path.join(image_folder, '%s.png'%name)
                 # 检查图像文件是否存在
                if not os.path.exists(image_file):
                    print(f"Image file not found: {image_file}")
                    continue
                
                # 读取原始图像
                img = plt.imread(image_file)
                # 进行CRF推理
                crf_score = crf_inference_ysh(img, bgcam_score, t=t_crf, scale_factor=scale_factor, labels=labels)
                predict = crf_score.argmax(axis=0).astype(np.uint8)
                count_of_ones = np.sum(predict == 1)
                # print("predict 中值为 1 的个数:", count_of_ones)
                # exit()

            gt_file = os.path.join(gt_folder,'%s.png'%name)
            gt = np.array(Image.open(gt_file))
            gt = gt/ 255
            cal = gt<255 # Reject object boundary
            mask = (predict==gt) * cal
      
            for i in range(num_cls):
                # debug by nachuan:
                P[i] += np.sum((predict == i) * cal)  # 预测为 i 类的像素总数
                T[i] += np.sum((gt == i) * cal)       # 实际的 i 类像素总数
                TP[i] += np.sum((gt == i) * mask)     # 预测为 i 类且预测正确的像素数

    for i in range(8):  # 原来的任务划分循环保留
        compare(i, 8, TP, P, T, task, threshold)
   
    precision = []
    recall = []
    IoU = []
    for i in range(num_cls):
        # debug by nachuan:
        precision.append(TP[i]/(P[i]+1e-10))
        recall.append(TP[i]/(T[i]+1e-10))
        IoU.append(TP[i]/(T[i]+P[i]-TP[i]+1e-10))
    
    loglist = {}

    miou = np.mean(np.array(IoU))
    mp = np.mean(np.array(precision))
    mr = np.mean(np.array(recall))
    
    loglist['mIoU'] = miou * 100
    loglist['mP'] = mp
    loglist['mR'] = mr
    loglist['IoU_0'] = IoU[0]
    loglist['IoU_1'] = IoU[1]

    if printlog:
        for i in range(num_cls):
            if i%2 != 1:
                print('%11s:%7.3f%%'%(categories[i],precision[i]*100),end='\t')
            else:
                print('%11s:%7.3f%%'%(categories[i],precision[i]*100))
        print('\n======================================================')
        print('%11s:%7.3f%%'%('mIoU',miou*100))
    return loglist


def do_python_eval_seg(predict_folder, gt_folder, name_list, num_cls, task, threshold, printlog=False):
    # CRF 参数
    alpha = 0.5  # 背景得分的幂运算参数
    t_crf = 10  # CRF 推理的迭代次数
    scale_factor = 1  # 缩放因子
    labels = 2  # 类别数量

    TP = []
    P = []
    T = []
   
    # 初始化普通的 Python 列表
    TP = [0] * num_cls  # True Positives
    P = [0] * num_cls   # Predicted Positives
    T = [0] * num_cls   # Actual Positives

    for i in range(num_cls):
        TP[i] += 1  # 根据需要更新值
        P[i] += 1
        T[i] += 1
    
    def compare(start,step,TP,P,T,task,threshold):
        for idx in range(start,len(name_list),step):
            name = name_list[idx]

            predict_file = os.path.join(predict_folder,'%s.png'%name)
            # print(predict_file)
            predict = np.array(Image.open(predict_file).convert('L'))
            # print("predict.shape is {}".format(predict))
            # print("predict.shape is {}".format(predict.shape))
            predict = predict/ 255

            gt_file = os.path.join(gt_folder,'%s.png'%name)
            gt = np.array(Image.open(gt_file))
            gt = gt/ 255
            cal = gt<255 # Reject object boundary
            mask = (predict==gt) * cal

            # print(predict.shape)
            # print("....")
            # print(gt.shape)
            # exit()

            for i in range(num_cls):
                # debug by nachuan:
                P[i] += np.sum((predict == i) * cal)  # 预测为 i 类的像素总数
                T[i] += np.sum((gt == i) * cal)       # 实际的 i 类像素总数
                TP[i] += np.sum((gt == i) * mask)     # 预测为 i 类且预测正确的像素数

    for i in range(8):  # 原来的任务划分循环保留
        compare(i, 8, TP, P, T, task, threshold)
   
    precision = []
    recall = []
    IoU = []
    for i in range(num_cls):
        # debug by nachuan:
        precision.append(TP[i]/(P[i]+1e-10))
        recall.append(TP[i]/(T[i]+1e-10))
        IoU.append(TP[i]/(T[i]+P[i]-TP[i]+1e-10))
    
    loglist = {}

    miou = np.mean(np.array(IoU))
    mp = np.mean(np.array(precision))
    mr = np.mean(np.array(recall))
    
    loglist['mIoU'] = miou * 100
    loglist['mP'] = mp
    loglist['mR'] = mr
    loglist['IoU_0'] = IoU[0]
    loglist['IoU_1'] = IoU[1]

    if printlog:
        for i in range(num_cls):
            if i%2 != 1:
                print('%11s:%7.3f%%'%(categories[i],precision[i]*100),end='\t')
            else:
                print('%11s:%7.3f%%'%(categories[i],precision[i]*100))
        print('\n======================================================')
        print('%11s:%7.3f%%'%('mIoU',miou*100))
    return loglist



def eval_in_script(logger=None, eval_list='train', task='cam', pred_dir=None, gt_dir='./data/Deepcrack/all_labels'):
    
    eval_list = './data/Deepcrack/label_folder/' + eval_list + '.txt'
    df = pd.read_csv(eval_list, names=['filename'])
    name_list = df['filename'].values
    # print(name_list)
    # print("Number of items in name_list:", len(name_list))   # 226

    max_miou = 0
    max_th = 0
    # for i in range(20):
    for i in range(100):
        # t = i/100.0+0.15
        t = i/100.0 # (0-0.99)
        # print("t is {}".format(t))
        loglist = do_python_eval(pred_dir, gt_dir, name_list, 2, task, t, printlog=False)
        
        miou_temp = loglist['mIoU']
        # print("miou_temp is {}".format(miou_temp))
        
        if miou_temp>max_miou:
            max_miou = miou_temp
            max_th = t
            precision = loglist['mP']
            recall = loglist['mR']
            IoU_0 = loglist['IoU_0']
            IoU_1 = loglist['IoU_1']

    ret_dict = {}
    ret_dict['th'] = max_th
    ret_dict['miou'] = max_miou
    ret_dict['mp'] = precision
    ret_dict['mr'] = recall
    ret_dict['IoU_0'] = IoU_0 
    ret_dict['IoU_1'] = IoU_1 

    return ret_dict

def eval_in_script_CRF(logger=None, eval_list='train', task='cam', pred_dir=None, gt_dir='./data/Deepcrack/all_labels'):
    
    # eval_list: infer_226
    # pred_dir=tb.log_dir+'/dict'
    # gt_dir='./data/Deepcrack/all_labels'
    eval_list = './voc12/deepcrack/' + eval_list + '.txt'
    df = pd.read_csv(eval_list, names=['filename'])
    name_list = df['filename'].values
    # print(name_list)
    # print("Number of items in name_list:", len(name_list))   # 226

    t = 0.5
    loglist = do_python_eval_CRF(pred_dir, gt_dir, name_list, 2, task, t, printlog=False)
    miou_temp = loglist['mIoU']
    max_miou = miou_temp
    max_th = t
    precision = loglist['mP']
    recall = loglist['mR']
    IoU_0 = loglist['IoU_0']
    IoU_1 = loglist['IoU_1']


    ret_dict = {}
    ret_dict['th'] = max_th
    ret_dict['miou'] = max_miou
    ret_dict['mp'] = precision
    ret_dict['mr'] = recall
    ret_dict['IoU_0'] = IoU_0 
    ret_dict['IoU_1'] = IoU_1 
    return ret_dict

def eval_seg(logger=None, eval_list='train', task='cam', pred_dir=None, gt_dir='./data/Deepcrack/all_labels'):
    
    # eval_list: infer_226
    # pred_dir=tb.log_dir+'/seg_output'
    # gt_dir='./data/Deepcrack/all_labels'
    eval_list = './voc12/deepcrack/' + eval_list + '.txt'
    df = pd.read_csv(eval_list, names=['filename'])
    name_list = df['filename'].values
    # print(name_list)
    # print("Number of items in name_list:", len(name_list))   # 226

    t = 0.5
    # loglist = do_python_eval_CRF(pred_dir, gt_dir, name_list, 2, task, t, printlog=False)
    loglist = do_python_eval_seg(pred_dir, gt_dir, name_list, 2, task, t, printlog=False)
    miou_temp = loglist['mIoU']
    max_miou = miou_temp
    max_th = t
    precision = loglist['mP']
    recall = loglist['mR']
    IoU_0 = loglist['IoU_0']
    IoU_1 = loglist['IoU_1']

    ret_dict = {}
    ret_dict['th'] = max_th
    ret_dict['miou'] = max_miou
    ret_dict['mp'] = precision
    ret_dict['mr'] = recall
    ret_dict['IoU_0'] = IoU_0 
    ret_dict['IoU_1'] = IoU_1 
    return ret_dict

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--list", default="train", type=str)
    parser.add_argument("--task", required=True, type=str)
    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--dict_dir", required=True, type=str)
    parser.add_argument("--gt_dir", default='./data/VOC2012/SegmentationClass', type=str)
    
    args = parser.parse_args()

    eval_list = './data/VOC2012/ImageSets/Segmentation/' + args.list + '.txt'
    df = pd.read_csv(eval_list, names=['filename'])
    name_list = df['filename'].values

    pred_dir = osp.join(args.dict_dir)  
    print('Evaluate ' + pred_dir + ' with ' + eval_list)

    if args.task=='cam':
        for i in range(10):
            t = i/100.+0.25
            loglist = do_python_eval(pred_dir, args.gt_dir, name_list, 3, args.task, t, printlog=False)
            print(loglist)
            # print('%d/60 threshold: %.3f\tmIoU: %.3f \tmP: %.3f \tmR: %.3f \tmF: %.3f%%'%(i, t, loglist['mIoU'], loglist['mP'], loglist['mR'], loglist['mF']))
    
    elif args.task=='crf':
        loglist = do_python_eval(pred_dir, args.gt_dir, name_list, 3, args.task, 0, printlog=True)

    elif args.task=='dl'or args.task=='png':
        loglist = do_python_eval(pred_dir, args.gt_dir, name_list, 3, args.task, 0, printlog=True)
