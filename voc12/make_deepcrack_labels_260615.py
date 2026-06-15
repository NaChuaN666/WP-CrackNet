import argparse
# import voc12.data
from .data import *
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_list", default='/root/autodl-tmp/ACR_used_boundary/WPCRACK/voc12/deepcrack/img_train.txt', type=str)
    parser.add_argument("--val_list", default='/root/autodl-tmp/ACR_used_boundary/WPCRACK/voc12/deepcrack/img_test.txt', type=str)
    parser.add_argument("--out", default="cls_deepcrack_labels.npy", type=str)
    # parser.add_argument("--voc12_root", required=True, type=str)
    parser.add_argument("--deepcrack_root", default="/root/autodl-tmp/ACR_used_boundary/WPCRACK/data/Deepcrack/", type=str)
    args = parser.parse_args()

    
    img_name_list = load_deepcrack_img_name_list(args.train_list)
    # print(img_name_list)
    print(img_name_list)
    
    img_name_list.extend(load_deepcrack_img_name_list(args.val_list))
    print(img_name_list)
    print(len(img_name_list))
    
    label_list = load_deepcrack_label_list_from_xml(img_name_list, args.deepcrack_root)
    # print(label_list)
    

    d = dict()
    for img_name, label in zip(img_name_list, label_list):
        d[img_name] = label
    
    
    # d:'2011_003197': array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0], dtype=float32)

    np.save(args.out, d)
    a = np.load(args.out, allow_pickle=True)
    print(a)
    print(type(a))
    print(a.shape)