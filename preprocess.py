import os
import pandas as pd
import copy
import argparse
import random
from random import shuffle
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='make dataloder')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('-t', '--table', type=str, default="./data/Table.xls", help='classify data based Table')
parser.add_argument('--dyskaryosis', type=int, default=1, help='with dyskaryosis')


def create_folder_dyskaryosis(args):
    # os.system("rm -rf ./data/binary_classification")
    # os.system("rm -rf ./data/multi_classification")

    dyskaryosis_set = set()
    for root, dirs, names in os.walk(os.path.join(args.data, "dyskaryosis")):
        for name in names:
            dyskaryosis_set.add(os.path.join(root, name))

    baomai_information = pd.read_excel(args.table)
    baomai_cancer_id = set(list(baomai_information[baomai_information["Table"]=="Cancer"]["病理号"]))
    baomai_normal_id = set(list(baomai_information[baomai_information["Table"]=="Heterogeneity"]["病理号"]))

    cancer_set = set()
    non_cancer_set = set()

    for item in dyskaryosis_set:
        img_name = item.split("/")[-1][:9]
        if img_name in baomai_cancer_id:
            cancer_set.add(item)
        elif img_name in baomai_normal_id:
            non_cancer_set.add(item)
        else:
            print("Unknown:", img_name)
    
    print("&:", len(cancer_set & non_cancer_set))
    
    non_cancer_set = list(non_cancer_set)
    cancer_set = list(cancer_set)

    shuffle(non_cancer_set)
    shuffle(cancer_set)

    print("non_cancer & cancer:", len(set(non_cancer_set)&set(cancer_set)))

    train_non_cancer_set = non_cancer_set[:int(len(non_cancer_set) * 0.7)]
    val_non_cancer_set = non_cancer_set[int(len(non_cancer_set) * 0.7):]
    train_cancer_set = cancer_set[:int(len(cancer_set) * 0.7)]
    val_cancer_set = cancer_set[int(len(cancer_set) * 0.7):]
    print("number of train non-cancer:", len(train_non_cancer_set))
    print("number of val non-cancer:", len(val_non_cancer_set))
    print("number of train cancer:", len(train_cancer_set))
    print("number of val cancer:", len(val_cancer_set))

    for i in train_non_cancer_set:
        os.system("cp " + i + " ./data/binary_classification_dyskaryosis/train/non-cancer/" + i.split("/")[-1])
    for i in val_non_cancer_set:
        os.system("cp " + i + " ./data/binary_classification_dyskaryosis/val/non-cancer/" + i.split("/")[-1])

    for i in train_cancer_set:
        os.system("cp " + i + " ./data/binary_classification_dyskaryosis/train/cancer/" + i.split("/")[-1])
    for i in val_cancer_set:
        os.system("cp " + i + " ./data/binary_classification_dyskaryosis/val/cancer/" + i.split("/")[-1])


def create_folder(args):
    # os.system("rm -rf ./data/binary_classification")
    # os.system("rm -rf ./data/multi_classification")

    original_cancer_set = set()
    for root, dirs, names in os.walk(os.path.join(args.data, "cancer")):
        for name in names:
            original_cancer_set.add(os.path.join(root, name))

    dyskaryosis_set = set()
    for root, dirs, names in os.walk(os.path.join(args.data, "dyskaryosis")):
        for name in names:
            dyskaryosis_set.add(os.path.join(root, name))

    normal_set = set()
    for root, dirs, names in os.walk(os.path.join(args.data, "normal")):
        for name in names:
            normal_set.add(os.path.join(root, name))

    baomai_information = pd.read_excel(args.table)
    baomai_cancer_id = set(list(baomai_information[baomai_information["Table"]=="Cancer"]["病理号"]))
    baomai_normal_id = set(list(baomai_information[baomai_information["Table"]=="Heterogeneity"]["病理号"]))
    baomai_exclude_id = set(list(baomai_information[baomai_information["Table"]=="Exclude"]["病理号"]))

    cancer_set = set()
    non_cancer_set = set()

    for item in original_cancer_set:
        img_name = item.split("/")[-1][:9]
        if img_name in baomai_cancer_id:
            cancer_set.add(item)
        elif img_name in baomai_normal_id:
            non_cancer_set.add(item)
        elif img_name in baomai_exclude_id:
            print("original cancer set delete:", img_name)
        else:
            cancer_set.add(item)
    
    for item in dyskaryosis_set:
        img_name = item.split("/")[-1][:9]
        if img_name in baomai_cancer_id:
            cancer_set.add(item)
        elif img_name in baomai_normal_id:
            non_cancer_set.add(item)
        elif img_name in baomai_exclude_id:
            print("original dyskaryosis set delete:", img_name)
    
    for item in normal_set:
        img_name = item.split("/")[-1][:9]
        if img_name in baomai_cancer_id:
            cancer_set.add(item)
        elif img_name in baomai_normal_id:
            non_cancer_set.add(item)
        elif img_name in baomai_exclude_id:
            print("original normal set delete:", img_name)
        else:
            non_cancer_set.add(item)
    
    print("&:", len(cancer_set & non_cancer_set))
    
    non_cancer_set = list(non_cancer_set)
    cancer_set = list(cancer_set)

    shuffle(non_cancer_set)
    shuffle(cancer_set)

    print("non_cancer & cancer:", len(set(non_cancer_set)&set(cancer_set)))

    train_non_cancer_set = non_cancer_set[:int(len(non_cancer_set) * 0.7)]
    val_non_cancer_set = non_cancer_set[int(len(non_cancer_set) * 0.7):]
    train_cancer_set = cancer_set[:int(len(cancer_set) * 0.7)]
    val_cancer_set = cancer_set[int(len(cancer_set) * 0.7):]
    print("number of train non-cancer:", len(train_non_cancer_set))
    print("number of val non-cancer:", len(val_non_cancer_set))
    print("number of train cancer:", len(train_cancer_set))
    print("number of val cancer:", len(val_cancer_set))

    for i in train_non_cancer_set:
        os.system("cp " + i + " ./data/binary_classification/train/non-cancer/" + i.split("/")[-1])
    for i in val_non_cancer_set:
        os.system("cp " + i + " ./data/binary_classification/val/non-cancer/" + i.split("/")[-1])

    for i in train_cancer_set:
        os.system("cp " + i + " ./data/binary_classification/train/cancer/" + i.split("/")[-1])
    for i in val_cancer_set:
        os.system("cp " + i + " ./data/binary_classification/val/cancer/" + i.split("/")[-1])
    

def create_folder_without_dyskaryosis(args):
    # os.system("rm -rf ./data/binary_classification")
    # os.system("rm -rf ./data/multi_classification")

    original_cancer_set = set()
    for root, dirs, names in os.walk(os.path.join(args.data, "cancer")):
        for name in names:
            original_cancer_set.add(os.path.join(root, name))

    normal_set = set()
    for root, dirs, names in os.walk(os.path.join(args.data, "normal")):
        for name in names:
            normal_set.add(os.path.join(root, name))

    baomai_information = pd.read_excel(args.table)
    baomai_exclude_id = set(list(baomai_information[baomai_information["Table"]=="Exclude"]["病理号"]))

    cancer_set = set()
    non_cancer_set = set()

    for item in original_cancer_set:
        img_name = item.split("/")[-1][:9]
        if img_name in baomai_exclude_id:
            print("original cancer set delete:", img_name)
        else:
            cancer_set.add(item)
    
    for item in normal_set:
        img_name = item.split("/")[-1][:9]
        if img_name in baomai_exclude_id:
            print("original normal set delete:", img_name)
        else:
            non_cancer_set.add(item)
    
    print("&:", len(cancer_set & non_cancer_set))
    
    non_cancer_set = list(non_cancer_set)
    cancer_set = list(cancer_set)

    shuffle(non_cancer_set)
    shuffle(cancer_set)

    print("non_cancer & cancer:", len(set(non_cancer_set)&set(cancer_set)))

    train_non_cancer_set = non_cancer_set[:int(len(non_cancer_set) * 0.7)]
    val_non_cancer_set = non_cancer_set[int(len(non_cancer_set) * 0.7):]
    train_cancer_set = cancer_set[:int(len(cancer_set) * 0.7)]
    val_cancer_set = cancer_set[int(len(cancer_set) * 0.7):]
    print("number of train non-cancer:", len(train_non_cancer_set))
    print("number of val non-cancer:", len(val_non_cancer_set))
    print("number of train cancer:", len(train_cancer_set))
    print("number of val cancer:", len(val_cancer_set))

    for i in train_non_cancer_set:
        os.system("cp " + i + " ./data/binary_classification_without_dyskaryosis/train/non-cancer/" + i.split("/")[-1])
    for i in val_non_cancer_set:
        os.system("cp " + i + " ./data/binary_classification_without_dyskaryosis/val/non-cancer/" + i.split("/")[-1])

    for i in train_cancer_set:
        os.system("cp " + i + " ./data/binary_classification_without_dyskaryosis/train/cancer/" + i.split("/")[-1])
    for i in val_cancer_set:
        os.system("cp " + i + " ./data/binary_classification_without_dyskaryosis/val/cancer/" + i.split("/")[-1])
    

if __name__=="__main__":
    args = parser.parse_args()
    if args.dyskaryosis == 1:
        create_folder(args)
    elif args.dyskaryosis == 0:
        create_folder_without_dyskaryosis(args)
    elif args.dyskaryosis == 2:
        create_folder_dyskaryosis(args)