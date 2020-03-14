import argparse
import os
import copy
import random
import shutil
import time
import csv
from random import shuffle
import warnings
from PIL import Image
import cv2
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_auc_score, f1_score, recall_score, confusion_matrix, precision_score, accuracy_score, roc_curve, auc


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning_rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-e', '--evaluate', default=False, type=bool, help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
parser.add_argument('--arch', metavar='ARCH', default='googlenet')
parser.add_argument('--image_size', type=int, default=299)
parser.add_argument('--model_path', default="./model/model.pth", type=str)


def draw_roc(ground_truth, p_proba, args):
    fpr,tpr,threshold = roc_curve(ground_truth, p_proba)
    roc_auc = auc(fpr,tpr)

    plt.figure()
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example - ' + args.data.split("/")[-2] + "-" +  args.data.split("/")[-1] +"-" + args.model_path.split("/")[-1])
    plt.legend(loc="lower right")
    plt.savefig("./result/" + args.data.split("/")[-2] + "-" +  args.data.split("/")[-1] +"-" + args.model_path.split("/")[-1] + ".png")


def get_dataloader(args):
    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize(600),
            transforms.RandomRotation(degrees=15),
            transforms.RandomResizedCrop(args.image_size),
            # transforms.ColorJitter(brightness=(0, 36), contrast=(0, 10), saturation=(0, 25), hue=(-0.5, 0.5)),
            # transforms.RandomCrop((244, 244)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(args.data, x), data_transforms[x]) for x in ["train", "val"]}
    dataloader_dict = {x: DataLoader(image_datasets[x], shuffle=True, batch_size=args.batch_size, num_workers=args.workers) for x in ["train", "val"]}
    return dataloader_dict


def train(args):
    dataloaders = get_dataloader(args)
    if args.arch == "inception_v3":
        model = models.__dict__[args.arch](pretrained=True)
    else:
        model = models.__dict__[args.arch](aux_logits=False, transform_input=False, pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.cuda()

    save_model = copy.deepcopy(model)

    train_writer = SummaryWriter("./run/train")
    val_writer = SummaryWriter("./run/val")


    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.
    best_epoch = 0
    for epoch in range(args.epochs):
        print("Epoch {}/{}".format(epoch + 1, args.epochs))
        print("-" * 10)

        for phase in ["train", "val"]:
            running_loss = 0.
            running_corrects = 0.
            ground_truth = []
            pred_prob = []
            bin_pred = []

            if phase == "train":
                model.train()
            else:
                model.eval()
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.cuda()
                labels = labels.cuda()

                with torch.autograd.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                softmax_outputs = F.softmax(outputs, 1)
                pred_prob.extend(list(softmax_outputs.cpu().data.numpy()[:, 1]))
                ground_truth.extend(list(labels.cpu().data.numpy()))

                _, preds = torch.max(outputs, 1)
                bin_pred.extend(list(preds.cpu().data.numpy()))

                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds.view(-1) == labels.view(-1)).item()
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)
            recall = recall_score(ground_truth, bin_pred)
            precision = precision_score(ground_truth, bin_pred)
            auc = roc_auc_score(ground_truth, pred_prob)
            f1 = f1_score(ground_truth, bin_pred)

            print("{} Loss: {} Acc: {:0.4f} AUC: {:0.4f} Sensitivity: {:0.4f} Precision: {:0.4f} F1: {:0.4f}".format(
                phase,
                epoch_loss,
                epoch_acc,
                auc,
                recall,
                precision,
                f1))

            if phase == "train":
                train_writer.add_scalar('Loss', epoch_loss, global_step=epoch)
                train_writer.add_scalar("ACC", epoch_acc, global_step=epoch)
                train_writer.add_scalar("AUC", auc, global_step=epoch)
                train_writer.add_scalar("Sensitivity", recall, global_step=epoch)
                train_writer.add_scalar("Precision", precision, global_step=epoch)
                train_writer.add_scalar("F1", f1, global_step=epoch)
            else:
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_epoch = epoch + 1

                val_writer.add_scalar("Loss", epoch_loss, global_step=epoch)
                val_writer.add_scalar("ACC", epoch_acc, global_step=epoch)
                val_writer.add_scalar("AUC", auc, global_step=epoch)
                val_writer.add_scalar("Sensitivity", recall, global_step=epoch)
                val_writer.add_scalar("Precision", precision, global_step=epoch)
                val_writer.add_scalar("F1", f1, global_step=epoch)
        print()

        if epoch % 10 == 0:
            save_model.load_state_dict(best_model_wts)
            torch.save(save_model, "./model/" + args.arch + "_" + args.data.split("/")[-1] + ".pth")

    train_writer.close()
    val_writer.close()
    model.load_state_dict(best_model_wts)
    print(best_epoch, " model saved")
    return model


def evaluate(model, args):
    data_dir = args.data
    model.cuda()
    model.eval()
    transform = transforms.Compose([
                transforms.Resize((args.image_size, args.image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    res = []
    labels = []
    ground_truth = []
    postive_prob = []
    
    for root, dirs, names in os.walk(data_dir):
        for name in names:
            img_path = os.path.join(root, name)
            img = Image.open(img_path)
            img = transform(img)
            img = img.unsqueeze(0)
            img = img.cuda()

            with torch.no_grad():
                py = model(img)
            softmax_output = F.softmax(py, 1).cpu().data.numpy()
            np, pp = softmax_output[0][0], softmax_output[0][1]
            postive_prob.append(pp)
            if np >= 0.5:
                label = 0
            else:
                label = 1
            labels.append(label)
            # print(img_path.split("/")[-1], label, np, pp)
            if "non-cancer" in img_path:
                ground_truth.append(0)
                res.append([img_path.split("/")[-1], np, pp, 0, label])
            else:
                ground_truth.append(1)
                res.append([img_path.split("/")[-1], np, pp, 1, label])

    acc = accuracy_score(ground_truth, labels)
    recall = recall_score(ground_truth, labels)
    auc_score = roc_auc_score(ground_truth, postive_prob)
    f1 = f1_score(ground_truth, labels)

    result_file_path = "./result/acc-" + str(acc) + "-recall-" + str(recall) + "-auc-" + str(auc_score) + "-f1-" + str(f1) + "-" + args.data.split("/")[-2] + "-" +  args.data.split("/")[-1] +"-" + args.model_path.split("/")[-1] + ".csv"
    with open (result_file_path, "w", newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(["img", "non-cancer prob", "cancer prob", "ground truth", "predict label"])
        f_csv.writerows(res)

    # draw_roc(ground_truth, postive_prob, args)
    fpr,tpr,threshold = roc_curve(ground_truth, postive_prob)
    roc_auc = auc(fpr,tpr)

    plt.figure()
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example - ' + args.data.split("/")[-2] + "-" +  args.data.split("/")[-1] +"-" + args.model_path.split("/")[-1])
    plt.legend(loc="lower right")
    plt.savefig("./result/" + args.data.split("/")[-2] + "-" +  args.data.split("/")[-1] +"-" + args.model_path.split("/")[-1] + ".png")

    # res = pd.DataFrame(res, columns=["img", "non-cancer prob", "cancer prob", "ground truth", "predict label"])
    # res.to_csv("./result/acc-" + str(acc) + "-recall-" + str(recall) + "-auc-" + str(auc) + "-f1-" + str(f1) + "-" + args.data.split("/")[-2] + "-" +  args.data.split("/")[-1] +"-" + args.model_path.split("/")[-1] + ".csv", index=False)



def main():
    args = parser.parse_args()
    if not args.evaluate:
        model = train(args)
        torch.save(model, "./model/" + args.arch + "_" + args.data.split("/")[-1] + ".pth")
    else:
        evaluate(torch.load(args.model_path), args)


if __name__=="__main__":
    main()