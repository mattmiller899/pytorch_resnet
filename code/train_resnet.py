import torch
import torch.optim as optim
import torch.nn as nn
import resnet_utils as utils
import argparse
import os
import sys
import math
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import *
from torchsummary import summary
#from pytorch_lamb import Lamb, log_lamb_rs
import random
from torch.utils.tensorboard import SummaryWriter
import time
from resnet_models import ResNetModel
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

"""
python code/train_resnet.py -gd data/girus -vd data/virus -uc -cd embeddings/cont/ -up -pd embeddings/pos -us -sd embeddings/seq -ua -ad embeddings/aa -o results/ -ur  -g -b 10 -e 1 -sc 6

"""



def main(args):
    # Load data
    layers = [3, 4, 6, 3]
    random.seed(17)
    torch.manual_seed(17)  # Randomly seed PyTorch
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.use_gpu) else "cpu")
    print(f"device = {device}")
    # TODO change num_classes
    num_classes = 2
    if args.debug:
        print(f"num_classes = {num_classes}")
    NUM_KMERS = len(args.kmer_sizes)
    num_channels = args.use_cont + args.use_pos + args.use_seq + args.use_aa if args.split_channels else 1
    # Load embeddings
    all_embs, vec_sizes = utils.load_embeddings_no_torchtext(args.kmer_sizes, args.use_cont, args.use_pos, args.use_seq, args.use_aa, args.cont_dir, args.pos_dir, args.seq_dir, args.aa_dir)
    # Load data
    all_reads, all_labels, kmer_dict, num_kmers_per_read = utils.read_fastas_from_dirs_CNN(
        [args.girus_dir, args.virus_dir],
        args.read_size,
        args.kmer_sizes,
        args.use_stepk,
        use_rev=args.use_rev
    )
    dummy_arr = np.zeros((len(all_reads)))
    dev_kf = KFold(10, shuffle=True)
    for (train_idx, dev_idx) in dev_kf.split(dummy_arr):
        break
    train_dataset = utils.ReadsDatasetCNN(all_reads[train_idx], all_labels[train_idx])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=0,
                                                   drop_last=False)
    dev_dataset = utils.ReadsDatasetCNN(all_reads[dev_idx], all_labels[dev_idx])
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                                 drop_last=True)
    best_dev_loss = 1000000
    best_dev_epoch = -1
    output_file = f"{args.output_dir}/r{args.read_size}_k{args.kmer_sizes}.csv"
    model_dir = f"{args.output_dir}/models"
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    if not os.path.isdir(f"{args.output_dir}/runs"):
        os.makedirs(f"{args.output_dir}/runs")
    with open(output_file, "w") as out:
        # out.write("Fold,Epoch,Test Loss,Accuracy,F1-Score,Precision,Recall,AUPRC,AUROC\n")
        out.write("Epoch,Dev Loss,Accuracy,F1-Score,Precision,Recall\n")
        # Create model, train, and test k-fold times
        loss_writer = SummaryWriter(f"{args.output_dir}/runs")
        model = ResNetModel(layers,
                            all_embs,
                            num_kmers_per_read,
                            num_channels,
                            use_gpu=args.use_gpu,
                            debug=args.debug,
                            kmer_sizes=args.kmer_sizes,
                            vec_sizes=vec_sizes,
                            ).to(device)
        model.apply(utils.init_weights)
        opt = optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)
        # TODO test other loss functions
        criterion = nn.BCELoss()

        curr_epoch = 0
        while curr_epoch < args.epochs:
            model.train()
            total_loss = 0.0
            running_loss = 0.0
            for batch_count, (data, labels) in enumerate(train_dataloader):
                # Clear the model gradients and current gradient
                data, labels = data.to(device), labels.float().to(device)
                # writer.add_graph(model, data)
                # writer.close()
                model.zero_grad()
                opt.zero_grad()
                if args.debug:
                    print(f"data = {data}\ndata.shape = {data.shape}\nlabels = {labels}\nlabels shape = {labels.shape}")
                pred = model(data).view(labels.size(0))
                if args.debug:
                    print(f"pred = {pred}\npred shape = {pred.shape}")
                loss = criterion(pred, labels)
                # loss = criterion(torch.log(pred), labels)
                loss.backward()
                opt.step()
                curr_loss = float(loss.item())
                total_loss += curr_loss
                running_loss += curr_loss
                # if args.debug:
                #    print(f"mem usage after batch {batch_count}")
                #    utils.print_mem_usage(device)
                #    if batch_count == 2:
                #       exit()
                print(f"batch {batch_count}")
                if batch_count % args.log_interval == args.log_interval - 1:
                    loss_writer.add_scalar(f'Training Loss/batch {batch_count}',
                                           running_loss / args.log_interval,
                                           curr_epoch * len(train_dataloader) + batch_count)
                    running_loss = 0.0
            train_loss = total_loss / len(train_dataloader)
            # train_loss_arr.append(train_loss)
            # Evaluate model with dev set
            model.eval()
            print(f"len dev_dataloader = {len(dev_dataloader)}")
            avg_dev_loss, acc, f1, prec, rec = utils.test_resnet(model, dev_dataloader, criterion, device, args)

            model_path = f"{model_dir}/model_r{args.read_size}_k{args.kmer_sizes}" \
                         f"_layers{layers}_epoch{curr_epoch}.pt"
            torch.save(model.state_dict(), model_path)
            if avg_dev_loss < best_dev_loss:
                best_dev_epoch = curr_epoch
                best_dev_loss = avg_dev_loss
            print(f"Epoch {curr_epoch} train loss = {train_loss:.4f} dev loss = {avg_dev_loss:.4f}")
            out.write(f"{curr_epoch},{avg_dev_loss:.4f},{acc:.4f},{f1:.4f},{prec:.4f},{rec:.4f}\n")
            curr_epoch += 1
        best_model_path = f"{model_dir}/model_r{args.read_size}_k{args.kmer_sizes}" \
                          f"_layers{layers}_epoch{best_dev_epoch}.pt"
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        best_model_path = f"{model_dir}/model_r{args.read_size}_k{args.kmer_sizes}" \
                          f"_layers{layers}_epoch{best_dev_epoch}_best.pt"
        torch.save(model.state_dict(), best_model_path)

    #Test against GVMAGs
    print("Testing GVMAGs")
    test_file = f"{args.output_dir}/GVMAG_r{args.read_size}_k{args.kmer_sizes}.csv"
    model.load_state_dict(torch.load(best_model_path), map_location=device)
    with open(test_file, "w") as out:
        out.write("SC,Test Loss,Accuracy,F1-Score,Precision,Recall\n")
        losses, accs, f1s, precs, recs = [], [], [], [], []
        for superclade in range(1, 11):
            if superclade == 5:
                continue
            print(f"superclade {superclade}")
            tmp_in_dir = f"{args.input_dir}/SC{superclade}"
            # Load data
            test_reads, test_labels, kmer_dict, num_kmers_per_read = utils.read_fastas_from_dirs_CNN(
                [tmp_in_dir],
                args.read_size,
                args.kmer_sizes,
                args.use_stepk,
                use_rev=args.use_rev
            )
            test_dataset = utils.ReadsDatasetCNN(test_reads, test_labels)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                                                              num_workers=0,
                                                              drop_last=False)
            model.eval()
            avg_test_loss, acc, f1, prec, rec = utils.test_resnet(model, test_dataloader, criterion, device, args)
            out.write(f"{superclade},{avg_test_loss:.4f},{acc:.4f},{f1:.4f},{prec:.4f},{rec:.4f}\n")
            losses.append(avg_test_loss)
            accs.append(acc)
            f1s.append(f1)
            precs.append(prec)
            rec.append(rec)
        out.write(f"-1,-1,{np.average(losses):.4f},{np.average(accs):.4f},{np.average(f1s):.4f},"
                  f"{np.average(precs):.4f},{np.average(recs):.4f}\n")
        out.write(f"-2,-2,{np.std(losses):.4f},{np.std(accs):.4f},{np.std(f1s):.4f},{np.std(precs):.4f},"
                  f"{np.std(recs):.4f}\n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-gd", "--girus_dir", required=True,
                        help="path to the girus input dir")
    parser.add_argument("-vd", "--virus_dir", required=True,
                        help="path to the virus input dir")
    parser.add_argument("-td", "--test_dir", required=True,
                        help="path to directory containing GVMAG reads")
    parser.add_argument("-o", "--output_dir", required=True,
                        help="path to output dir")
    parser.add_argument("-cd", "--cont_dir", required=False,
                        help="path to the directory containing contextual kmer embeddings")
    parser.add_argument("-pd", "--pos_dir", required=False,
                        help="path to the dir containing positional kmer embeddings")
    parser.add_argument("-sd", "--seq_dir", required=False,
                        help="path to the dir containing sequential kmer embeddings")
    parser.add_argument("-ad", "--aa_dir", required=False,
                        help="path to the dir containing aa kmer embeddings")
    parser.add_argument("-e", "--epochs", type=int,
                        help="number of epochs to train",
                        default=1)
    parser.add_argument("-a", "--learning-rate", type=float,
                        help="learning rate",
                        default=1e-3)
    parser.add_argument("-r", "--read_size", type=int,
                        help="read length",
                        default=300)
    parser.add_argument("-b", "--batch-size", type=int,
                        help="batch size",
                        default=100)
    parser.add_argument("-g", "--use_gpu", dest="use_gpu", action="store_true",
                        help="indicate whether to use CPU or GPU")
    parser.set_defaults(use_gpu=False)
    parser.add_argument("-d", "--debug", dest="debug", action="store_true",
                        help="print all the statements")
    parser.set_defaults(debug=False)

    parser.add_argument("-uk", "--use_stepk", dest="use_stepk", action="store_false",
                        help="split reads into overlapping kmers")
    parser.set_defaults(use_stepk=True)
    parser.add_argument("-li", "--log-interval", type=int,
                        help="number of training batches between each log",
                        default=10)

    parser.add_argument("-sc", "--split_channels", dest="split_channels", action="store_true",
                        help="indicate whether to use separate channels for the seq/pos/cont embeddings")
    parser.set_defaults(split_channels=False)

    parser.add_argument("-up", "--use_pos", dest="use_pos", action="store_true",
                        help="indicate whether to use positional embeddings")
    parser.set_defaults(use_pos=False)

    parser.add_argument("-us", "--use_seq", dest="use_seq", action="store_true",
                        help="indicate whether to use sequential embeddings")
    parser.set_defaults(use_seq=False)

    parser.add_argument("-uc", "--use_cont", dest="use_cont", action="store_true",
                        help="indicate whether to use contextual embeddings")
    parser.set_defaults(use_cont=False)

    parser.add_argument("-ur", "--use_rev", dest="use_rev", action="store_true",
                        help="indicate whether to use reverse complement")
    parser.set_defaults(use_rev=False)

    parser.add_argument("-ua", "--use_aa", dest="use_aa", action="store_true",
                        help="indicate whether to use amino acid representations in addition to kmers (only works for 3/6/9/etc)")
    parser.set_defaults(use_aa=False)

    parser.add_argument("-fz", "--freeze", action="store_false",
                        help="indicate whether to freeze embeddings or make them trainable")
    parser.set_defaults(freeze=True)

    parser.add_argument("-p", "--padding", type=int,
                        help="amount of padding to use for convolution layers",
                        default=0)

    parser.add_argument("-do", "--dropout", type=float,
                        help="dropout rate for linear layers",
                        default=0.0)
    parser.add_argument("-wd", "--weight_decay", type=float,
                        help="weight_decay for L2 regularization",
                        default=0.0)
    parser.add_argument('kmer_sizes', metavar='kmer_sizes', type=int, nargs='+',
                        help='kmer sizes to use')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)
