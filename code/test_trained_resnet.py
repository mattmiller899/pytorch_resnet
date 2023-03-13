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
python code/test_trained_resnet.py -gd data/girus -vd data/virus -td data/GVMAG -uc -cd embeddings/cont/ -up -pd embeddings/pos -us -sd embeddings/seq -ua -ad embeddings/aa -o results/ -ur  -g -b 10 -e 1 -sc 6

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
    _, _, kmer_dict, num_kmers_per_read = utils.read_fastas_from_dirs_CNN(
        [args.girus_dir, args.virus_dir],
        args.read_size,
        args.kmer_sizes,
        args.use_stepk,
        use_rev=args.use_rev
    )
    # Load data
    # out.write("Fold,Epoch,Test Loss,Accuracy,F1-Score,Precision,Recall,AUPRC,AUROC\n")
    # Create model, train, and test k-fold times
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
    print("Testing GVMAGs")
    losses, accs, f1s, precs, recs = [], [], [], [], []
    model_dir = f"{args.output_dir}/models"
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    if not os.path.isdir(f"{args.output_dir}/runs"):
        os.makedirs(f"{args.output_dir}/runs")
    for superclade in range(1, 11):
        if superclade == 5:
            continue
        print(f"superclade {superclade}")
        tmp_in_dir = f"{args.test_dir}/SC{superclade}"
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
        for tmp_epoch in range(args.epochs):
            curr_model_path = f"{model_dir}/model_r{args.read_size}_k{args.kmer_sizes}" \
                      f"_layers{layers}_epoch{tmp_epoch}.pt"
            model.load_state_dict(torch.load(curr_model_path, map_location=device))

            model.eval()
            avg_test_loss, acc, f1, prec, rec = utils.test_resnet(model, test_dataloader, criterion, device, args)
            print(f"RESULTS,{superclade},{tmp_epoch},{avg_test_loss:.4f},{acc:.4f},{f1:.4f},{prec:.4f},{rec:.4f}\n")
            losses.append(avg_test_loss)
            accs.append(acc)
            f1s.append(f1)
            precs.append(prec)
            recs.append(rec)


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
