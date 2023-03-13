from pathlib import Path
import pickle
import csv
import os
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
import torch
import torch.utils.data
from math import floor
import glob
from Bio import SeqIO, Seq
#import h5py
import gc
from sklearn.model_selection import KFold
import itertools
from copy import deepcopy
#import h5py
import re
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight.data,
                                       nonlinearity="relu")


def generate_kmers(k):
    return [''.join(p) for p in itertools.product(["A", "C", "G", "T"], repeat=k)]

class ReadsDatasetCNN(torch.utils.data.Dataset):
    def __init__(self, all_reads, all_labels):
        self.num_reads = len(all_reads)
        self.labels = torch.LongTensor(all_labels)
        self.reads = torch.LongTensor(all_reads)

    def __len__(self):
        return len(self.reads)

    def __getitem__(self, idx):
        curr_read = self.reads[idx]
        curr_label = self.labels[idx]
        return curr_read, curr_label
        #return self.embedding(torch.LongTensor(curr_read)), torch.LongTensor(curr_label)


def load_embeddings_no_torchtext(kmer_sizes, use_cont, use_pos, use_seq, use_aa, cont_dir, pos_dir, seq_dir, aa_dir):
    all_embs, vec_sizes = [], []
    if use_cont:
        #with open(cont_file, "r") as f:
        #    _ = f.readline()
        #    ncols = len(f.readline().rstrip().split(" "))
        #    print(f"ncols = {ncols}")
        #cont_arr = np.loadtxt(cont_file, delimiter=" ", skiprows=1, usecols=range(1,ncols))
        #cont_kmers = np.loadtxt(cont_file, delimiter=" ", skiprows=1, usecols=0, dtype=np.string)
        #cont_arr = np.genfromtxt(cont_file, delimiter=" ", skip_header=1, dtype=None)
        #all_embs.append(torch.from_numpy(cont_arr))
        cont_tensor, cont_size = read_embeddings_no_torchtext(kmer_sizes, cont_dir)
        all_embs.append(cont_tensor)
        vec_sizes.append(cont_size)
    if use_pos:
        pos_tensor, pos_size = read_embeddings_no_torchtext(kmer_sizes, pos_dir)
        all_embs.append(pos_tensor)
        vec_sizes.append(pos_size)
    if use_seq:
        seq_tensor, seq_size = read_embeddings_no_torchtext(kmer_sizes, seq_dir)
        all_embs.append(seq_tensor)
        vec_sizes.append(seq_size)
    if use_aa:
        aa_tensor, aa_size = read_embeddings_no_torchtext(kmer_sizes, aa_dir)
        all_embs.append(aa_tensor)
        vec_sizes.append(aa_size)
    if len(vec_sizes) == 0:
        print("ERROR: No embedding vectors loaded. Exiting...")
        exit(1)
    return all_embs, vec_sizes


def read_embeddings_no_torchtext(kmer_sizes, emb_dir):
    emb_arr = None
    for k in kmer_sizes:
        kmer_glob = glob.glob(f"{emb_dir}/*{k}k*.txt")
        kmer_glob.extend(glob.glob(f"{emb_dir}/*{k}mer*.txt"))
        kmer_glob.extend(glob.glob(f"{emb_dir}/*{k}aa*.txt"))
        print(f"glob = {kmer_glob}")
        for in_file in kmer_glob:
            with open(in_file, "r") as f:
                ncols = len(f.readline().rstrip().split(" "))
            print(f"ncols = {ncols}")
            tmp_arr = np.loadtxt(in_file, delimiter=" ", usecols=range(1,ncols))
            if emb_arr is None:
                emb_arr = tmp_arr
            else:
                emb_arr = np.vstack((emb_arr, tmp_arr))
    emb_size = len(emb_arr[0])
    emb_tensor = torch.from_numpy(emb_arr)
    return emb_tensor, emb_size


def read_fastas_from_dirs_CNN(in_dirs, read_size, kmer_sizes, use_stepk=True, use_rev=True, kmer_dict=None):
    all_reads, all_labels = [], []
    all_nucs = ["A", "C", "T", "G"]
    if kmer_dict == None:
        all_kmers = []
        for k in kmer_sizes:
            all_kmers.extend(generate_kmers(k))
        kmer_dict = {}
        for i, tmp_kmer in enumerate(all_kmers):
            kmer_dict[tmp_kmer] = i
    for label_num, in_dir in enumerate(in_dirs):
        for count, in_file in enumerate(glob.glob(f"{in_dir}/*.fa")):
            print(f"reading {in_file}")
            for record in SeqIO.parse(in_file, "fasta"):
                tmp_nuc_arr = []
                skipped = False
                for kmer in kmer_sizes:
                    if use_stepk:
                        step_size = kmer
                    else:
                        step_size = 1
                    for_seq = str(record.seq.upper())
                    for_seq = re.sub(r'[BDEFHIJKLMNOPQRSVWXYZ]', random.choice(all_nucs), for_seq)
                    for_seq = re.sub(r'[U]', 'T', for_seq)
                    length = len(for_seq)
                    if length < read_size:
                        skipped = True
                        continue
                    # seq and rev_comp
                    tmp_for = []
                    for j in range(0, read_size - kmer + 1, step_size):
                        # try:
                        tmp_for.append(kmer_dict[for_seq[j:j + kmer]])
                        # except:
                        #    tmp_for.append(nuc_dict[random.choice(all_nucs)])
                    # tmp_for = for_seq
                    tmp_nuc_arr.extend(tmp_for)
                    if use_rev:

                        rev_seq = re.sub(r'[BDEFHIJKLMNOPQRSVWXYZ]', random.choice(all_nucs), str(Seq.reverse_complement(for_seq).upper()))
                        rev_seq = re.sub(r'[U]', 'T', rev_seq)
                        # tmp_rev = rev_seq
                        tmp_rev = []
                        for j in range(0, read_size - kmer + 1, step_size):
                            # try:
                            tmp_rev.append(kmer_dict[rev_seq[j:j + kmer]])
                            # except:
                            #    tmp_rev.append(nuc_dict[random.choice(all_nucs)])
                        tmp_nuc_arr.extend(tmp_rev)
                if not skipped:
                    all_reads.append(tmp_nuc_arr)
                    all_labels.append(label_num)
                # print(f"tmp_nuc_arr = {tmp_nuc_arr}\ntmp_arr = {tmp_arr}")
                # exit()
    num_kmers_per_read = len(all_reads[0])
    return np.array(all_reads), np.array(all_labels), kmer_dict, num_kmers_per_read


def test_resnet(model, dataloader, criterion, device, args):
    preds = []
    truths = []
    round_preds = []
    total_loss = 0.0
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.float().to(device)
            model.zero_grad()
            if args.debug:
                print(f"data = {data}\nlabels = {labels}")
            pred = model(data).view(labels.size(0))
            if args.debug:
                print(f"pred = {pred}\npred shape = {pred.shape}")
            total_loss += float(criterion(pred, labels).item())
            round_pred = torch.round(pred).cpu().numpy()
            round_preds.extend(round_pred)
            preds.extend(pred.cpu().numpy())
            truths.extend(labels.cpu().numpy())
    truths = np.array(truths)
    print(f"truths shape = {truths.shape}")
    print(f"unique truths = {np.unique(truths)}\nunique round_preds = {np.unique(round_preds)}")
    acc = accuracy_score(truths, round_preds)
    f1 = f1_score(truths, round_preds, average="macro")
    prec = precision_score(truths, round_preds, average="macro")
    rec = recall_score(truths, round_preds, average="macro")
    avg_loss = total_loss / len(dataloader)
    return avg_loss, acc, f1, prec, rec

