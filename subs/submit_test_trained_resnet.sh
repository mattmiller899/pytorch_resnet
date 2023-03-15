#!/bin/bash

set -u
export STDERR_DIR="./err"
export STDOUT_DIR="./out"
#init_dir "$STDERR_DIR" "$STDOUT_DIR"

#DIRS
MULTIKMER_DIR="/xdisk/bhurwitz/mattmiller899/girus/embeddings/sorted"
#INPUT_DIR="/xdisk/bhurwitz/mattmiller899/girus/READ_files"
INPUT_DIR="/xdisk/bhurwitz/mattmiller899/girus/pytorch_resnet/reads"
#TODO CHANGE BACK
#POS_DIR="${MULTIKMER_DIR}/pos_in_one_hot"
POS_DIR="${MULTIKMER_DIR}/pos_in"
CONT_DIR="${MULTIKMER_DIR}/cont_in"
SEQ_DIR="${MULTIKMER_DIR}/seq_in"
AA_DIR="${MULTIKMER_DIR}/aa_in"
#TODO CHANGE BACK
RESULTS_DIR="./results"

#declare -a FLAGS=("-g -ct -cv -sc -up -us -ur -uc -ua" "-g -ct -cv -sc -up -us -uc -ua" "-g -ct -cv -sc -up -us -ur -uc" "-g -ct -cv -sc -up -us -uc")
#declare -a FLAGS=("-g -ct -cv -sc -up -us -ur -uc -ua")
declare -a FLAGS=("-g -d -uk -sc -uc -up -us -ur -ua")
#declare -a FLAGS=("-g -uk -sc -uc -ua -ur")
EPOCHS=2
BATCH=50
KFOLD=4
PADDING=3
DROPOUT=0.0
WD=0.0
#LOOPERS
declare -a KMERS=("6")
declare -a NO_NS=("no_n")
CONTIGS=(300)
ORGS=("virus")
#ARGS="-q standard -W group_list=bhurwitz -M mattmiller899@email.arizona.edu -m a"
ARGS="--partition=standard --account=bhurwitz --mail-user=mattmiller899@email.arizona.edu --mail-type=ALL"
for READ in ${CONTIGS[@]}; do
    for ORG in ${ORGS[@]}; do
        OUT_DIR="${RESULTS_DIR}/${READ}/${ORG}"
        GV_DIR="${INPUT_DIR}/girus/${READ}"
        V_DIR="${INPUT_DIR}/${ORG}/${READ}"
        GVMAG_DIR="${INPUT_DIR}/GVMAG/${READ}"
        #init_dir "$OUT_DIR"
        #IN_FILE="${BUG_DIR}/kmers/${READ}/${KMER}/${ORG}-reads.txt"
        for KMER in "${KMERS[@]}"; do #Surround in quotes so that spaces in the strings dont get split
            KMERSTR=${KMER//\ /-}
            for NO_N in ${NO_NS[@]}; do
                for FLAG in "${FLAGS[@]}"; do
                    OUTFLAGS=""
                    if [[ $FLAG = *-ur* ]]; then
                        OUTFLAGS="${OUTFLAGS}revfor_"  
                    else
                        OUTFLAGS="${OUTFLAGS}for_"
                    fi
                    if [[ $FLAG = *-uc* ]]; then
                        OUTFLAGS="${OUTFLAGS}cont_"
                    fi
                    if [[ $FLAG = *-us* ]]; then
                        OUTFLAGS="${OUTFLAGS}seq_"
                    fi
                    if [[ $FLAG = *-up* ]]; then
                        if [[ $POS_DIR = *pos_in_one_hot* ]]; then
                            OUTFLAGS="${OUTFLAGS}onehot_"
                        else
                            OUTFLAGS="${OUTFLAGS}pos_"
                        fi
                    fi
                    if [[ $FLAG = *-ua* ]]; then
                        OUTFLAGS="${OUTFLAGS}aa_"
                    fi
                    if [[ ${#KMER} -gt 2 ]]; then
                        OUTFLAGS="${OUTFLAGS}multikmer_"
                    else
                        OUTFLAGS="${OUTFLAGS}monokmer_"
                    fi
                    if [[ $FLAG = *-fz* ]]; then
                        OUTFLAGS="${OUTFLAGS}frozen_"
                    else
                        OUTFLAGS="${OUTFLAGS}unfrozen_"
                    fi
                    OUTFLAGS="${OUTFLAGS}${NO_N}"
                    OUTPUT_DIR="${OUT_DIR}/${KMERSTR}mer_${EPOCHS}eps_${OUTFLAGS}_1000filts"
                    #init_dir "$OUTPUT_DIR" 
                    MODEL_DIR="${OUTPUT_DIR}/models"
                    #FIG_DIR="${OUT_DIR}/figs/${KMERSTR}mer_${FILTER}f_${CONV}nc_${FC}fc_${PAT}pa_${EPOCHS}eps_${OUTFLAGS}_1000filts"
                    NEWCONT_DIR="${CONT_DIR}_${NO_N}"
                    NEWSEQ_DIR="${SEQ_DIR}_${NO_N}"
                    NEWPOS_DIR="${POS_DIR}_${NO_N}"
                    
                    export PY_ARGS="${FLAG} -gd ${GV_DIR} -vd ${V_DIR} -td ${GVMAG_DIR} -cd ${NEWCONT_DIR} -pd ${NEWPOS_DIR} -sd ${NEWSEQ_DIR} -ad ${AA_DIR} -r ${READ} -b ${BATCH} -e ${EPOCHS} -o ${OUTPUT_DIR} ${KMER}"
                    echo $PY_ARGS
                    if [[ $FLAG == *-g* ]]; then
                        echo "Using GPU"
                        #echo "qsub $ARGS -v PY_ARGS -N t${KMERSTR}_${FC}fc_${CONV}c_${FILTER}f -e $STDERR_DIR -o $STDOUT_DIR ./run_train_multikmer_gpu.sh"
                        #JOB_ID=`qsub $ARGS -v PY_ARGS -N t${KMERSTR}_${FC}fc_${CONV}c_${FILTER}f -e $STDERR_DIR -o $STDOUT_DIR ./run_train_cnn_multikmer_gpu.sh`
                        JOB_ID=`sbatch $ARGS --export=PY_ARGS --job-name=test_$(basename $OUTPUT_DIR) -e $STDERR_DIR/%x.err -o $STDOUT_DIR/%x.out ./runs/run_test_trained_resnet.sh`
                    else
                        echo "Using CPU"
                        #JOB_ID=`qsub $ARGS -v PY_ARGS -N train_${KMERSTR} -e $STDERR_DIR -o $STDOUT_DIR ./run_train_cnn_multikmer_cpu.sh`
                        JOB_ID=`sbatch $ARGS --export=PY_ARGS --job-name=t${KMERSTR}_${FC}fc_${CONV}c_${FILTER}f_${READ} -e $STDERR_DIR/%x.err -o $STDOUT_DIR/%x.out ./runs/run_train_resnet.sh`
                    fi
                    if [ "${JOB_ID}x" != "x" ]; then
                        echo Job: \"$JOB_ID\"
                    else
                        echo Problem submitting job. Job terminated.
                        exit 1
                    fi
                    echo "job successfully submitted"
                done
            done
        done
    done
done
