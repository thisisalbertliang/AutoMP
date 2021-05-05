BATCH_SIZE=256
VOCAB_SIZE=2048
# NPROC=2
NUM_LAYER=3

HIDDEN_SIZE=1024

# clear log
rm -rf benchmark

# nprocs=(1 2 4 8 16)
for nproc in 1 2 4 8 16 #$nprocs
do
    # if [ $nproc -gt 2 ]
    # then
    # emb
    # all_vocab_size=(256 512 1024 1536 1920 2304 3072 4096 8192 16384 32768 65536 131072 262144 524288)
    echo "### EMB EXPERIMENTS"
    for vs in 1024 3072 8192 32768 65536 131072 262144 524288 #$all_vocab_size
    do
        echo "## running nproc=$nproc, vs=$vs"
        if python3 -m torch.distributed.launch --nproc_per_node=$nproc train_embeddings.py \
            --batch-size=$BATCH_SIZE \
            --hidden-size=$HIDDEN_SIZE \
            --vocab-size=$vs; then
            echo "yes"
        else
            break
        fi
    done
    # fi

    #transformer layer
    # all_hs=(256 512 1024 1536 1920 2304 3072 4096)
    # all_nah=(8 16 32 64)
    # if [ $nproc -eq 1 ]
    # then
    #     continue
    # fi

    echo "### TRANSFORMER LAYER EXPERIMENTS"
    for nah in 8 16 32 64 #$all_nah
    do
        for hs in 256 512 1024 1536 1920 2304 3072 4096 #$all_hs
        do
        echo "## running nproc=$nproc, nah=$nah, hs=$hs"
        if python3 -m torch.distributed.launch --nproc_per_node=$nproc train_transformer_layer.py \
            --batch-size=$BATCH_SIZE \
            --hidden-size=$hs \
            --num-attention-heads=$nah \
            --vocab-size=$VOCAB_SIZE; then
            echo "yes"
        else
            break
        fi
        done
    done

    #gpt2
    echo "### GPT-2 EXPERIMENTS"
    for nah in 8 16 32 64 #$all_nah
    do
        for hs in 256 512 1024 1536 1920 2304 3072 4096 #$all_hs
        do
            echo "## running nproc=$nproc, nah=$nah, hs=$hs"
            if python3 -m torch.distributed.launch --nproc_per_node=$nproc train_gpt2.py \
                --batch-size=$BATCH_SIZE \
                --hidden-size=$hs \
                --num-attention-heads=$nah \
                --num-layers=$NUM_LAYER \
                --vocab-size=$VOCAB_SIZE; then
                echo "yes"
            else
                break
            fi
        done
    done

done