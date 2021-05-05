BATCH_SIZE=256
VOCAB_SIZE=2048
# NPROC=2
NUM_LAYER=3

HIDDEN_SIZE=1024

# clear log
rm -rf benchmark

nprocs=(1 2 4 8)
for nproc in $nprocs
    # emb
    all_vocab_size=(256 512 1024 1536 1920 2304 3072 4096 8192 16384 32768 65536 131072 262144 524288)
    echo "### EMB EXPERIMENTS"
    for vs in $all_vocab_size
    do
        echo "## running vs=$vs"
        python3 -m torch.distributed.launch --nproc_per_node=$nproc train_embeddings.py \
            --batch-size=$BATCH_SIZE \
            --hidden-size=$HIDDEN_SIZE \
            --vocab-size=$vs
    done

    #transformer layer
    all_hs=(256 512 1024 1536 1920 2304 3072 4096)
    all_nah=(8 16 32 64)
    echo "### TRANSFORMER LAYER EXPERIMENTS"
    for nah in $all_nah
    do
        for hs in $all_hs
        do
        echo "## running nah=$nah, hs=$hs"
        python3 -m torch.distributed.launch --nproc_per_node=$nproc train_transformer_layer.py \
            --batch-size=$BATCH_SIZE \
            --hidden-size=$hs \
            --num-attention-heads=$nah \
            --vocab-size=$VOCAB_SIZE
        done
    done

    #gpt2
    echo "### GPT-2 EXPERIMENTS"
    for nah in $all_nah
    do
        for hs in $all_hs
        do
            echo "## running nah=$nah, hs=$hs, nl=$nl"
            python3 -m torch.distributed.launch --nproc_per_node=$nproc train_transformer_layer.py \
                --batch-size=$BATCH_SIZE \
                --hidden-size=$hs \
                --num-attention-heads=$nah \
                --num-layers=$NUM_LAYER \
                --vocab-size=$VOCAB_SIZE
        done
    done