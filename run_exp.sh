BATCH_SIZE=256
VOCAB_SIZE=512800
NPROC=2

HIDDEN_SIZE=1024

# clear log
rm -rf benchmark

# emb
all_vocab_size=(256 512 1024 1536 1920 2304 3072 4096 8192 16384 32768 65536 131072 262144 524288)
echo "### EMB EXPERIMENTS"
for vs in $all_vocab_size
do
    echo "## running vs=$vs"
    python3 -m torch.distributed.launch --nproc_per_node=$NPROC train_embeddings.py \
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
    python3 -m torch.distributed.launch --nproc_per_node=$NPROC train_transformer_layer.py \
        --batch-size=$BATCH_SIZE \
        --hidden-size=$hs \
        --num-attention-heads=$nah \
        --vocab-size=$VOCAB_SIZE
    done
done

#gpt2
all_nl=(4 8 16 32 40 54 64 72 80)
echo "### GPT-2 EXPERIMENTS"
for nah in $all_nah
do
    for hs in $all_hs
    do
        for nl in $all_nl
        do
            echo "## running nah=$nah, hs=$hs, nl=$nl"
            python3 -m torch.distributed.launch --nproc_per_node=$NPROC train_transformer_layer.py \
                --batch-size=$BATCH_SIZE \
                --hidden-size=$hs \
                --num-attention-heads=$nah \
                --num-layers=$nl \
                --vocab-size=$VOCAB_SIZE
        done
    done
done