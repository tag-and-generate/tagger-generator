# Prepare BPE

tgt="$1"
base_folder="$2"
VOCAB_SIZE=16000
python src/subwords.py train \
    --model_prefix "${base_folder}"/en${tgt}_subwords \
    --vocab_size "${VOCAB_SIZE}" \
    --model_type bpe \
    --input "${base_folder}"/en${tgt}_parallel.train.$tgt,"${base_folder}"/en${tgt}_parallel.train.en

# Apply BPE
for split in train dev test
do
    for l in $tgt en
    do
        python src/subwords.py segment \
        --model "${base_folder}"/en${tgt}_subwords.model \
        < "${base_folder}"/en${tgt}_parallel.$split.$l \
        > "${base_folder}"/en${tgt}_parallel.bpe.$split.$l
    done
done
