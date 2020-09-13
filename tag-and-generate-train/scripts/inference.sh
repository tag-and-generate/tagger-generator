#!/bin/bash
# Given a sentence, runs the tag-generate model on it
set -u


input_file="$1"         # the input test file which needs to be transferred
jobname="$2"            # unique identifier for the inference job 
tagger_target="$3"      # target [argument we pass when we run scripts/train_tagger.sh] used to train tagger
generator_target="$4"   # target [argument we pass when we run scripts/train_generator.sh] used to train generator
dataset="$5"            # dataset [argument we pass when we train tagger or generator -- used to identify model paths for tagger and generator]
src_tag="$6"            # style_0_label used to create training data [src style] 
tgt_tag="$7"            # style_1_label used to create training data [tgt style]
base_folder="$8"        # path to data folder, where outputs of the training data creation process are stored
device="$9"             # gpu id [comment line in script if it needs to be run on cpu]

tag_generate_base="experiments/" # base dir to store outputs of inference
mkdir -p $tag_generate_base

# SET UNSET BPE HERE
BPE=1
if [ "$BPE" -eq  1 ]; then
    MODEL_PTH="bpe"
    echo "Using BPE"
else
    MODEL_PTH="nobpe"
    echo "Not using BPE"
fi
# 

## ARCHITECTURE
HSZ=512
EMBED_DIM=512
NHEAD=4
NL=4
##



function infer() {
    # run inference
    infile="$1"
    src="$2"
    tgt="$3"
    model="$4"
    outfile="$5"
    prefer_gtag="$6"
    if [ "$BPE" -eq  1 ]; then
        CUDA_VISIBLE_DEVICES=$device python src/translate.py --cuda --src "$src" \
            --tgt "$tgt" \
            --model-file "$model" \
            --search "beam_search" \
            --hidden-dim $HSZ \
            --embed-dim $EMBED_DIM \
            --n-heads $NHEAD \
            --n-layers $NL \
            --beam-size 5 \
            --bpe \
            --prefer_gtag "$prefer_gtag" \
            --tag "$src_tag" \
            --input-file "$infile" \
            --output-file "$outfile" \
            --base-folder "$base_folder"
    else
        CUDA_VISIBLE_DEVICES=$device python src/translate.py --cuda --src "$src" \
            --tgt "$tgt" \
            --model-file "$model" \
            --search "beam_search" \
            --hidden-dim $HSZ \
            --embed-dim $EMBED_DIM \
            --n-heads $NHEAD \
            --n-layers $NL \
            --beam-size 5 \
            --tag "$src_tag" \
            --prefer_gtag "$prefer_gtag" \
            --input-file "$infile" \
            --output-file "$outfile" \
            --base-folder "$base_folder"
    fi
        
}


function add_eos() {
    # append eos to each line of the file
    ip="$1"
    awk '{printf("%s <eos>\n", $0)}' $ip > "${ip}.bak"
    mv "${ip}.bak" $ip
}



# Step 1: Run Preprocess/BPE on the input
TAGGER_INPUT="${tag_generate_base}/${jobname}_tagger_input"
if [ $BPE -eq 1 ]; then
    echo "Running BPE on input"
    CUDA_VISIBLE_deviceS=$device python src/subwords.py segment\
                                --model "$base_folder/en${tagger_target}_subwords.model" < "$input_file"\
                                > "$TAGGER_INPUT"
else     
    cp "$input_file" "$TAGGER_INPUT"
fi
echo "Adding eos to the input"
add_eos "$TAGGER_INPUT"


# Step 2: Tag the input
echo "Running tagger"
infer "$TAGGER_INPUT" "en" "$tagger_target" "models/${dataset}/${MODEL_PTH}/en-${tagger_target}-tagger.pt"\
      "${tag_generate_base}/${jobname}_tagged" 1
### SRC_TAG -> TGT_TAG
sed -i "s/${src_tag}/${tgt_tag}/g" "${tag_generate_base}/${jobname}_tagged"


# Step 3: Run Preprocess/BPE on the tagger output
GENERATOR_INPUT="${tag_generate_base}/${jobname}_generator_input"
if [ $BPE -eq 1 ]; then
    echo "Running BPE on masked output"
    CUDA_VISIBLE_deviceS=$device python src/subwords.py segment\
                            --model "$base_folder/en${generator_target}_subwords.model" < "${tag_generate_base}/${jobname}_tagged" > "$GENERATOR_INPUT"
        
else
    cp "${tag_generate_base}/${jobname}_tagged"  "$GENERATOR_INPUT"
fi
add_eos "$GENERATOR_INPUT"


# Step 4: Generate
echo "Running generator"
infer "$GENERATOR_INPUT" "en" "${generator_target}" "models/${dataset}/${MODEL_PTH}/en-${generator_target}-generator.pt"\
      "${tag_generate_base}/${jobname}_output" 0
sed -i 's/^\"//g' "${tag_generate_base}/${jobname}_output"


# Step 5: Run sacrebleu
cat "$input_file"|sacrebleu -w2  "${tag_generate_base}/${jobname}_output"
