"""Generates tags
Usage:
    run.py [options]

Options:
    --data_pth=<str>                        Path to the data directory
    --outpath=<str>                         Output path
    --style_0_label=<str>                   Label for style 0
    --style_1_label=<str>                   Label for style 1
    --ngram_range_min=<int>                 Min n_gram_range [default: 1]
    --ngram_range_max=<int>                 Max n_gram_range [default: 2]
    --style_label_col=<str>                 Name of the column that has style label column [default: style]
    --thresh=<float>                        tf-idf ratio threshold [default: 0.90]
    --is_unimodal=<bool>                    Whether the dataset is unimodal (like politeness) or has two styles (like yelp)
"""
from docopt import docopt
import json
import pandas as pd
import pandas as pd
import numpy as np
import tempfile
import sys
import subprocess
from collections import Counter
from typing import List
import logging

from src.style_tags import TFIDFStatsGenerator, RelativeTagsGenerator, TrainDataGen


def tag_style_markers(data_pth, outpath, style_0_label, style_1_label, tgt_lang="tagged", thresh=0.90, ngram_range=(1, 2),
                      ignore_from_tags=None, style_label_col="label", drop_duplicates=False,
                      gen_tags=True):
    """Runs tag generator. After this step, the following files are generated in the ``outpath`` directory:
        * entgt_lang_parallel.{split}.en.style_N_label: Sentences in style N
        * entgt_lang_parallel.{split}.taged.style_N_label: Sentences in style N with attribute phrases tagged
        (Here N is either 0 or 1, and split is one of {train, test, dev})
        * style_N_tags.json: Attribute tags for style N (0 or 1)

        A combination of the above files is sufficient to generate training data 
        for seq2seq models used by the tag-and-generate approach.
    Args:
        data_pth ([type]): Path to a file with the data. Each file should have the following columns:
            txt: The actual text
            split: train/test/dev
            style_label_col: indicates the style
        outpath ([type]): [description]
        style_0_label ([type]): [description]
        style_1_label ([type]): [description]
        tgt_lang ([type]): [description]
        thresh (float, optional): [description]. Defaults to 0.90.
        ngram_range (tuple, optional): [description]. Defaults to (1, 2).
        ignore_from_tags ([type], optional): [description]. Defaults to None.
        style_label_col (str, optional): [description]. Defaults to "label".
    """
    data = pd.read_csv(data_pth, sep="\t")
    if drop_duplicates:
        data = data.drop_duplicates(subset="txt")

    # Step 1
    logging.info("Reading the data")
    data_style_0 = data[data[style_label_col] == style_0_label]
    data_style_1 = data[data[style_label_col] == style_1_label]

    if gen_tags:
        # Step 2
        logging.info("Getting TF-IDF stats for both the corpora")
        logging.info(f"#Records {style_0_label} = {len(data_style_0)}")
        logging.info(f"#Records {style_1_label} = {len(data_style_1)}")

        tags_style_0, tags_style_1 = generate_tags(df_txt_class_1=data_style_0[data_style_0["split"] != "test"]["txt"],
                                                   df_txt_class_2=data_style_1[data_style_1["split"]
                                                                               != "test"]["txt"],
                                                   tag_class_1=style_0_label,
                                                   tag_class_2=style_1_label,
                                                   ignore_from_tags=ignore_from_tags,
                                                   thresh=thresh,
                                                   ngram_range=ngram_range)

        with open(f"{outpath}/{style_0_label}_tags.json", "w") as f:
            json.dump(tags_style_0, f)

        with open(f"{outpath}/{style_1_label}_tags.json", "w") as f:
            json.dump(tags_style_1, f)

    else:
        with open(f"{outpath}/{style_0_label}_tags.json", "r") as f:
            tags_style_0 = json.load(f)
        with open(f"{outpath}/{style_1_label}_tags.json", "r") as f:
            tags_style_1 = json.load(f)

    # Step 3
    logging.info("Generating the tagged data")
    TrainDataGen(data=data_style_0, outpath=outpath, tags=tags_style_0,
                 tag_token=style_0_label, tgt_lang=tgt_lang).generate()
    TrainDataGen(data=data_style_1, outpath=outpath, tags=tags_style_1,
                 tag_token=style_1_label, tgt_lang=tgt_lang).generate()


def generate_tags(df_txt_class_1,
                  df_txt_class_2,
                  tag_class_1,
                  tag_class_2,
                  thresh,
                  ngram_range,
                  ignore_from_tags=None,
                  ):
    stats_class_1 = TFIDFStatsGenerator(
        df_txt_class_1, tag_class_1, ngram_range=ngram_range)
    stats_class_2 = TFIDFStatsGenerator(
        df_txt_class_2, tag_class_2, ngram_range=ngram_range)

    class_1_tags = RelativeTagsGenerator(main_class_stats=stats_class_1,
                                         relative_class_stats=stats_class_2,
                                         ignore_from_tags=ignore_from_tags,
                                         thresh=thresh).tags

    class_2_tags = RelativeTagsGenerator(main_class_stats=stats_class_2,
                                         relative_class_stats=stats_class_1,
                                         thresh=thresh).tags
    return class_1_tags, class_2_tags


def prepare_parallel_data_tagger(outdir, style_0_label, style_1_label, is_unimodal):
    subprocess.check_call(f"scripts/prep_tagger.sh {outdir} {outdir} tagged {int(is_unimodal)} {style_0_label} {style_1_label}",
                          shell=True)


def prepare_parallel_data_generator(outdir, style_0_label, style_1_label, is_unimodal):
    # "${MASKED_OP_DIR}" "${MASKED_OP_DIR}" "$prefix"masked "$prefix"unmasked "$isunimodal" "$posmask" "$negmask"
    subprocess.check_call(f"scripts/prep_generator.sh {outdir} {outdir} tagged generated {int(is_unimodal)} {style_0_label} {style_1_label}",
                          shell=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    args = docopt(__doc__)
    is_unimodal = int(args["--is_unimodal"] == "True")

    # step 1: generate attribute markers, tagged dataset
    tag_style_markers(data_pth=args["--data_pth"],
                      outpath=args["--outpath"],
                      style_0_label=args["--style_0_label"],
                      style_1_label=args["--style_1_label"],
                      thresh=float(args["--thresh"]),
                      ngram_range=(int(args["--ngram_range_min"]),
                                   int(args["--ngram_range_max"])),
                      style_label_col=args["--style_label_col"])

    

    # step 2: generate parallel dataset for the tagger
    prepare_parallel_data_tagger(
        args["--outpath"], args["--style_0_label"], args["--style_1_label"], is_unimodal)

    # step 3: generate parallel dataset for the generator
    prepare_parallel_data_generator(
        args["--outpath"], args["--style_0_label"], args["--style_1_label"], is_unimodal)
