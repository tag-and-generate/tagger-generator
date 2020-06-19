## Data Preparation

- This repository contains the code for creating parallel data that can be used to train the ``tagger`` and ``generator`` modules on your dataset https://arxiv.org/abs/2004.14257

- The `data/catcher/` directory contains some sample text that can be used to test the codebase.

## Usage


```py
python src/run.py --data_pth PTH\
                  --outpath OUTPATH\
                  --style_0_label label0\
                  --style_1_label label1\
                  --is_unimodal True | False
```

Where:
- `PTH` should point to a tab-separated file (tsv) that contains the corpus. We assume that the corpus is made up of a set of sentences. The `tsv` is supposed to have three fields: 1) txt: the sentence, 2) split: train/test/val, and 3) style: label that identifies the style of the sentence (one of `style_0_label` or `style_1_label`)

- `OUTPATH` is the location of the output

- `label0` and `label1` are tags that identify individual styles. This explicit assignment is important for unimodal cases, such as politeness and captions (please see the paper for more details)

- `is_unimodal` should be set to `True` for datasets that have only one stylistic information 
Please see run.py for the details on other options.

## Outputs

While the program creates a number of files in the `OUTPATH` dir, only a subset of them are required for training `tagger` and `generator`. All of the files are named according to the following format:

`en{target}\_parallel.{split}.[en | {target}]`

Where `split` is either `train`, `test`, or `val`, and `target` is either set to `tagged` (for tagger) or `generated`) for generator. We always use `en` to refer to the source files.

Further, the attribute tags can also be found under the name `{style_label}_tags.json`

## Walkthrough 

We walk through the usage of the data prep codebase by creating parallel data for our tag and generate system using the sample data present in `data/catcher`. The (toy) data consists of a few lines from the Catcher in the Rye and Romeo & Juliet. 

Some sample rows from the dataset are shown below:

| txt 	| style 	| split 	|
|-	|-	|-	|
| How've you been, Mrs. Spencer? 	| catcher 	| test 	|
| C'mon, c'mon 	| catcher 	| train 	|
| And the place death, considering who thou art, 	| romeo-juliet 	| train 	|
| He's got a lot of dough, now. 	| catcher 	| test 	|
| My life were better ended by their hate, 	| romeo-juliet 	| train 	|
| He lent me counsel and I lent him eyes. 	| romeo-juliet 	| train 	|
| It wasn't all my fault. 	| catcher 	| test 	|
| If you don't, you feel even worse. 	| catcher 	| test 	|


Using the defaults specified in src/run.py, we can generate the parallel data for training tag and generator using the following command:

```py
python3 src/run.py --data_pth data/catcher/data.tsv\
                   --outpath data/tmp/\
                   --style_0_label romeo-juliet\
                   --style_1_label catcher\
                   --is_unimodal False
```

After running this command, the specified output directory `data/tmp` will contain a number of files. The important ones are listed below.

- Style attribute tags:
    - `romeo-juliet_tags.json`: The style tags for style 0 (romeo-juliet)
    - `catcher_tags.json`: The style tags for style 1 (catcher)

- Tagger training files:
    - `entagged_parallel.[train|test|val].en`: Source files for the tagger 
    - `entagged_parallel.[train|test|val].tagged`: Target files for the tagger 

- Generator training files:
    - `engenerated_parallel.[train|test|val].en`: Source files for the generator 
    - `engenerated_parallel.[train|test|val].generated`: Target files for the generator
