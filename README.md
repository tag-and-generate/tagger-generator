# Tagger and Generator

## Dataset preparation: [tag-and-generate/tagger-generator/tag-and-generate-data-prep](https://github.com/tag-and-generate/tagger-generator/tree/master/tag-and-generate-data-prep)
## Training, inference, evaluation: [tag-and-generate/tagger-generator/tag-and-generate-train](https://github.com/tag-and-generate/tagger-generator/tree/master/tag-and-generate-train)

--- 

## Walkthrough 
We will now present an example of training the politeness transfer system from scratch.
The process has five steps:
  * [Step 1: Getting the code](#step-1-getting-the-code)
  * [Step 2: Getting the training data](#step-2-getting-the-training-data)
  * [Step 3: Preparing parallel data for training](#step-3-preparing-parallel-data-for-training)
  * [Step 4: Training the tagger and generator](#step-4-training-the-tagger-and-generator)
  * [Step 5: Running inference](#step-5-running-inference)
  
### Step 1: Getting the code

We begin by cloning this repo:

```sh
git clone https://github.com/tag-and-generate/tagger-generator.git
```
The cloned folder contains: i) ``tag-and-generate-data-prep`` the codebase used for creating the parallel tag and generate dataset, and ii) ``tag-and-generate-train``, the training code.

Each of these folders has a ``requirements.txt`` file that can be used to download the dependencies.

Next, let's create a folder inside ``tagger-generator`` to save all the datasets/tags:

```sh
cd tagger-generator
mkdir data
```


### Step 2: Getting the training data.

The training data in a ready to use format is located [here](https://drive.google.com/file/d/1E9GHwmVM9DL9-KiaIaG5lm_oagLWe908/view?usp=sharing).

Download the zip file to the ``data`` folder created above and extract ```politeness.tsv```.

```sh
unzip politeness_processed.zip
head politeness.tsv
```
**txt**|**style**|**split**
-----|-----|-----
forwarded by tana jones / hou / ect on 09/28/2000|P\_2|train
the clickpaper approvals for 9/27/00 are attached below .|P\_7|train
"hello everyone : please let me know if you have a subscription to "" telerate "" ?"|P\_7|train
we are being billed for this service and i do not know who is using it .|P\_0|train

As we can see, the data is in the tsv format and has the right header.


You can also use ``gdown`` to directly download the file:

```sh
gdown --id 1E9GHwmVM9DL9-KiaIaG5lm_oagLWe908
```






Now that we have the codebase and the dataset, let's start by creating the parallel data required for training the models. Let's do a listing of the folder so far to make sure we are on the same page:

```sh
(dl) tutorial@sa:~/tagger-generator$ ls
data  LICENSE  README.md  tag-and-generate-data-prep  tag-and-generate-train
```
So, we are in the repo (tagger-generator), and see the two code folders (``tag-and-generate-data-prep`` and ``tag-and-generate-train``), as well as the data folder (``data``).
Further, the data folder has the ``politeness.tsv`` file that we just downloaded:
```sh
(dl) tutorial@sa:~/tagger-generator$ ls data/
politeness_processed.zip  politeness.tsv
```

### Step 3: Preparing parallel data for training

We prepare the parallel data using ``tag-and-generate-data-prep``:

```sh
cd tag-and-generate-data-prep
python src/run.py --data_pth ../data/politeness.tsv --outpath ../data/ --style_0_label P_9 --style_1_label P_0 --is_unimodal True
```
More details on these options are located in [tag-and-generate/tagger-generator/tag-and-generate-data-prep](https://github.com/tag-and-generate/tagger-generator/tree/master/tag-and-generate-data-prep). In summary, we specify the input file, the label for the style of interest (``P_9``) and a neutral/contrastive style (``P_0``). Importantly, we specify ``--is_unimodal True``. This option ensures that the parallel data is created as per the unimodal style setting (Figure 3 in [the paper](https://arxiv.org/pdf/2004.14257.pdf)).

After data-prep finishes, we see several files in ``../data/``.
The important files are described below:

* P_9_tags.json: these are the politeness tags or phrases identified as polite phrases:

```"thank you"
"thank"
"looking forward"
"glad"
"be interested"
```

* The data prep code creates two sets of training files: one for the ``tagger`` and another for the ``generator``. 
To understand these, let's take a sample sentence ```please get back to me if you have any additional concerns .``` and look at how it is represented in different files:

    - ``entagged_parallel.train.en`` (input to the tagger):
        -  ``back to me have concerns .``
    - ``entagged_parallel.train.tagged`` (output of the tagger): 
        - ``[P_90] back to me [P_91] have [P_92] concerns .``
    - ``engenerated_parallel.train.en`` (input to the generator):
        - ``[P_90] back to me [P_91] have [P_92] concerns .``
    -  ``engenerated_parallel.train.generated`` (output of the generator)
        - ``please get back to me if you have any additional concerns .``

    Here, ``P_9`` is the style tag, and the number after the style tag captures the position of the tag in the sentence.

With the data files ready, we are ready to run training.


### Step 4: Training the tagger and generator

All the training and inference related scripts/code is present in ``tag-and-generate-train``, so let's ``cd`` to it.

```sh
cd tag-and-generate-train
```

In order to prepare the files for training, we first process them using ``BPE. ``

```sh
bash scripts/prepare_bpe.sh tagged ../data/
bash scripts/prepare_bpe.sh generated ../data/
```

We can now start training the tagger and generator:

```sh
nohup bash scripts/train_tagger.sh tagged politeness ../data/ > tagger.log &
nohup bash scripts/train_generator.sh generated politeness ../data/ > generator.log &
```

```politeness``` is a user-defined handle that we will use during inference. 

After the training finishes, the best models (given by validation perplexity) are stored in ``models``:

```sh
(dl) tutorial@sa:~/tagger-generator/tag-and-generate-train$ ls models/politeness/bpe/
en-generated-generator.pt  en-tagged-tagger.pt
```

For our run, at the end of 5 epochs, the validation perplexity was 1.26 for the tagger, and 1.76 for the generator.

### Step 5: Running inference

Let's test out the trained models on some sample sentences:

```sh
(dl) tutorial@sa:~/tagger-generator/tag-and-generate-train$ cat > input.txt
send me the text files.
look into this issue.

bash scripts/inference.sh input.txt sample tagged generated politeness P_9 P_9 ../data/ 3
```

Here ``sample`` is a unique identifier for the inference job, and ``politeness`` is the identifier we used for the training job. ``P_9`` is the style tag (kept the same for unimodal jobs). (Please see the README at [tag-and-generate/tagger-generator/tag-and-generate-train](https://github.com/tag-and-generate/tagger-generator/tree/master/tag-and-generate-train) for more details).

The final and intermediate outputs are located in experiments folder:

```sh
(dl) tutorial@sa:~/tagger-generator/tag-and-generate-train$ ls experiments/sample_*
experiments/sample_generator_input  experiments/sample_tagged
experiments/sample_output       experiments/sample_tagger_input
```

Let's look at the final output:

```sh
(dl) tutorial@sa:~/tagger-generator/tag-and-generate-train$ cat experiments/sample_output 
please send me the text files.
we would like to look into this issue.
```
Not bad! 

We hope this walkthrough is helpful in understanding and using the codebase. Here are some additional helpful links:

- [Trained Models](https://drive.google.com/drive/folders/1tXLC4WbXc_WLgvQu2mTa3jDe0efZ3dz1?usp=sharing).
- [Outputs](https://github.com/tag-and-generate/outputs)
- [Datasets](https://github.com/tag-and-generate/politeness-dataset)


