"""Generates the tags for training the tager
"""
import pandas as pd
from tqdm.autonotebook import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy
import scipy.sparse
from collections import namedtuple
import pickle
import tqdm
import numpy as np
import csv
from tqdm import tqdm
import logging
tqdm.pandas()

TFIDFStats = namedtuple("TFIDFStats", ["data_id", "id_to_word", "word_to_id", "tfidf_avg", "word_to_idf", "counts"])


class TrainDataGen:
    """
    Generates the training data
    """
    def __init__(self, data, outpath, tags, tag_token, tgt_lang):
        super().__init__()
        self.data = data
        
        self.outpath = outpath
        self.tag_token = tag_token
        self.tags = tags
        self.tgt_lang = tgt_lang
    
    def generate(self):
        self.tag_and_dump(split="train")
        self.tag_and_dump(split="test")
        self.tag_and_dump(split="val")
    

    def tag_and_dump(self, split):
        """Iterate over the given split, tags the sentences and write out the data
        
        Arguments:
            split {[str]} -- [description]
        """
        orig_sents, taged_sents = [], []
        data_in = self.data[self.data["split"] == split]
        for _, r in data_in.iterrows():
            orig = r["txt"].strip().replace("\n", "")
            orig_sents.append(orig)
            taged_sents.append(TrainDataGen.tag_sentence(orig, self.tags, self.tag_token).strip().replace("\n", ""))
            #polite_out.write(f"{orig}\n")
            #polite_taged_out.write(f"{taged_sent}\n")
        with open(f"{self.outpath}/en{self.tgt_lang}_parallel.{split}.en.{self.tag_token}", "w") as orig_out,\
             open(f"{self.outpath}/en{self.tgt_lang}_parallel.{split}.{self.tgt_lang}.{self.tag_token}", "w") as taged_out:
            for orig, taged in tqdm(zip(orig_sents, taged_sents), total=len(taged_sents)):
                if self.tag_token in taged:
                    ### ONLY WRITE OUT THE tagED DATA
                    orig_out.write(f"{orig.strip()}\n")
                    taged_out.write(f"{taged.strip()}\n") 

    def tag_and_dump_batched(self, split):
        """Iterate over the given split, tags the sentences and write out the data
        
        Arguments:
            split {[str]} -- [description]
        """
        orig_sents, taged_sents = [], []
        self.data["txt_taged"] = self.data["txt"].progress_apply(lambda x: \
                                                                  TrainDataGen.tag_sentence(orig, self.tags,\
                                                                                             self.tag_token).strip().replace("\n", ""))
        
        
        with open(f"{self.outpath}/en{self.tgt_lang}_parallel.{split}.en.{self.tag_token}]", "w") as orig_out,\
             open(f"{self.outpath}/en{self.tgt_lang}_parallel.{split}.{self.tgt_lang}.{self.tag_token}]", "w") as taged_out:
            for orig, taged in tqdm(zip(orig_sents, taged_sents), total=len(taged_sents)):
                if self.tag_token in taged:
                    ### ONLY WRITE OUT THE tagED DATA
                    orig_out.write(f"{orig.strip()}\n")
                    taged_out.write(f"{taged.strip()}\n") 

    @staticmethod
    def tag_sentence(sent, tag_dict, tag_token,
                      pos_weight: int = 3,
                      max_pos_indicator: int = 20,
                      concat = True):
        """Given a sentence and a dictionary from 
        tag_value to tag_probability, replaces all the words mw that are in the tag_dict
        with a probability tag_dict[mw]
        
        Arguments:
            sent {[str]}       -- [the given sentence]
            tag_dict {[dict]} -- [the tag dictionary]
            tag_token {[str]} -- [the taging token]
            dont_concat        -- [do not concat]
        
        Returns:
            [str] -- [the taged sentence]
        """
        i = 0
        sent = sent.split()
        taged_sent = []
        prev_tag = False
        while i < len(sent):
            loc = min(i // pos_weight, max_pos_indicator)
            key_bi_gram = " ".join(sent[i: i + 2])
            key_tri_gram = " ".join(sent[i: i + 3])
            key_quad_gram = " ".join(sent[i: i + 4])
            
            if key_quad_gram in tag_dict and np.random.rand() < tag_dict[key_quad_gram]:
                if not concat or not prev_tag:
                    taged_sent.append(f"[{tag_token}{loc}]")
                prev_tag = True
                i += 4

            elif key_tri_gram in tag_dict and np.random.rand() < tag_dict[key_tri_gram]:
                if not concat or not prev_tag:
                    taged_sent.append(f"[{tag_token}{loc}]")
                prev_tag = True
                i += 3
            elif key_bi_gram in tag_dict and np.random.rand() < tag_dict[key_bi_gram]:
                if not concat or not prev_tag:
                    taged_sent.append(f"[{tag_token}{loc}]")
                prev_tag = True
                i += 2
            elif sent[i] in tag_dict and np.random.rand()< tag_dict[sent[i]]:
                if not concat or not prev_tag:
                    taged_sent.append(f"[{tag_token}{loc}]")
                prev_tag = True
                i += 1
            else:
                taged_sent.append(sent[i])
                prev_tag = False
                i += 1
        return " ".join(taged_sent)

class TFIDFStatsGenerator:

    def __init__(self, data, data_id, ngram_range):
        super().__init__()
        self.ngram_range = ngram_range
        self.data_id = data_id
        self.data = data
        self.generate()

    def get_word_counts(self):
        """Generates the counts for various n-grams for the given corpus
        
        Returns:
            a dictionary from phrase to word count
        """
        cv = CountVectorizer(ngram_range=self.ngram_range)
        cv_fit = cv.fit_transform(self.data)
        feature_names = cv.get_feature_names() 
        X = np.asarray(cv_fit.sum(axis=0)) # sum counts across sentences
        word_to_id = {feature_names[i]: i for i in range(len(cv.get_feature_names()))}
        word_count = {}
        for w in word_to_id:
            word_count[w] = X[0, word_to_id[w]]
        return word_count

    def generate(self):
        """Generates various TFIDF related stats
        for the given data and wraps them in a namedtuple
        
        Returns:
            [type] -- [description]
        """
        logging.info("Running TfidfVectorizer")
        vectorizer = TfidfVectorizer(ngram_range=self.ngram_range)
        X = vectorizer.fit_transform(self.data)
        feature_names = vectorizer.get_feature_names()
        id_to_word = {i: feature_names[i] for i in range(len(vectorizer.get_feature_names()))}
        word_to_id = {v: k for k, v in id_to_word.items()}
        X = np.asarray(X.mean(axis=0)).squeeze(0) # / num_docs
       
        idf = vectorizer.idf_
        counts = self.get_word_counts()
        word_to_idf = dict(zip(feature_names, idf))

        self.id_to_word = id_to_word
        self.word_to_id = word_to_id
        self.tfidf_avg = X
        self.word_to_idf = word_to_idf
        self.counts = counts


class RelativeTagsGenerator:

    def __init__(self, main_class_stats, relative_class_stats,
                 min_freq: int = 2, thresh: float = 0.90,
                 ignore_from_tags = None):
        """Generates tags for the main class relative to 
        the relative class. This is done on the basis of relative TF-IDF ratios of the words.

        Arguments:
            main_class_stats {[type]} -- [description]
            ref_class_stats {[type]} -- [description]
        
        Keyword Arguments:
            min_freq {int} -- [Minimum freq in the main class for the phrase to be considered] (default: {1})
            thresh {float} -- [The relative tf-idf scores are converted to percentiles. These percentiles are then
                               used to select the tag phrases. In this case, the cutoff for such phrases is 0.90] (default: {0.90})
            ignore_from_tags {[set]} -- [Set of words like the NER words, which might have to be ignored] (default: {None})
        """
        super().__init__()
        self.main_class_stats = main_class_stats
        self.relative_class_stats = relative_class_stats
        self.min_freq = min_freq
        self.c1_tag = main_class_stats.data_id
        self.c2_tag = relative_class_stats.data_id
        self.thresh = thresh
        self.ignore_from_tags = ignore_from_tags

        self.generate_tfidf_report()
        self.generate_relative_tags()
        

    def generate_tfidf_report(self):
        """Given TFIDF statistics on two datasets, returns a common tf-idf report. 
        The report measures various statistics on the words that appear in class_2
        
        Arguments:
            class1_tfidf_report {[TFIDFStats]} -- [TFIDFStats for class1]
            class2_tfidf_report {[TFIDFStats]} -- [TFIDFStats for class2]
        """
        report = []
        for word in self.main_class_stats.word_to_id.keys():
            if self.main_class_stats.counts[word] >= self.min_freq and word in self.relative_class_stats.word_to_id:
                    res = {}
                    res["word"] = word
                    res["freq"] = self.main_class_stats.counts[word]
                    res[f"{self.c1_tag}_mean_tfidf"] = self.main_class_stats.tfidf_avg[self.main_class_stats.word_to_id[word]]
                    res[f"{self.c2_tag}_mean_tfidf"] = self.relative_class_stats.tfidf_avg[self.relative_class_stats.word_to_id[word]]
                    res[f"{self.c1_tag}_idf"] = self.main_class_stats.word_to_idf[word]
                    res[f"{self.c2_tag}_idf"] = self.relative_class_stats.word_to_idf[word]
                    report.append(res) 
        self.report = pd.DataFrame(report)

    def generate_relative_tags(self):
        """Returns a dictionary of phrases that are important in class1 relative to
        class2
        """
        c1_over_c2 = f"{self.c1_tag}_over_{self.c2_tag}"
        c2_over_c1 = f"{self.c2_tag}_over_{self.c1_tag}"
        # tfidf_report["np_over_p"] = (tfidf_report["np_mean_tfidf"] / len(data_p_0)) / (tfidf_report["p_mean_tfidf"] /  len(data_p_9))
        self.report[c1_over_c2] = self.report[f"{self.c1_tag}_mean_tfidf"] / self.report[f"{self.c2_tag}_mean_tfidf"] #ratio of tf-idf in the two corpora

        self.report[c2_over_c1] = 1 / self.report[c1_over_c2]

        self.report[f"{self.c1_tag}_tag"] = (self.report[c1_over_c2] / self.report[c1_over_c2].sum()) ** 0.75
        # ^ add support for the small values

        self.report[f"{self.c1_tag}_tag"] = self.report[f"{self.c1_tag}_tag"] / self.report[f"{self.c1_tag}_tag"].sum()
        # ^ make a probability
        
        self.report.sort_values(by=f"{self.c1_tag}_tag", ascending=False, inplace=True)
        self.report['rank'] = self.report[f"{self.c1_tag}_tag"].rank(pct=True)
        # ^ assign percentile


        important_phrases = self.report[self.report["rank"] >= self.thresh]
        # ^ only take phrases that clear the threshold (default: 0.9)

        important_phrases["score"] = (important_phrases["rank"] - self.thresh) / (1 - self.thresh) 
        # ^ make a distribution again
        
        tags= {}
        for i, r in important_phrases.iterrows():
            tags[r["word"]] = r["score"]
        
        self.tags = tags

        if self.ignore_from_tags is not None:
            logging.info("Ignoring tags")
            self.tags = self.filter_tags_with_ignored_entities()

    def filter_tags_with_ignored_entities(self):
        res = {}
        for k, v in self.tags.items():
            if not any(k_part in self.ignore_from_tags for k_part in k.split()):
                res[k] = v
        return res
