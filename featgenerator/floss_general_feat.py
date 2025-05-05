import json
import os
import re
import unicodedata
import warnings

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import textstat
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from .config import Config

warnings.filterwarnings("ignore")


class Util:
    def lexical_features(self, test_data):
        def get_lexical_feature(method_name, value):
            try:
                method = getattr(textstat, method_name)
                return method(value)
            # Attribute error means function does not exist in textstat
            except (ValueError, TypeError, ZeroDivisionError, AttributeError):
                return 0.0
        method_names = {
            "automated_readability_index": 0,
            "dale_chall_readability_score": 0,
            "difficult_words": 0,
            "monosyllabcount": 0,
            "syllable_count": 0
        }
        for k in method_names:
            method_names[k] = get_lexical_feature(k, test_data)
        return method_names

class FlossFeatures():
    def __init__(self):
        conf = Config()
        self.root_dir = conf.get_root_dir()
        self.floss_filename = conf.get_floss_file()
        self.model_path = conf.get_string_model()

    def get_dataset(self, root_dir, floss_filename):
        hashes_df = []
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                file_hash = os.path.basename(os.path.normpath(subdir))
                if file_hash != file:
                    continue
                floss_file = os.path.join(subdir, floss_filename)
                hashed_obj = {
                    "hash": file_hash,
                    "strings": set()
                }
                if not os.path.isfile(floss_file):
                    raise ValueError(f"error: hash {file_hash} doesn't have a floss file at {floss_file}")
                try:
                    with open(floss_file) as f:
                        data = json.load(f)
                        all_strings = data.get("strings", {})
                        if len(all_strings) == 0:
                            raise ValueError("error: could not find strings in floss file")
                        static_strings = all_strings.get("static_strings", [])
                        raw_static_strings = {static_string.get("string").strip() for static_string in static_strings}
                        hashed_obj['strings'] = raw_static_strings
                    hashes_df.append(hashed_obj)
                except Exception as e:
                    raise ValueError(f"error: could not process hash {file_hash}: {e}")
        return hashes_df

    def text_processing(self, inpx):
        strings_without_numbers = []
        for each_str in inpx:
            has_string = re.search(r"\s", each_str.strip().rstrip().lstrip())
            if not has_string:
                continue
            x = each_str.split()
            if len(each_str) <= 1:
                continue
            tmp_res = []
            for i in x:
                """
                Attempt to convert the object ot an floating if you get a 
                value error it implies that the ojbect cannot be an floating.
                Should work for integers as well. This will still allow us 
                to keep things like IP address and version numbers 1.1.3
                """
                try:
                    float(i)
                    break
                except ValueError:
                    pass
                if len(i) > 1:
                    tmp_res.append(i)
            if len(tmp_res) <= 1:
                break
            tmp_stringified = " ".join(tmp_res)
            if textstat.automated_readability_index(tmp_stringified) >= 250:
                break
            if textstat.difficult_words(tmp_stringified) >= 10:
                break
            strings_without_numbers.append(tmp_stringified)
        return "||".join(strings_without_numbers)

    def text_cleaner(self,s):
        s = re.sub('[!()’\-\[\]«»“{};:€\'",+=<>/?@#$|%^&\n\t—*–_~…]', '', s)
        s = re.sub('\.{2,10}', '', s)
        s = re.sub('[\s\.]{3,}', '', s)
        s = re.sub('[\s]{2,}', '', s)
        s = re.sub('( \. )+', '', s)
        s = re.sub('(\.){2,}', '', s)
        s = re.sub(' [α-ω]{1,3}(\.) ', '', s)
        s = re.sub(' [α-ω]\.[α-ω]\. ', '', s)
        s = re.sub(' [α-ω]\.[α-ω]\.[α-ω]\. ', '', s)
        s = re.sub(' +', '', s)
        s = re.sub('^ ', '', s)
        s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn').lower()
        return s

    def get_features(self):
        """
            We join all the strings from the floss results together with a full stop and space. 
            This allows us to simulate a proper sentence, for NLP purposes. For each sample 
            strings X_1, X_2 ... X_N become X_1. X_2. ... X_N.
            
            Pipeline:
                ('vect', TfidfVectorizer(sublinear_tf=True,
                                        max_df = 0.50, analyzer="word",stop_words="english", max_features=1000000)),
                ('tfidf', TfidfTransformer()),
                ('svd', TruncatedSVD()),
                ('normalizer', Normalizer())
        """
        hashed_obj = self.get_dataset(self.root_dir, self.floss_filename)
        if len(hashed_obj) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(hashed_obj)
        df_features = df[['hash', 'strings']]
        model = joblib.load(self.model_path)
        df_features['strings'] = df_features['strings'].apply(lambda x: ". ".join(x))
        trained_strings = model.transform(df_features.strings)
        df_features.drop(columns=['strings'], inplace=True)
        normalized_strings = pd.DataFrame(trained_strings)
        df_features = df_features.join(normalized_strings)
        return df_features

    def get_all_features(self):
        hashed_obj = self.get_dataset(self.root_dir, self.floss_filename)
        if len(hashed_obj) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(hashed_obj)
        df_features = df[['hash', 'strings']]
        df_features['strings_with_spaces'] = df_features['strings'].apply(
            lambda x: self.text_processing(x))
        df_features['lexical_strings'] = df_features['strings_with_spaces'].apply(
            lambda x: Util().lexical_features(x))
        df_features = df_features.join(pd.json_normalize(df_features[['lexical_strings']].to_dict(
            orient="records"), record_prefix="strings", meta_prefix="strings"))
        df_features.drop(columns=['lexical_strings'], inplace=True)
        result_vector = []
        vfunc = np.vectorize(self.text_cleaner)
        string_arr = []
        for raw_string in df_features.strings:
            if len(raw_string) != 0:
                str_temp = []
                for temp in raw_string:
                    str_temp.append(vfunc(temp))
                ring1 = ". ".join(raw_string)
            else:
                ring1 = " "
            string_arr.append(ring1)

        pipeline = Pipeline([
            ('vect', TfidfVectorizer(sublinear_tf=True,
                                    max_df = 0.50, analyzer="word",stop_words="english", max_features=1000000)),
            ('tfidf', TfidfTransformer()),
            ('svd', TruncatedSVD()),
            ('normalizer', Normalizer())
        ])
        pipeline.set_params()
        A = pipeline.fit_transform(string_arr)
        normalized_strings = pd.DataFrame(A)
        df_features.merge(normalized_strings)

        return df_features


class custom_tfidf(BaseEstimator,TransformerMixin):
    def __init__(self,tfidf):
        self.tfidf = tfidf

    def fit(self, X, y=None):
        joined_X = X.apply(lambda x: ' '.join(x), axis=1)
        self.tfidf.fit(joined_X)        
        return self

    def transform(self, X):
        joined_X = X.apply(lambda x: ' '.join(x), axis=1)

        return self.tfidf.transform(joined_X)   