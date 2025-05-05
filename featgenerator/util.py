import concurrent
import concurrent.futures
import csv
import hashlib
import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from ipaddress import IPv4Address, IPv6Address
from typing import Any, Optional, Tuple, Union
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import regex
import textstat
import torch
import torch.nn.functional as torch_ff
import tqdm
from datasketch import MinHash, MinHashLSH
from itables import init_notebook_mode, show
from itables.options import classes
from nltk.corpus import words
from scipy.signal import normalize
from scipy.sparse import lil_matrix
from scipy.spatial.distance import cdist
from sklearn.base import ClusterMixin
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, KMeans
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.metrics import auc
from sklearn.metrics import auc as auc_calc
from sklearn.metrics import (
    classification_report,
    davies_bouldin_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
    v_measure_score,
)
from sklearn.preprocessing import (
    LabelEncoder,
    MultiLabelBinarizer,
    QuantileTransformer,
    label_binarize,
    normalize,
)
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    BatchEncoding,
    BertModel,
    BertTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from .floss_general_feat import FlossFeatures


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
    

    def get_sentence_tok_model(self) -> Tuple[AutoModel, AutoTokenizer]:
        """Returns the sentence transformer model and tokenizer for the migration task.
        
        Returns:
            Tuple[AutoModel, AutoTokenizer]: The model and tokenizer for the migration task.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
        )
        model = AutoModel.from_pretrained("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        return model, tokenizer

    def list_str_feat(self, dSeries, prefix):
        def normalize_string(inpstr):
            return prefix+"-"+re.sub("\s{1,}", "", inpstr.strip().lstrip().rstrip().replace("\n",""))
        
        allkeys = {normalize_string(el) for lst in dSeries for el in lst if len(el) > 0}
        res = []
        for row in dSeries:
            tD = {}
            for key in allkeys:
                if len(key) == 0:
                    continue
                tD[key] = int(0)
            for el in set(row):
                el = normalize_string(el)
                if tD.get(key):
                    tD.setdefault(el, int(1))
                else:
                    tD.setdefault(el, int(1))
            res.append(tD)
        df = pd.DataFrame(res)
        df = df.fillna(0)
        for col in df.columns:
            df[col] = df[col].astype(int)
        return df
    
    def get_dummies_for_columns(self, df, columns_list):
        # Ensure that the columns_list is a list; if not, convert it to a list
        if not isinstance(columns_list, list):
            columns_list = [columns_list]
        
        # Initialize an empty DataFrame for results
        result_df = pd.DataFrame(index=df.index)  # Use the original DataFrame's index

        # Loop through each column in columns_list and create dummy variables for it
        for column in columns_list:
            # Generate dummy variables for the current column
            dummies = pd.get_dummies(df[column], prefix=column, dtype=int, dummy_na=False)
            # Concatenate the dummy variables with the result_df
            result_df = pd.concat([result_df, dummies], axis=1)
        
        return result_df

    def normalize_string(self, inpstr, prefix):
        """Normalize string by removing spaces, newlines, and adding prefix."""
        return prefix + "-" + re.sub(r"\s+", "", inpstr.strip())

    def list_to_categorical_feat(self, dSeries, prefix):
        # Generate all unique keys across all rows after normalization
        allkeys = {self.normalize_string(el, prefix) for lst in dSeries for el in lst if el}

        # Initialize result list to store row dictionaries
        res = []
        for row in dSeries:
            # Initialize a dictionary for the current row with all keys set to 0
            tD = {key: 0 for key in allkeys}
            # Update dictionary for elements present in the row
            for el in set(row):
                if el:  # Check if element is not empty
                    normalized_el = self.normalize_string(el, prefix)
                    tD[normalized_el] = 1
            res.append(tD)

        # Create DataFrame from the list of dictionaries
        return pd.DataFrame(res)

    def convert_list_columns_to_categorical(self, df, columns_list):
        result_df = pd.DataFrame(index=df.index)  # Use the original DataFrame's index
        for column in columns_list:
            categorical_df = self.list_to_categorical_feat(df[column], column)
            result_df = pd.concat([result_df, categorical_df], axis=1)
        
        return result_df



    def process_raw_strings_dataset(self, root_dir, floss_filename, hash_list, max_features=1000000):
        """
        Processes raw strings dataset to generate a normalized character matrix.
        
        Parameters:
        - root_dir: Directory where the dataset is located.
        - floss_filename: Name of the file containing the dataset.
        - hash_list: List of hash values to filter the dataset.
        - max_features: Maximum number of features for vectorization (default is 1,000,000).
        
        Returns:
        - A DataFrame containing the normalized character matrix of processed strings.
        """
        # Load dataset
        floss_feat = FlossFeatures()
        raw_strings_dataset = floss_feat.get_dataset(root_dir, floss_filename)
        raw_string_df = pd.DataFrame(raw_strings_dataset)[['hash', 'strings']]
        
        # Filter dataset based on hash_list
        raw_string_df = raw_string_df[raw_string_df['hash'].isin(hash_list)]
        
        # Process strings
        string_processor = StringProcessing()
        raw_string_df['filtered_strings'] = raw_string_df['strings'].apply(
            lambda x: " , ".join(string_processor.process_strings(strings=list(x)))
        )
        
        # Convert to character matrix
        filtered_strings_list = raw_string_df['filtered_strings'].tolist()
        vectorizer = CountVectorizer(analyzer='char', stop_words='english', ngram_range=(1, 3), max_features=max_features)
        character_matrix = vectorizer.fit_transform(filtered_strings_list)
        
        # Normalize matrix
        normalized_matrix = normalize(character_matrix)
        
        # Convert to DataFrame and return
        embedding_df = pd.DataFrame(normalized_matrix.toarray(), columns=vectorizer.get_feature_names_out())
        return embedding_df


    @cache
    def is_popular_domain(self, url, top_domains=r'data\top_500_domains.csv'):
        is_top_domain = False
        try:
            domain_address = urlparse(url).netloc
        except Exception:
            return is_top_domain
        if domain_address == "":
            return is_top_domain    
        list_of_popular_domain = []
        file = open(top_domains)
        csvreader = csv.reader(file)
        header = []
        header = next(csvreader)
        for row in csvreader:
            list_of_popular_domain.append(row[1])
        for top_domain in list_of_popular_domain:
            if (top_domain in domain_address) or (domain_address in top_domain):
                is_top_domain = True
                break
        return is_top_domain

    def get_features_from_function_lists(self, exported_func_list, prefix):
        exported_func_dist = dict()
        list_of_list = [set(i) for i in exported_func_list]
        for item in list_of_list:
            for i in item:
                cnt = exported_func_dist.get(i)
                if not cnt:
                    exported_func_dist.setdefault(i, 1)
                if cnt:
                    exported_func_dist[i] = cnt + 1

        one_occurrence = dict()
        more_than_one_occurence = dict()
        for exported_function in exported_func_dist:
            func_val = exported_func_dist.get(exported_function)
            if func_val == 1:
                one_occurrence[exported_function] = func_val
            if func_val > 5:
                more_than_one_occurence[exported_function] = func_val

        res = []
        for exported_functions in exported_func_list:
            result_dict = more_than_one_occurence.copy()
            occurences = '_low_total_occurences'
            presence =  '_low_occurence_'
            result_dict.setdefault(presence,0)
            result_dict.setdefault(occurences,0)
            set_of_our_functions = set(exported_functions)
            for _function in result_dict:
                if _function not in set_of_our_functions:
                    result_dict[_function] = 0

            for exported_function in exported_functions:
                highly_occ_func = more_than_one_occurence.get(exported_function)
                lowly_occ_func = one_occurrence.get(exported_function)
                if lowly_occ_func and not highly_occ_func:
                    result_dict[presence]  = 1
                    result_dict[occurences] += 1
                if highly_occ_func:
                    result_dict[exported_function] = 1

            result_dict = {f"{prefix}_{i}":result_dict[i] for i in result_dict}
            res.append(result_dict)
        return res
    
    def get_reports(self, report_path="collection_from_vt.json"):
        collect_res = pd.read_json(report_path)
        collections_results = json.loads(collect_res.to_json(orient="records"))
        maps = []
        for k in collections_results:
            a = k.get("data")
            keys = k.get("links").get("self").split("/")[6]
            int_map = {"hash": keys, "reports": set()}
            for val in a:
                report = val.get("attributes").get("name")
                int_map["reports"].add(report)
            maps.append(int_map)
        return pd.DataFrame(maps)
    
    """
    """
    def non_similar_df(to_compare):
        drape_keys = json.loads(to_compare.to_json(orient="records"))
        similar_values = []
        for samp in drape_keys:
            dat_json = {}
            for k in samp:
                val = samp[k]
                try:
                    int(k)
                    continue
                except ValueError:
                    pass
                if val != 0:
                    dat_json[k] = val
            similar_values.append(dat_json)
        return pd.DataFrame(similar_values)

    def silhouette_scorer(estimator: ClusterMixin, X: np.ndarray) -> float:
        y_pred = estimator.fit_predict(X)
        return silhouette_score(X, y_pred)

    def train_autoencoder(self, X: np.ndarray,
                        learning_rate: float = 0.001,
                        batch_size: int = 32,
                        num_epochs: int = 20) -> Tuple[Model, np.ndarray]:
        """
        Trains an autoencoder model on the given dataset X.

        Parameters:
        - X: Input dataset of shape (n_samples, n_features).
        - learning_rate: Learning rate for the optimizer.
        - batch_size: Batch size for training.
        - num_epochs: Number of epochs for training.

        Returns:
        - A tuple containing the trained autoencoder model and the encoded features X_encoded.
        """
        # Create the model architecture
        input_layer = Input(shape=(X.shape[1],))
        hidden_layer1 = Dense(32, activation='relu')(input_layer)
        hidden_layer2 = Dense(16, activation='relu')(hidden_layer1)
        output_layer = Dense(X.shape[1], activation='sigmoid')(hidden_layer2)
        autoencoder = Model(inputs=input_layer, outputs=output_layer)
        
        # Compile the model
        autoencoder.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=learning_rate))
        
        # Train the model
        autoencoder.fit(X, X, batch_size=batch_size, epochs=num_epochs, verbose=0)  # Added verbose=0 for less output during training
        
        # Encode the input data
        X_encoded = autoencoder.predict(X)
        
        return autoencoder, X_encoded

class MinHashLSHForest:

    def series_to_minhash(self, data: pd.Series, n_perm) -> list:
            minhashes = []
            distinct_elements = Counter(item for row in data for item in row)
            label_dict = {item: index for index, item in enumerate(distinct_elements.keys())}

            for row in data:
                if not isinstance(row, (list, np.ndarray)):
                    row = np.array(list(row))
                row = sorted(row)

                encoded_row = [label_dict[item] for item in row]
                minhash = MinHash(num_perm=n_perm)
                for item in encoded_row:
                    minhash.update(str(item).encode("utf8"))
                minhashes.append(minhash)
            return minhashes

    def compute_pairwise_jaccard_similarity(self, series, num_perm, threshold):
        num_rows = len(series)
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        minhashes = {}
        jaccard_cache = {}

        for i in range(num_rows):
            minhash = MinHash(num_perm=num_perm)
            for string in series.iloc[i]:
                minhash.update(string.encode('utf8'))
            lsh.insert(i, minhash)
            minhashes[i] = minhash

        pairwise_similarity = pd.Series(index=series.index, dtype=object)

        for i in range(num_rows):
            if i in jaccard_cache:
                similar_rows = jaccard_cache[i]
            else:
                similar_rows = lsh.query(minhashes[i])
                jaccard_cache[i] = similar_rows

            if len(similar_rows) > 1:
                pairwise_similarity[i] = similar_rows

        return pairwise_similarity

    def add_series_to_dataframe(self, dataframe, series, column_name):
        new_dataframe = dataframe.copy()
        new_dataframe[column_name] = series
        return new_dataframe

    def build_lsh_forest(self, data, num_perm=128, threshold=0.5):
        results = pd.DataFrame()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for column in data.columns:
                future = executor.submit(self.compute_pairwise_jaccard_similarity, data[column], num_perm, threshold)
                futures.append(future)

            for column, future in zip(data.columns, futures):
                res = future.result()
                results = self.add_series_to_dataframe(results, res, column)

        return results

    
    def process_column(self, col, data):
        prefix = f"{col}_"
        unique_items = set()

        for lst in data[col]:
            if isinstance(lst, list):
                unique_items.update(lst)

        exploded_col = []

        for lst in data[col]:
            if not isinstance(lst, list):
                continue
            encoded = [int(item in lst) for item in unique_items]
            exploded_col.append(pd.Series(encoded))

        exploded_col = pd.concat(exploded_col, axis=1)
        exploded_col.columns = [f"{prefix}{i}" for i in unique_items]
        exploded_col = exploded_col.fillna(0).astype(int)

        return exploded_col


    def explode_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        exploded_cols = []
        num_rows = len(data)

        for col in tqdm(data.columns, desc="Exploding Columns"):
            prefix = f"{col}_"
            exploded_col = lil_matrix((num_rows, num_rows), dtype=int)

            for i, lst in enumerate(data[col]):
                if isinstance(lst, list):
                    for item in lst:
                        exploded_col[i, i] = 1

            exploded_col = pd.DataFrame.sparse.from_spmatrix(exploded_col, columns=[f"{prefix}{i}" for i in range(num_rows)])
            exploded_cols.append(exploded_col)

        result_df = pd.concat(exploded_cols, axis=1)
        return result_df

class ClusteringMetrics():
    def __init__(self) -> None:
        pass

    @staticmethod
    def get_auc(truth_matrix, evaluation_column):
        n_classes = np.max(truth_matrix[evaluation_column]) + 1
        y_true = label_binarize(truth_matrix[evaluation_column], classes=np.arange(n_classes))

        y_scores = np.zeros_like(y_true, dtype=np.float64)
        for i, label in enumerate(truth_matrix.majority_vote):
            y_scores[i, label] = 1.0

        auc_values = []
        for class_label in range(n_classes):
            auc = roc_auc_score(y_true[:, class_label], y_scores[:, class_label])
            auc_values.append(auc)

        return np.median(auc_values)

    @staticmethod
    def metrics_using_labels(merged, evaluation_column = "Adversary", label_encode=False) -> None:
        """
            If the label_encode value is true the evalution columns name has a suffix _encoded.
        """
        truth_map = []
        evaluation_column_name = f"{evaluation_column}"
        if label_encode:
            label_encoder = LabelEncoder()
            transformed = label_encoder.fit_transform(merged[evaluation_column] )
            merged[evaluation_column] = transformed
            evaluation_column_name = f"{evaluation_column}_encoded"

        for element in merged.labels.dropna().unique():
            lookup_dataset = merged[(merged['labels'] == element) & (merged['labels'] != -1)]
            if lookup_dataset.empty:
                continue
            popuplar_adversary = lookup_dataset[evaluation_column].value_counts()
            if popuplar_adversary.empty:
                continue
            popuplar_adversary = popuplar_adversary.idxmax()
            for _, item in lookup_dataset.iterrows():
                tp_vals = {
                 "hash": item['hash'],
                 "labels": item['labels'],
                 f"{evaluation_column}": item[evaluation_column],
                 "majority_vote": popuplar_adversary,
                 "TP": 1 if item[evaluation_column] == popuplar_adversary else 0,
                 "FP": 1 if item[evaluation_column] != popuplar_adversary else 0
                }
                truth_map.append(tp_vals)

        truth_matrix = pd.DataFrame(truth_map)
        precision = truth_matrix['TP'].sum() / (truth_matrix['TP'].sum() + truth_matrix['FP'].sum())
        recall = truth_matrix['TP'].sum() / (truth_matrix['TP'].sum() + (len(truth_matrix) - truth_matrix['TP'].sum() - truth_matrix['FP'].sum()))
        return truth_matrix, precision, recall

    @staticmethod
    def metrics_using_ground_truth(merged, evaluation_column = "Adversary"):
        truth_map = []
        for element in merged[evaluation_column].dropna().unique():
            lookup_dataset = merged[(merged[evaluation_column] == element) & (merged['labels'] != -1)]
            if lookup_dataset.empty:
                continue
            popuplar_adversary = lookup_dataset['labels'].value_counts()
            if popuplar_adversary.empty:
                continue
            popuplar_adversary = popuplar_adversary.idxmax()
            for idx, item in lookup_dataset.iterrows():
                tp_vals = {
                 "hash": item['hash'],
                 "labels": item['labels'],
                 "Adversary": item[evaluation_column],
                 "majority_vote": popuplar_adversary,
                 "TP": 1 if item['labels'] == popuplar_adversary else 0,
                 "FP": 1 if item['labels'] != popuplar_adversary else 0
                }
                truth_map.append(tp_vals)

        truth_matrix = pd.DataFrame(truth_map)
        precision = truth_matrix['TP'].sum() / (truth_matrix['TP'].sum() + truth_matrix['FP'].sum())
        recall = truth_matrix['TP'].sum() / (truth_matrix['TP'].sum() + (len(truth_matrix) - truth_matrix['TP'].sum() - truth_matrix['FP'].sum()))
        return truth_matrix, precision, recall
    
    @staticmethod
    def calculate_recall(merged, evaluation_column, label_columns):
        recall_values = []
        reference_clusters = merged[evaluation_column].dropna().unique()

        # Precompute and store the predicted hashes for each cluster
        cluster_hashes = {}
        for cluster in merged[label_columns].dropna().unique():
            predicted_hashes = set(merged[merged[label_columns] == cluster].hash.unique())
            cluster_hashes[cluster] = predicted_hashes

        # Calculate recall for each reference cluster
        for reference_cluster in reference_clusters:
            reference_hashes = set(merged[merged[evaluation_column] == reference_cluster].hash.unique())
            max_recall_intersection = 0

            # Retrieve precomputed predicted hashes for each cluster
            for cluster, predicted_hashes in cluster_hashes.items():
                num_intersect = len(reference_hashes.intersection(predicted_hashes))
                max_recall_intersection = max(max_recall_intersection, num_intersect)

            recall_values.append(max_recall_intersection)

        recall = np.sum(recall_values) / len(merged)  # Calculate recall using the number of samples in merged

        return recall

    @staticmethod
    def calculate_precision(merged, evaluation_column, label_columns):
        precision_values = []
        predicted_clusters = merged[label_columns].dropna().unique()

        # Precompute and store the predicted hashes for each cluster
        reference_hashes = dict()
        for reference_cluster in merged[evaluation_column].dropna().unique():
            groundtruth_hashes = set(merged[merged[evaluation_column] == reference_cluster].hash.unique())
            reference_hashes[reference_cluster] = groundtruth_hashes

        # Calculate recall for each reference cluster
        for cluster in predicted_clusters:
            predicted_hashes = set(merged[merged[label_columns] == cluster].hash.unique())
            max_precision_score = 0

            # Retrieve precomputed predicted hashes for each cluster
            for _, reference_hash in reference_hashes.items():
                num_intersect = len(reference_hash.intersection(predicted_hashes))
                max_precision_score = max(max_precision_score, num_intersect)

            precision_values.append(max_precision_score)

        precision = np.sum(precision_values) / len(merged)  # Calculate recall using the number of samples in merged

        return precision

class StringProcessing():
    def __init__(self):
        pass
    
    def is_garbage_string(self, string, exclusions=None):
        if not all(ord(c) < 128 and c.isprintable() for c in string):
            return True

        valid_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
        valid_chars |= set(regex.escape("你好世界Привет мир"))

        for char in string:
            if char not in valid_chars:
                return True

        patterns = [
            r'(;[a-zA-Z0-9])',
            r'([^\w\s])\1+',
            # r'(.)\1+',
            r'([<>\\|])',
        ]

        for pattern in patterns:
            if regex.search(pattern, string):
                return True

        if len(set(string)) == 1:
            return True

        if exclusions:
            for exclusion in exclusions:
                if regex.search(exclusion, string):
                    return True
        return False

    def process_strings(self, strings, exclusions=None):
        valid_strings = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(lambda string: self.is_garbage_string(string, exclusions), strings)
            for string, result in zip(strings, results):
                if not result:
                    valid_strings.append(string)
        return valid_strings
    
    
class Visualization():
    def __init__(self):
        pass
    
    def generate_precision_sse(self, all_params):
        sse_values = [params['sse'] for params in all_params]
        precision_values = [params['precision'] for params in all_params]

        # Calculate the coefficients of the line of best fit
        coefficients = np.polyfit(sse_values, precision_values, 1)
        line = np.poly1d(coefficients)

        # Generate x values for the line of best fit
        x_values = np.linspace(min(sse_values), max(sse_values), 100)

        # Calculate distances between data points and the line of best fit
        distances = []
        for i in range(len(sse_values)):
            distance = abs(line(sse_values[i]) - precision_values[i])
            distances.append(distance)

        # Plot the scatter plot
        plt.scatter(sse_values, precision_values)
        plt.xlabel('SSE')
        plt.ylabel('Precision')
        plt.title('SSE vs Precision')

        plt.plot(x_values, line(x_values), color='r')

        for i in range(len(sse_values)):
            plt.plot([sse_values[i], sse_values[i]], [precision_values[i], line(sse_values[i])], linestyle=':', color='gray')
            
    
            

        return plt
    

    @staticmethod
    def best_truth_curve(best_truth_matrix, evaluation_column, Graph_Label):

        y_true = best_truth_matrix[evaluation_column].values
        y_pred = best_truth_matrix['majority_vote'].values

        n_classes = np.max(y_true) + 1
        y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
        y_pred_bin = label_binarize(y_pred, classes=np.arange(n_classes))


        # Compute ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for class_label in range(n_classes):
            fpr[class_label], tpr[class_label], _ = roc_curve(y_true_bin[:, class_label], y_pred_bin[:, class_label])
            roc_auc[class_label] = auc_calc(fpr[class_label], tpr[class_label])

        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_bin.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Create traces for each class
        traces = []
        for class_label in range(n_classes):
            trace = go.Scatter(
                x=fpr[class_label],
                y=tpr[class_label],
                mode='lines',
                name='{} {}'.format(Graph_Label, class_label+1),
                line=dict(width=2)
            )
            traces.append(trace)

        # Add micro-average trace
        trace_micro = go.Scatter(
            x=fpr['micro'],
            y=tpr['micro'],
            mode='lines',
            name='Micro-average (AUC = {:.2f})'.format(roc_auc['micro']),
            line=dict(width=1)
 
        )
        traces.append(trace_micro)

        # Create layout
        layout = go.Layout(
            title='Receiver Operating Characteristic (ROC) Curve (Multiclass)',
            xaxis=dict(title='False Positive Rate'),
            yaxis=dict(title='True Positive Rate'),
            showlegend=True
        )

        # Create figure
        fig = go.Figure(data=traces, layout=layout)
        #fig.write_image('output_file.pdf', format='pdf')
        
        fig.update_layout(width=750, 
                          height=500,
                          font=dict(
                                size= 10))


        # Show the figure
        return fig.show()

class Modelling:

    @staticmethod
    def find_best_agglo(combined_features, n_clusters, all_features, evaluation_column, label_encode=True, metrics = ["euclidean"], linkages=["average"], n_distances = None):
        """
        Find the best Agglomerative Clustering parameters.

        Args:
            combined_features (numpy.ndarray): The combined feature matrix.
            n_clusters (list): List of numbers of clusters to try.
            n_distances (list): List of distance thresholds. If this is present n_clusters has to be null
            all_features (pd.DataFrame): It should only have two columns, the hash and the evaluation columns. 
            evaluation_column (str): E.g Adversary_Tag or Campaign_Tag.
            label_encode (bool): Label encode the evaluation column.
            metrics: Agglomerative metrics for eval. E.g: etrics = ["euclidean", "manhattan"]
            linkage: The choice of linkage algorithm. E.g linkages=["ward", "complete", "average"]

        Returns:
            tuple: A tuple containing a list of all parameters tried,  the best parameter set, and the best truth matrix.

        """
        clustering_metrics = ClusteringMetrics()

        # Select the hash and evaluation column.
        all_features = all_features[["hash", evaluation_column]]
        best_precision = -1
        best_auc = -1
        best_sil = -1
        best_sk_f1_score = -1
        best_truth_matrix = None
        best_n = 0

        best_param = {}
        all_params = []
        if n_clusters and n_distances:
            raise ValueError("You can't have both clusters and distances, we can only use one")
        if n_clusters:
            search_values = n_clusters
        else:
            search_values = n_distances
        for metric in tqdm(metrics):
            for search_value in tqdm(search_values, leave=False):
                for linkage in tqdm(linkages, leave=False):
                    if n_clusters:
                        agglomerative = AgglomerativeClustering(n_clusters=search_value, distance_threshold=None, metric=metric, linkage=linkage)
                    if n_distances:
                        agglomerative = AgglomerativeClustering(n_clusters=None, distance_threshold=search_value, metric=metric, linkage=linkage)

                    y_pred = agglomerative.fit_predict(combined_features)
                    ground_truth = all_features.copy()
                    ground_truth['labels'] = y_pred.copy()
                    truth_matrix, score, recall = clustering_metrics.metrics_using_labels(ground_truth, evaluation_column, label_encode=True)
                    sk_precision = classification_report(truth_matrix[evaluation_column], truth_matrix.majority_vote, output_dict=True, zero_division=True).get("weighted avg", {}).get("precision", 0)
                    sk_recall = classification_report(truth_matrix[evaluation_column], truth_matrix.majority_vote, output_dict=True, zero_division=True).get("weighted avg", {}).get("recall", 0)
                    sk_f1_score = classification_report(truth_matrix[evaluation_column], truth_matrix.majority_vote, output_dict=True, zero_division=True).get("weighted avg", {}).get("f1-score", 0)

                    sk_auc = clustering_metrics.get_auc(truth_matrix, evaluation_column)
                    sil_score = silhouette_score(combined_features, y_pred)
                    n_distinct_centroids = len(set(y_pred))
                    cluster_centers = np.zeros((n_distinct_centroids, combined_features.shape[1]))
                    for cluster_label in np.unique(y_pred):
                        cluster_centers[cluster_label] = np.mean(combined_features[y_pred == cluster_label], axis=0)

                    distances = np.sum((combined_features - cluster_centers[y_pred])**2, axis=1)
                    sse_score = np.sum(distances)

                    n_major_cluster = truth_matrix.majority_vote.nunique()
                    params = {
                        "metric": metric,
                        "search_value": search_value,
                        "n_major_cluster": n_major_cluster,
                        "linkage": linkage,
                        "precision": sk_precision,
                        "recall": sk_recall,
                        "f1_score": sk_f1_score,
                        "auc": sk_auc,
                        "silhouette": sil_score,
                        "sse": sse_score,
                        "optimizing_on": "f1"
                       
                    }
                    all_params.append(params)

                    if sil_score > best_sil:
                        best_sil = sil_score

                    if sk_precision > best_precision:
                        best_precision = sk_precision

                    if sk_f1_score > best_sk_f1_score:
                        best_sk_f1_score = sk_f1_score
                        best_truth_matrix = truth_matrix
                        best_n = len(set(y_pred))
                        best_param = params

        return all_params, best_param, best_truth_matrix


class DataProcessor():

    def __init__(self):
        self.known_words = set(words.words())

    def is_valid_unix_path(self, file_path):
        try:
            if len(file_path) > 128:  # Adjust the limit as needed
                return False

            # Split the path using both forward slash and backslash
            path_components = [component for component in re.split(r'[\\\/]', file_path) if component]

            # Check if the path components are valid
            for component in path_components:
                if component.startswith('.') or '/' in component or '\\' in component:
                    return False

            # Check if consecutive slashes are present (not allowed)
            if '//' in file_path or '\\\\' in file_path:
                return False

            # Check if at least one segment contains a valid known word according to the dictionary
            for component in path_components:
                words_in_component = re.findall(r'\b\w+\b', component)
                if any(word.lower() in self.known_words and len(word) > 3 for word in words_in_component):
                    return True

            # If none of the above conditions are met, return False
            return False

        except Exception:
            # Any exceptions indicate an invalid path
            return False

    def validate_ip_addresses(self, ip_addresses):
        valid_ips = []

        for ip in ip_addresses:
            try:
                # Attempt to create an IPv4Address object
                IPv4Address(ip)
                valid_ips.append(ip)
            except ValueError:
                pass
            try:
                # Attempt to create an IPv4Address object
                IPv6Address(ip)
                valid_ips.append(ip)
            except ValueError:
                pass
        return valid_ips

    def mean_pooling(self, model_output: BatchEncoding, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Perform mean pooling on token embeddings based on attention masks.

        This function calculates the mean of token embeddings, weighted by the attention mask,
        for each input sequence in the batch.

        Parameters:
            model_output: The output of a transformer-based model.
            attention_mask: A binary mask indicating which tokens in the input are valid (1) and which are padding (0).

        Returns:
            Mean-pooled embeddings for each input sequence.
        """
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode_list_of_texts(self, texts: list[str], tokenizer: PreTrainedTokenizer, model: PreTrainedModel) -> torch.Tensor:
        """
        Encode a list of text sequences into embeddings using a transformer-based model.

        This function tokenizes the input texts, computes token embeddings, performs mean pooling,
        and normalizes the resulting embeddings.

        Parameters:
            texts: A list of text sequences to be encoded.
            tokenizer: The tokenizer associated with the transformer-based model.
            model: The transformer-based model for encoding text.

        Returns:
            Normalized embeddings for the input text sequences.
        """
        # Tokenize sentences
        encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input, return_dict=True)

        # Perform pooling
        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        embeddings = torch_ff.normalize(embeddings, p=2, dim=1)

        return embeddings

    def unique_elem_dict(self, data: pd.DataFrame, column: str) -> dict[str, Any]:
        """
        Create a dictionary with unique elements in a specified column of a DataFrame
        and their corresponding indices in the DataFrame.

        Parameters:
        data (pd.DataFrame): The DataFrame to process.
        column (str): The name of the column in the DataFrame to extract unique elements from.

        Returns:
        dict: A dictionary where keys are unique elements from the specified column,
        and values are lists of indices in the DataFrame where those elements appear.
        """

        unique_email_dict = {}

        for index, email_list in tqdm(enumerate(data[column].dropna())):
            if email_list:  # Check if the list is not empty
                for email in email_list:
                    if email not in unique_email_dict:
                        unique_email_dict[email] = [index]
                    else:
                        unique_email_dict[email].append(index)
        return unique_email_dict


    def encode_list_of_texts_batched(self, texts: list[str], tokenizer: PreTrainedTokenizer, model: PreTrainedModel, batch_size: int = 32) -> torch.Tensor:
        """Encode a list of text sequences into embeddings using a transformer-based model with batching.

        This function tokenizes the input texts, computes token embeddings, performs mean pooling,
        and normalizes the resulting embeddings in batches to prevent MemoryError.

        Parameters:
            texts (List[str]): A list of text sequences to be encoded.
            tokenizer (PreTrainedTokenizer): The tokenizer associated with the transformer-based model.
            model (PreTrainedModel): The transformer-based model for encoding text.
            batch_size (int): The batch size for processing text sequences.

        Returns:
            Normalized embeddings for the input text sequences.
        """
        # Initialize a list to store the embeddings
        embeddings_list = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Process texts in batches
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i + batch_size]

            # Tokenize sentences for the current batch
            encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt').to(device)

            with torch.no_grad():
                model_output = model(**encoded_input, return_dict=True)
            # Perform pooling for the batch
            batch_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])

            # Normalize batch embeddings
            batch_embeddings = torch_ff.normalize(batch_embeddings, p=2, dim=1)

            # Append batch embeddings to the list
            embeddings_list.append(batch_embeddings)
            del batch_embeddings, encoded_input
            torch.cuda.empty_cache()

        # Concatenate batch embeddings to get the final result
        if len(embeddings_list) > 0:
            embeddings = torch.cat(embeddings_list)
        else:
            embeddings = torch.empty(0)

        return embeddings

    def compute_similar_candidates(self, unique_values_sets: list, doc_emb: torch.Tensor, sim_threshold: float=0.9) -> dict:
        """Calculate similar candidates for each element in a list based on their embeddings.

        Parameters:
            unique_values_sets (list): A list of elements for which similar candidates need to be found.
            doc_emb (torch.Tensor): A tensor containing embeddings for all elements.
            sim_threshold: The threshold for considering things to be similar.
        Returns:
            A dictionary where keys are elements from 'unique_values_sets', and values are lists
        of similar candidates based on the similarity score (greater than 0.85).
        """

        similar_candidates = {}

        # Loop through each element in unique_values_sets
        for idx, value in tqdm(enumerate(unique_values_sets)):
            # Get the embedding for the current element
            query_emb = doc_emb[idx:idx+1]  # Ensure query_emb is a 2D tensor
            

            # Calculate the similarity scores with all elements in doc_emb
            # Perform matrix multiplication and squeeze any singleton dimensions
            scores = torch.mm(query_emb, doc_emb.transpose(0, 1)).squeeze()

            # Check if scores is a scalar (not iterable) by checking its dimension
            if scores.dim() == 0:
                # If scores is a scalar, convert it to a list with one element
                scores_list = [scores.item()]
            else:
                # If scores is not a scalar, convert the tensor to a list
                scores_list = scores.cpu().tolist()


            # Find similar candidates with a score greater than 0.85
            similar_indices = [i for i, score in enumerate(scores_list) if score > sim_threshold]

            # Exclude the element itself from similar candidates
            similar_indices = [i for i in similar_indices if i != idx]

            # Store the similar candidates in the dictionary
            similar_candidates[value] = [unique_values_sets[i] for i in similar_indices]

        return similar_candidates


    def calculate_cardinality(self, cell_item, unique_object_dict):
        # Function to calculate the cardinality of an email address
        if cell_item in unique_object_dict:
            return len(unique_object_dict[cell_item])
        return 0

    def find_similar_candidates(self, cell_item, similar_candidates):
        if cell_item in similar_candidates:
            return similar_candidates[cell_item]
        return []

    def normalize_fields_sim(self, row):
        return [
            element for element in row if self.calculate_cardinality(element) > 1
        ] + [
            candidate for element in row for candidate in self.find_similar_candidates(element)
        ]

    def normalize_cell_elements(self, row: list[str], use_similarity: bool = False, 
                                similar_candidates: dict[str, Any] = None, 
                                unique_element_dict: dict[str, Any] = None,
                                cardinality_lower_bound = 1,
                                cardinality_ratio = 0.75
                               ) -> list[str]:
        """
        Normalizes cell elements based on the specified criteria.

        Parameters:
            row: The list of strings representing the row to be normalized.
            use_similarity: A flag indicating whether to use similarity criteria for normalization. Default is False.
            similar_candidates: A dictionary containing potential similar candidates for elements in the row.
            unique_element_dict: A dictionary mapping each element to its uniqueness score.
            cardinality_lower_bound: The minimum number of elements within the dataframe that should be similar.
                - We default to 1. 
            cardinality_ratio: If an element/value has a ratio of occurence greater than this number, we drop it.             

        Returns:
            The list of normalized cell elements.
        """

        if unique_element_dict is None:
            raise ValueError("unique_element_dict cannot be None")

        if not row:
            return []  # Return an empty list if row is empty

        # Precompute cardinality values for all elements in the row
        cardinalities = [self.calculate_cardinality(element, unique_element_dict) for element in row]
        max_cardinality = max(cardinalities)
        if use_similarity:
            if similar_candidates is None:
                raise ValueError("similar_candidates cannot be None when use_similarity is True")

            return [
                element for element, cardinality in zip(row, cardinalities) if 
                cardinality > cardinality_lower_bound and cardinality < cardinality_ratio * max_cardinality
            ] + [
                candidate 
                for element in row 
                for candidate in self.find_similar_candidates(element, similar_candidates)
            ]

        return [
            element for element, cardinality in zip(row, cardinalities) if 
            cardinality > cardinality_lower_bound and cardinality < cardinality_ratio * max_cardinality
        ]


    def string_feature_embed_similarity(self, data: pd.DataFrame, column: str, tokenizer: PreTrainedTokenizer, model: PreTrainedModel, similarity_threshold=0.70, **kwargs) -> pd.Series:
        """This function takes a DataFrame, a column name containing elements, and performs normalization
        on the elements in the cell using embeddings and similarity scores.

        Parameters:
            data : The DataFrame containing the data.
            column : The name of the column in the DataFrame that contains the elements.
            tokenizer : The tokenizer associated with the transformer-based model.
            model : The transformer-based model for encoding text.

        Returns:
            A Pandas Series containing the normalized elements based on the provided logic.
        """

        # Create a set of unique elements from the specified column
        unique_values_sets = [i.rstrip().lstrip() for i in list(set().union(*data[column].dropna())) if i]
        doc_emb = self.encode_list_of_texts_batched(unique_values_sets, tokenizer, model, 1000)

        # Create a dictionary of unique elements and their indices in the DataFrame
        unique_element_dict = self.unique_elem_dict(data, column)

        # Calculate similar candidates for elements based on a similarity threshold
        similar_candidates = self.compute_similar_candidates(unique_values_sets, doc_emb, similarity_threshold)

        # Apply the normalization function to the specified column
        normalized_data_elements = data[column].apply(self.normalize_cell_elements,
                                                            use_similarity=True,
                                                           similar_candidates=similar_candidates,
                                                           unique_element_dict=unique_element_dict,
                                                     **kwargs)

        return normalized_data_elements


    def normalize_column_using_popularity(self, data: pd.DataFrame, column: str, **kwargs):
        """Normalize the elements in a specified column of a DataFrame using a custom normalization function.

        Parameters:
            data (pd.DataFrame): The DataFrame containing the data.
            column (str): The name of the column in the DataFrame that contains the elements to be normalized.

        Returns:
            A Pandas Series containing the normalized elements based on the provided normalization function.
        """
        if column not in data:
            raise ValueError(f"Column '{column}' not found in the DataFrame.")

        unique_elements_obj = self.unique_elem_dict(data, column)
        return data[column].apply(self.normalize_cell_elements, use_similarity=False,unique_element_dict=unique_elements_obj, **kwargs)


    def one_hot_encode_list_column(self, large_dataframe: pd.DataFrame, column_name: str, to_lower=False) -> pd.DataFrame:
        """One-hot encode a column in a DataFrame containing lists of strings.

        Parameters:
            large_dataframe (pd.DataFrame):
                The DataFrame containing the data to be one-hot encoded.
            column_name (str):
                The name of the column in the DataFrame that contains lists of strings.

        Returns:
            A new DataFrame with the specified column one-hot encoded and concatenated
            with the original DataFrame.
        """
        # Create an instance of MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        if to_lower:
            large_dataframe[column_name] = large_dataframe[column_name].apply(lambda x: list({i.lower() for i in x}))
        # Fit and transform the specified column
        one_hot_encoded = pd.DataFrame(mlb.fit_transform(large_dataframe[column_name]), columns=mlb.classes_)

        # Concatenate the one-hot encoded DataFrame with the original DataFrame
        result = pd.concat([large_dataframe, one_hot_encoded], axis=1).drop(columns=[column_name])

        return result

    def merge_and_relabel(self, df1, df2, merge_column, reference_label_col, other_label_col):
        """
        Merges two DataFrames (df1 and df2) on a specified column (merge_column) and generates a new
        DataFrame with relabeled data using either df1 or df2 labels as a reference.

        Parameters:
        - df1, df2 (pd.DataFrame): DataFrames to merge and relabel.
        - merge_column (str): The column name on which to merge the DataFrames.
        - reference_label_col, other_label_col (str): The column names containing the labels to use
          as reference and the other to compare.

        Returns:
        - new_label_assignments_df (pd.DataFrame): DataFrame containing the merged and relabeled data.
        """

        # Merge the DataFrames
        merged_labels = pd.merge(df1, df2, on=merge_column, suffixes=('_ref', '_other'))

        # Initialize seen hashes and new label assignments
        seen_hashes = set()
        new_label_assignments = []

        # Iterate over unique labels in the reference label column
        for label_ref in merged_labels[reference_label_col + '_ref'].unique():

            # If all hashes for this label have been seen, continue
            if set(merged_labels.loc[merged_labels[reference_label_col + '_ref'] == label_ref, merge_column]).issubset(seen_hashes):
                continue

            # Get the relevant subset of hashes for the current label
            ref_hashes = set(merged_labels.loc[merged_labels[reference_label_col + '_ref'] == label_ref, merge_column]) - seen_hashes

            # Check the corresponding other labels for each hash in this subset
            other_labels = merged_labels.loc[merged_labels[merge_column].isin(ref_hashes), other_label_col + '_other'].unique()

            # For each other label, get all hashes and add them with the reference label to new assignments
            for label_other in other_labels:
                new_hashes = set(merged_labels.loc[merged_labels[other_label_col + '_other'] == label_other, merge_column])

                for hash_ in new_hashes:
                    new_label_assignments.append({merge_column: hash_, 'labels': label_ref})

                # Update seen hashes
                seen_hashes.update(new_hashes)

        # Convert the list of dictionaries to a DataFrame
        new_label_assignments_df = pd.DataFrame(new_label_assignments)

        return new_label_assignments_df