# ADAPT – Troubleshooting & Developer Guide

As researchers, we aim to make our code fully reproducible. However, due to the evolving nature of third-party APIs, tools, and malware samples, 
issues may arise. This guide outlines key modules, common pitfalls, and optimization tips for working and extending ADAPT.

---

## Key Modules & Dependencies

- **Third-party tools & APIs:**  
  - [Censys API](https://search.censys.io/api)
  - [lief](https://lief.quarkslab.com/)  
  - [oletools](https://github.com/decalage2/oletools)  
  - [malcat yara](https://malcat.fr/)  
  - FLOSS and Exiftool (included in `bin/` directory with fixed versions)

---

## Input Structure


/downloaded_samples_folder/  
├── 0123abcd…/  
   ├── 0123abcd…           # Sample file (PDF, EXE, DOC, etc.)  
   ├── 0123abcd….json      # VT metadata file (required for Censys queries)


 **Note:**  
The presence of the `{file_hash}.json` VT metadata file is **crucial**.  
It provides the **first submission date**, which is used to narrow the time window in Censys certificate and host queries.

## Running the Feature Extraction

Before running the pipeline, ensure that the file paths are correctly configured for your local environment.

```python
async def main():
    BASE_DIR = r"provide\\the\\folderpath\\malware\\samples"
...
```
Update BASE_DIR to point to your folder containing malware samples.

## Regex Matching Strategies

ADAPT implements two different strategies for regex-based feature extraction:

---

### 1. `feature_processing.py` – Individual Pattern Matching

- **Approach:** Matches each regex individually against string content.
- **Pros:** High flexibility; useful for detailed analysis and debugging.
- **Cons:** Very **slow** on large datasets (~10,000+ samples).
- **Use case:** Debugging or working with small datasets.
- **Reference:** See code block starting around line **1366** in `feature_processing.py`.

---

### 2. `optimized_feature_processing_v1.py` – Combined Pattern Matching

- **Approach:** Combines all regexes into a **single large pattern** using named groups.
- **Pros:** Much **faster**; optimized for batch processing.
- **Cons:** Less fine-grained control; regex patterns must be carefully structured.
- **Optimization:** Strings longer than **2000 characters** are skipped to avoid regex timeouts and high memory usage.

#### Example Code:
```python
candidate_strings = [
    s.get("string").strip()
    for s in all_strings.get("static_strings", [])
    if s.get("string") and len(s.get("string")) < 2000
]
```
You can adjust or remove the string length constraint in the snippet above depending on your dataset and system capabilities.


## Handling Censys API Responses

ADAPT queries the Censys API to extract certificate and host metadata.  
To enable this, you need to set your API credentials as environment variables.  
In the code, the credentials are accessed like this:

```python
censys_api_id = os.getenv("CENSYS_API_ID")
censys_api_secret = os.getenv("CENSYS_API_SECRET")
```

These must be set in your system or runtime environment and avoid hardcoding them into the script for security reasons.
These credentials are required to authenticate your requests with the Censys API and fetch metadata reliably.

The Censys responses can change over time depending on how Censys structures its API.
Refer to the following code block in case the response structure from Censys is not producing expected results.

```python
def censys_certificate_data(domain_name: str, sample_left_date: datetime, sample_right_date: str = "*") -> list:
    sample_left_date_str = sample_left_date.strftime("%Y-%m-%d")

    certificate_query = (
        f"parsed.extensions.subject_alt_name.dns_names:{domain_name}"
        f"AND added_at:[{sample_left_date_str} TO {sample_right_date}]"
    )
```


## Expected Output Folder Structure

After feature extraction, your folder structure should look like this. 


/downloaded_samples_folder/  
├── 0123abcd…/  
   ├── 0123abcd…                    # Sample file (PDF, EXE, DOC, etc.)  
   ├── 0123abcd….json               # VT metadata file (required for Censys cert queries)  
   ├── censys_features_withhostdata.json  
   ├── exiftool_results.json  
   ├── flossresults_reduced_7.json  
   ├── lief_features.json           # Present only for PE (executable) files  
   ├── malcatYararesults.json  
   ├── oletool_features_updated.json  
   └── regex_results.json


## Group Attribution & Embedding-Based Feature

The `groupAttribution.ipynb` notebook performs clustering of malware samples based on extracted group-level features. 
It merges features from several sources: 
- exiftool metadata  
- malcat rule matches  
- regex-matched patterns  
- censys data  

These features often include string-based metadata (e.g., authors, company names, email addresses), which can be semantically similar even if lexically different. To normalize these and group similar entries, ADAPT computes text embeddings using a transformer-based language model (Model Name: `sentence-transformers/multi-qa-MiniLM-L6-cos-v1`).

---

### Embedding Computation Pipeline

The core logic for embedding-based feature normalization is implemented in the following files:

- **`group_features.py`**
```python
def compute_embeddings(self, data):
    # This function loops over selected columns and applies embedding-based normalization using string_feature_embed_similarity.
```
- **`util.py`**

Similarity Computation: Computes cosine similarity between embeddings. 
For each value, find similar entries above a given threshold.

```python
def compute_similar_candidates(self, unique_values_sets, doc_emb, sim_threshold=0.9) -> dict:
    ...
Returns a mapping: {original_value: [similar_candidates...]}.
```
Below is where the following computation happens. 
```python
scores = torch.mm(query_emb, doc_emb.transpose(0, 1)).squeeze()
scores_list = scores.cpu().tolist()
```
torch.mm creates a full similarity vector for each input string, and if you have 10,000 unique values, you’re creating and holding a 10,000 x 10,000 similarity matrix. That's 100M floats (~400MB just for the scores).

And finally, the below function performs normalization that includes extracting unique strings from the column, embedding them using a transformer, and finding similar values using cosine similarity.

```python
 def string_feature_embed_similarity(self, data, column, tokenizer, model, similarity_threshold=0.70) -> pd.Series:
```

### Memory Considerations

Computing embeddings for a large number of unique strings can consume significant memory, especially on machines without GPUs or with limited VRAM.

Additionally, certain scenarios can greatly increase processing time:

Large Malware Samples: Samples larger than 20 MB and with extensive string content (evidenced by large FLOSS output files) can significantly slow down embedding generation,

Agglomerative Clustering: This step can be computationally expensive, particularly if:

You're clustering a large number of samples, or

You're trying more than ~50 clusters — this can drastically increase runtime due to pairwise distance computations and hierarchical merging.

#### Possible Error:

```plaintext
RuntimeError: CUDA out of memory
```

### Solutions:

- Use `encode_list_of_texts_batched()` instead of the full-text version.
- Reduce batch size (e.g., `batch_size=16`).
- Filter very long strings (e.g., skip strings longer than 2000 characters).
- Consider switching to a smaller transformer model (e.g., `distilbert` instead of `bert-large`).
- Run on CPU (slower but safer): comment out `.to(device)` or set `device = torch.device("cpu")`.
- Pre-filter large samples (e.g., skip samples > 20 MB unless necessary).

Feel free to open issues or pull requests if you encounter any bugs or improvements!



