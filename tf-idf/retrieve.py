import numpy as np
import os
import pandas as pd
import re
from collections import Counter


def get_cnt_string(str):
    """Count up word occurrences for a single string."""
    counter = Counter()
    # remove certain punctuation
    regex = re.compile(r"[,\.!?]")
    words = regex.sub("", str.lower()).split()
    return Counter(words)


def get_per_doc_cnt(documents):
    """Count up word occurrences for each document in the input directory."""
    doc_counters = {}

    for document in documents:
        with open(document, "r", encoding="utf-8") as f:
            text = f.read()
            counter = get_cnt_string(text)

        # Clean up document name
        doc_name = document.replace("input/", "").replace(".txt", "")
        doc_counters[doc_name] = counter

    return doc_counters


def calculate_tf_idf(counters, is_query=False, doc_counters=None):
    """Calculate tf and tf-idf for each term in each document or query.

    Here, tf is calculated as 1 + log10(count) and idf is calculated as log10(N / df),
    where N is the total number of documents and df is the number of documents containing the term.
    """
    data = []

    # Handle single query counter case
    if is_query:
        # For query, wrap the counter in a dict with 'query' as the key
        items_to_process = {"query": counters}
        corpus_counters = doc_counters
        N = len(doc_counters)
    else:
        # For documents, use the counters directly
        items_to_process = counters
        corpus_counters = counters
        N = len(counters)

    for doc_name, counter in items_to_process.items():
        for word, count in counter.items():
            # Calculate tf using log scaling
            tf = 1 + np.log10(count) if count > 0 else 0

            # Calculate idf based on the document corpus
            df = sum(1 for c in corpus_counters.values() if word in c)
            # Avoid division by zero if the word doesn't appear in any document
            idf = np.log10(N / df) if df > 0 else 0
            tf_idf = tf * idf

            data.append(
                {
                    "document": doc_name,
                    "word": word,
                    "count": count,
                    "tf": round(tf, 3),
                    "tf_idf": round(tf_idf, 3),
                }
            )

    return pd.DataFrame(data)


def vector_length(df):
    """Compute vector length for each document (or query) as the square root of the sum of squares of the tf-idf values."""
    for doc_name, group in df.groupby("document"):
        sum_of_squares = np.sum(group["tf_idf"] ** 2)
        vec_len = np.sqrt(sum_of_squares)
        df.loc[df["document"] == doc_name, "vector_length"] = round(vec_len, 3)
    return df


def get_cosine_similarity(tfidf_docs, tfidf_query, vector_lengths):
    """Calculate cosine similarity between document vector and query vector.

    Cosine similarity is defined as the dot product of the two vectors divided by the product of their lengths.
    """
    cosine_similarities = {}
    for doc in tfidf_df.index:
        if doc == "query":
            print("continued")
            continue
        # Calculate the dot product between the document and query
        dot_product = np.dot(tfidf_df.loc[doc].values, query_vector.values)

        doc_length = vector_lengths.loc[
            vector_lengths["document"] == doc, "vector_length"
        ].values[0]
        query_length = vector_lengths.loc[
            vector_lengths["document"] == "query", "vector_length"
        ].values[0]

        # Calculate cosine similarity
        similarity = dot_product / (doc_length * query_length)
        cosine_similarities[doc] = similarity
    return cosine_similarities


if __name__ == "__main__":
    # Collect document paths
    documents = []
    for item in os.listdir("input"):
        item_path = os.path.join("input", item)
        if os.path.isfile(item_path):
            documents.append(item_path)

    query = "sweet love"

    # Process document files and query separately
    doc_counters = get_per_doc_cnt(documents)
    query_counter = get_cnt_string(query)

    # Calculate tf and tf-idf for documents
    doc_metrics = calculate_tf_idf(doc_counters, is_query=False)

    # Calculate tf and tf-idf for the query using idf from the documents
    query_metrics = calculate_tf_idf(
        query_counter, is_query=True, doc_counters=doc_counters
    )

    results_df = pd.concat([doc_metrics, query_metrics], ignore_index=True)

    pivot_df = results_df.pivot(
        index="document", columns="word", values=["count", "tf", "tf_idf"]
    ).fillna(0)

    print("\nTerm frequencies:")
    print(pivot_df["tf"])

    print("\nTF-IDF values:")
    print(pivot_df["tf_idf"])

    # Compute vector length for each document (and the query)
    results_df = vector_length(results_df)
    vector_lengths_df = results_df[["document", "vector_length"]].drop_duplicates()

    print("\nVector lengths:")
    print(vector_lengths_df)

    # Get each set of tf-idf values
    tfidf_df = pivot_df["tf_idf"]
    query_vector = tfidf_df.loc["query"]

    # Initialize a dictionary to store cosine similarities
    cosine_similarities = get_cosine_similarity(
        tfidf_df, query_vector, vector_lengths_df
    )

    print("\nCosine similarities:")
    for doc, similarity in cosine_similarities.items():
        print(f"{doc}: {similarity:.3f}")
