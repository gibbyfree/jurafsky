import numpy as np
import os
import pandas as pd
import re
from collections import Counter


def get_cnt(documents):
    """Count up word occurrences for each document in the input directory."""
    doc_counters = {}
    # remove certain punctuation
    regex = re.compile(r"[,\.!?]")

    for document in documents:
        counter = Counter()
        with open(document, "r", encoding="utf-8") as f:
            words = (
                regex.sub("", word.lower())
                for line in f
                for word in line.strip().split()
            )
            counter.update(words)
        # Clean up document name
        doc_name = document.replace("input/", "").replace(".txt", "")
        doc_counters[doc_name] = counter

    return doc_counters


def calculate_metrics(doc_counters):
    """Calculate all metrics for documents in one pass."""
    data = []

    for doc_name, counter in doc_counters.items():
        for word, count in counter.items():
            # Calculate tf
            tf = 1 + np.log10(count) if count > 0 else 0

            # Calculate tf-idf
            idf = np.log10(
                len(doc_counters)
                / (0 + sum(1 for c in doc_counters.values() if word in c))
            )
            tf_idf = tf * idf

            data.append(
                {
                    "document": doc_name,
                    "word": word,
                    "count": count,
                    "tf": tf,
                    "tf_idf": tf_idf,
                }
            )

    return pd.DataFrame(data)


if __name__ == "__main__":
    # Collect document paths
    documents = []
    for item in os.listdir("input"):
        item_path = os.path.join("input", item)
        if os.path.isfile(item_path):
            documents.append(item_path)

    # Get word counts for each document
    doc_counters = get_cnt(documents)

    # Calculate tf, tf-idf
    results_df = calculate_metrics(doc_counters)

    # Pivot for a cleaner matrix view
    pivot_df = results_df.pivot(
        index="document", columns="word", values=["count", "tf", "tf_idf"]
    ).fillna(0)

    print("\nTerm frequencies:")
    print(pivot_df["tf"])

    print("\nTF-IDF values:")
    print(pivot_df["tf_idf"])
