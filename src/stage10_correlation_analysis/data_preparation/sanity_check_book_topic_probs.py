import pandas as pd
import numpy as np

df = pd.read_parquet("results/correlation_analysis/data_preparation/topic_probabilities/book_topic_probs.parquet")

print("shape:", df.shape)
print("cols:", df.columns.tolist())
print("dtypes:", df.dtypes)

# uniqueness
dup = df.duplicated(["book_id","topic_id"]).sum()
print("duplicate (book_id, topic_id) rows:", dup)

n_books = df["book_id"].nunique()
n_topics = df["topic_id"].nunique()
print("unique books:", n_books)
print("unique topics:", n_topics)

# probability sanity
print("prob min/max:", df["prob"].min(), df["prob"].max())
print("prob < 0:", (df["prob"] < 0).sum())
print("prob NaN:", df["prob"].isna().sum())

# per-book sums
sums = df.groupby("book_id")["prob"].sum()
print("book sum prob: min/median/max:", sums.min(), sums.median(), sums.max())
print("books outside [0.99,1.01]:", (~sums.between(0.99,1.01)).sum())

# sparsity-ish
mass = df.groupby("book_id")["prob"].apply(lambda s: (s > 0.001).mean())
print("fraction topics > 0.001: min/median/max:", mass.min(), mass.median(), mass.max())

