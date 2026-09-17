# Intent Qualification

## Constantinescu Theodor-Cristian

### Approach

  The core pipeline follows a hybrid retrieval architecture that
decouples quantitative data-based filtering for qualitative vector
semantic search.

  It includes a First Normal Form processor, a dense vector encoder,
and a query deconstructor which parses unfiltered natural language
user queries into numerical/boolean metadata predicates (the dataset
  columns) and a semantic query.

The search engine applies the metadata filters to the normalized database
then executes cosine similarity between query vector and database candidates.

I chose this design because it keeps the semantic understanding of an LLM with
the speed of embeddings. For efficiency, I thought from the beginning of a
filtering method in order to reduce the dataset search.

### Tradeoffs

The system was designed for high accuracy (filtering eliminated out
of bounds candidates) and simplicity(numpy and pandas dataframe operations).
The speed is moderate: despite the slow nature of LLM, vector similarity
runs over pre-filtered subsets instead of the whole set. Instead of sending
each company individually to the LLM, the LLM usage is once per user query.

### Error Analysis

1. Missing metadata:

One error is over-filtering due to missing metadata. A hard metadata
filter could drop a suitable candidate because the database is incomplete
and certain columns can be marked as null. For example, a query with
"year_founded >= 1990" automatically drops companies without a year of
foundation even if they founding year meets the criteria.

2. Short single word queries:

The LLM might extend single-world queries into multiple semantic descriptions,
moving away from the desired result. For example, querying "Rompetrol"
might return "Romanian oil and gas", changing the whole meaning.

### Scaling architecture

For more companies per query I believe integrating the SQLite
engine is very fit for managing a larger database.

### Failure methods

1. System failure might occur when the pipeline gives a higher cosine
similarity despite delivering not the most accurate result. For
example, ambiguous regions such as "scandinavia" or "midwest" might
miss important regional factors (finland might not be included in
Scandinavia).


2. A Good metric for evaluating the dataset is measuring the value
of the cosine similarity.

3. Is also important to measure the frequency of bypassing the hard
filters due to zero matches. A spike indicates overly aggressive hard
filtering by the query parser.

### Future improvements

For a better improvement of this search engine I believe a good
ideea is to apply a weighted search. It's role is to combine
high semantic meaning with precise world matching (using BM25).
An intuitive formula would be:

Final_score = a * (BM25) + (1 - a) * (cosine_similarity)