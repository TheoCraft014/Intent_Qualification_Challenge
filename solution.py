import pandas as pd
import config
from data_processing import process_entire_dataset
from embeddings import EmbeddingEngine
from search_engine import SearchEngine

def main():
    # Init and process dataset
    df_1nf = process_entire_dataset(config.DATASET_PATH)

    # Generate embeddings
    embedding_engine = EmbeddingEngine()
    company_embeddings = embedding_engine.build_embeddings(df_1nf)

    # Init the search engine
    search_engine = SearchEngine(
        df=df_1nf,
        embeddings=company_embeddings,
        embedding_engine=embedding_engine
    )

    test_queries = [
        "Logistic companies in Romania",
        "Public software companies with more than 1,000 employees.",
        "Food and beverage manufacturers in France",
        "Companies that could supply packaging materials for a direct-to-consumer cosmetics brand",
        "Construction companies in the United States with revenue over $50 million",
        "Pharmaceutical companies in Switzerland",
        "B2B SaaS companies providing HR solutions in Europe",
        "Clean energy startups founded after 2018 with fewer than 200 employees",
        "Fast-growing fintech companies competing with traditional banks in Europe.",
        "E-commerce companies using Shopify or similar platforms",
        "Renewable energy equipment manufacturers in Scandinavia",
        "Companies that manufacture or supply critical components for electric vehicle battery production",
        "Rompetrol"
    ]

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    for i, query in enumerate(test_queries, 1):
        print(f"\n==================== QUERY {i}: '{query}' ====================")
        results = search_engine.execute_search(user_query=query, top_k=5)

        if results.empty:
            print("-> No companies match")
        else:
            print(results[["operational_name", "country_code", "employee_count", "similarity_score"]])

if __name__ == "__main__":
    main()



