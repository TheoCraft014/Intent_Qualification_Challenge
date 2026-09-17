import pandas as pd
from openai import AzureOpenAI
from sentence_transformers import util

import config
from schemas import QuerySchema, SYSTEM_INSTRUCTION
from embeddings import EmbeddingEngine

class SearchEngine:
    def __init__(self, df: pd.DataFrame, embeddings, embedding_engine: EmbeddingEngine):
        self.df = df
        self.embeddings = embeddings
        self.embedding_engine = embedding_engine
        self.llm_client = AzureOpenAI(
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_key=config.AZURE_OPENAI_API_KEY,
            api_version=config.AZURE_OPENAI_API_VERSION,
        )

    def parse_query_with_llm(self, user_query: str) -> QuerySchema:
        response = self.llm_client.beta.chat.completions.parse(
            model=config.LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": user_query},
            ],
            response_format=QuerySchema,
        )
        return response.choices[0].message.parsed

    def execute_search(self, user_query: str, top_k: int = 5) -> pd.DataFrame:
        parsed = self.parse_query_with_llm(user_query)
        filtered = self.df.copy()

        # Execute hard metadata filtering
        if parsed.country_codes:
            codes_lower = [c.lower() for c in parsed.country_codes]
            filtered = filtered[filtered["country_code"].isin(codes_lower)]

        if parsed.min_employees is not None:
            filtered = filtered[
                filtered["employee_count"].notna()
                & (filtered["employee_count"] >= parsed.min_employees)
            ]

        if parsed.max_employees is not None:
            filtered = filtered[
                filtered["employee_count"].notna()
                & (filtered["employee_count"] <= parsed.max_employees)
            ]

        if parsed.min_revenue is not None:
            filtered = filtered[
                filtered["revenue"].notna()
                & (filtered["revenue"] >= parsed.min_revenue)
            ]

        if parsed.max_revenue is not None:
            filtered = filtered[
                filtered["revenue"].notna()
                & (filtered["revenue"] <= parsed.max_revenue)
            ]

        if parsed.founded_after is not None:
            filtered = filtered[
                filtered["year_founded"].notna()
                & (filtered["year_founded"] >= parsed.founded_after)
            ]

        if parsed.founded_before is not None:
            filtered = filtered[
                filtered["year_founded"].notna()
                & (filtered["year_founded"] <= parsed.founded_before)
            ]

        if parsed.is_public is not None:
            filtered = filtered[filtered["is_public"] == parsed.is_public]

        # If no match after hard filtering
        # Process the full dataset
        if filtered.empty:
            filtered = self.df.copy()

        # Vector search execution on candidate subset
        query_vec = self.embedding_engine.encode_query(parsed.semantic_query)
        subset_positions = filtered.index.tolist()
        subset_vecs = self.embeddings[subset_positions]

        scores = util.cos_sim(query_vec, subset_vecs)[0].cpu().numpy()
        filtered["similarity_score"] = scores

        return (
            filtered.sort_values(by="similarity_score", ascending=False)
            .drop_duplicates(subset=["operational_name"], keep="first")
            .head(top_k)
        )