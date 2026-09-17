import pandas as pd
from sentence_transformers import SentenceTransformer
import config

class EmbeddingEngine:
    def __init__(self, model_name: str = config.EMBEDDER_MODEL_NAME):
        self.embedder = SentenceTransformer(model_name)

    def row_to_text(self, row) -> str:
        def clean(val):
            if isinstance(val, list):
                return ", ".join(map(str, val))
            return "" if pd.isna(val) or val is None else str(val)

        return (
            f"Company: {clean(row.get('operational_name'))} | "
            f"Description: {clean(row.get('description'))} | "
            f"NAICS: {clean(row.get('primary_naics_label'))} | "
            f"Offerings: {clean(row.get('core_offerings'))} | "
            f"Target Markets: {clean(row.get('target_markets'))} | "
            f"Business Model: {clean(row.get('business_model'))}"
        )

    def build_embeddings(self, df: pd.DataFrame):
        combined_text = df.apply(self.row_to_text, axis=1).tolist()
        return self.embedder.encode(
            combined_text,
            show_progress_bar=True,
            convert_to_tensor=True,
        )

    def encode_query(self, query: str):
        return self.embedder.encode([query], convert_to_tensor=True)