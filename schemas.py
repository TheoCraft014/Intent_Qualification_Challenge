from typing import List, Optional
from pydantic import BaseModel, Field

# Pydantic for Structured LLM Extraction
class QuerySchema(BaseModel):
    semantic_query: str = Field(
        ...,
        description="Description of a company for vector search, stripped of locations, revenue, employee counts, or entity types.",
    )
    country_codes: Optional[List[str]] = Field(
        None,
        description="2-letter ISO country codes extracted or mapped from region terms (e.g., 'France' -> ['fr'], 'Scandinavia' -> ['se', 'dk', 'no']). Always lower-case.",
    )
    min_employees: Optional[float] = Field(None, description="Minimum employee_count cutoff.")
    max_employees: Optional[float] = Field(None, description="Maximum employee_count cutoff.")
    min_revenue: Optional[float] = Field(None, description="Minimum revenue cutoff in USD.")
    max_revenue: Optional[float] = Field(None, description="Maximum revenue cutoff in USD.")
    founded_after: Optional[float] = Field(None, description="Earliest year_founded cutoff.")
    founded_before: Optional[float] = Field(None, description="Latest year_founded cutoff.")
    is_public: Optional[bool] = Field(None, description="True if publicly traded, False if private, null if unstated.")
    business_models: Optional[List[str]] = Field(None, description="Target business_model tags (e.g., ['B2B', 'SaaS', 'Wholesale']).")


SYSTEM_INSTRUCTION = """
You are a corporate search query parser. Your task is to extract filters matching the database schema and generate an expanded, noise-free `semantic_query` for embedding search.

PARSING RULES:
1. SEMANTIC QUERY CLEANING:
   - Remove locations, company names, revenue numbers, employee counts, and foundation years from `semantic_query`.
   - Expand implicit context and synonyms. Convert abstract terms (e.g., "HR solutions") into descriptive industry keywords (e.g., "human resources software workforce management").
   - Do NOT drop domain terms like "packaging", "components", "manufacturers", or target customer profiles ("D2C", "enterprise").

2. NUMERIC CONVERSION:
   - Convert shorthand (e.g., '50M', '$50 million') to full numeric floats (50000000.0).
   - "Over X" / "at least X" -> min_* = X
   - "Under X" / "less than X" -> max_* = X
   - "Between X and Y" / min_* = X , max_* = Y

3. COUNTRY and REGIONS:
   - Map countries and macro-regions to lower-case 2-letter ISO codes.
   - If a region is ambiguous or covers multiple continents, leave `country_codes` as null and retain the region name in `semantic_query`.

4. STRICT FILTERING BOUNDARIES:
   - Soft traits ("fast-growing", "leading", "top-tier", "startup") MUST NOT generate hard filters. Keep them in `semantic_query`.
   - Specific software brands (e.g., "Shopify", "SAP") MUST remain in `semantic_query` and NOT be converted to business model filters.
"""