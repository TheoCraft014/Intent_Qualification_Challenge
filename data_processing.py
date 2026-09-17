import ast
import uuid
import pandas as pd

def parse_dict_field(val):
    if isinstance(val, dict):
        return val
    if pd.isna(val) or not isinstance(val, str):
        return {}
    try:
        res = ast.literal_eval(val)
        return res if isinstance(res, dict) else {}
    except (ValueError, SyntaxError):
        return {}


# Set dataset to Normal Form 1 for easier search
def process_entire_dataset(jsonl_path: str) -> pd.DataFrame:
    df_raw = pd.read_json(jsonl_path, lines=True)

    if 'company_id' not in df_raw.columns:
        df_raw['company_id'] = [str(uuid.uuid4()) for _ in range(len(df_raw))]

    addresses = df_raw['address'].apply(parse_dict_field)
    df_raw['country_code'] = addresses.apply(lambda x: str(x.get('country_code', '')).lower())
    df_raw['region_name'] = addresses.apply(lambda x: x.get('region_name'))
    df_raw['town'] = addresses.apply(lambda x: x.get('town'))
    df_raw['latitude'] = addresses.apply(lambda x: x.get('latitude'))
    df_raw['longitude'] = addresses.apply(lambda x: x.get('longitude'))

    naics = df_raw['primary_naics'].apply(parse_dict_field)
    df_raw['primary_naics_code'] = naics.apply(lambda x: str(x.get('code', '')) if x.get('code') is not None else None)
    df_raw['primary_naics_label'] = naics.apply(lambda x: x.get('label') if x.get('label') is not None else None)

    sec_naics = df_raw['secondary_naics'].apply(parse_dict_field)
    df_raw['secondary_naics_code'] = sec_naics.apply(lambda x: str(x.get('code', '')) if x.get('code') is not None else None)
    df_raw['secondary_naics_label'] = sec_naics.apply(lambda x: x.get('label') if x.get('label') is not None else None)
    df_raw['secondary_naics_share'] = sec_naics.apply(lambda x: float(x.get('share')) if x.get('share') is not None else None)

    df_fn1 = df_raw.drop(columns=['address', 'primary_naics', 'secondary_naics'], errors='ignore').reset_index(drop=True)
    return df_fn1