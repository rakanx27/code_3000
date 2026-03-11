import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """

    # Merge on the quasi-identifiers specified in the notebook
    merged = anon_df.merge(aux_df, on=["age", "gender", "zip3"], how="inner")

    # Keep only anonymized records that match exactly one person
    counts = merged.groupby("anon_id").size().reset_index(name="count")
    unique_ids = counts[counts["count"] == 1]["anon_id"]

    unique_matches = merged[merged["anon_id"].isin(unique_ids)]

    return unique_matches[["anon_id", "name"]].rename(columns={"name": "matched_name"})


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    return len(matches_df) / len(anon_df)