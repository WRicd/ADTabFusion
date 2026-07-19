from __future__ import annotations

import pandas as pd

from src.tadpole.dictionary_parser import (
    build_dictionary_index,
    load_tadpole_dictionary,
    match_dictionary_columns,
)


def test_dictionary_parser_matches_full_source_suffix(tmp_path):
    path = tmp_path / "dictionary.csv"
    pd.DataFrame(
        {
            "FLDNAME": ["ABETA_UPENNBIOMK9_04_19_17"],
            "TBLNAME": ["UPENNBIOMK9"],
            "TEXT": ["Amyloid beta in CSF"],
            "TYPE": ["numeric"],
            "UNITS": ["pg/mL"],
        }
    ).to_csv(path, index=False)

    dictionary = load_tadpole_dictionary(path)
    matches = match_dictionary_columns(["ABETA_UPENNBIOMK9_04_19_17"], dictionary)

    assert matches["ABETA_UPENNBIOMK9_04_19_17"]["TBLNAME"] == "UPENNBIOMK9"
    assert len(build_dictionary_index(dictionary)) == 1

