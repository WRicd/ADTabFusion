import inspect

import pandas as pd
import pytest

from src.external.base_adapter import ExternalDatasetAdapter
from src.external.oasis_adapter import OasisAdapter
from src.robustness import summarize_missing_modality

REQUIRED_METHODS = ("load", "normalize_schema", "map_diagnosis", "build_index_visit", "align_features")


def test_base_adapter_cannot_be_instantiated_and_declares_the_full_contract():
    assert ExternalDatasetAdapter.__abstractmethods__ == frozenset(REQUIRED_METHODS)
    with pytest.raises(TypeError):
        ExternalDatasetAdapter()


def test_partial_adapters_are_rejected():
    class PartialAdapter(ExternalDatasetAdapter):
        def load(self):
            return None

    with pytest.raises(TypeError):
        PartialAdapter()


def test_complete_adapter_can_be_instantiated():
    class StubAdapter(ExternalDatasetAdapter):
        def load(self):
            return "loaded"

        def normalize_schema(self):
            return "normalized"

        def map_diagnosis(self):
            return "mapped"

        def build_index_visit(self):
            return "index"

        def align_features(self, frozen_schema):
            return list(frozen_schema)

    adapter = StubAdapter()
    assert adapter.load() == "loaded"
    assert adapter.align_features(["AGE", "MMSE"]) == ["AGE", "MMSE"]


def test_oasis_adapter_is_an_explicit_unconfigured_placeholder():
    adapter = OasisAdapter()
    assert isinstance(adapter, ExternalDatasetAdapter)
    with pytest.raises(NotImplementedError, match="OASIS-3 data access is not configured."):
        adapter.load()
    for name in REQUIRED_METHODS[1:]:
        method = getattr(adapter, name)
        arguments = [None] * (len(inspect.signature(method).parameters))
        with pytest.raises(NotImplementedError):
            method(*arguments)


def test_missing_modality_summary_passes_results_through_unchanged():
    results = pd.DataFrame({"modality": ["cognitive"], "macro_f1": [0.71]})

    summary = summarize_missing_modality(results)

    assert summary.equals(results)
