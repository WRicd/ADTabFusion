from __future__ import annotations

from src.external.base_adapter import ExternalDatasetAdapter


class OasisAdapter(ExternalDatasetAdapter):
    """Interface placeholder; OASIS-3 access is intentionally not required in Phase D."""

    def load(self):
        raise NotImplementedError("OASIS-3 data access is not configured.")

    def normalize_schema(self):
        raise NotImplementedError

    def map_diagnosis(self):
        raise NotImplementedError

    def build_index_visit(self):
        raise NotImplementedError

    def align_features(self, frozen_schema):
        raise NotImplementedError
