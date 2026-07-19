from __future__ import annotations

from abc import ABC, abstractmethod


class ExternalDatasetAdapter(ABC):
    @abstractmethod
    def load(self):
        raise NotImplementedError

    @abstractmethod
    def normalize_schema(self):
        raise NotImplementedError

    @abstractmethod
    def map_diagnosis(self):
        raise NotImplementedError

    @abstractmethod
    def build_index_visit(self):
        raise NotImplementedError

    @abstractmethod
    def align_features(self, frozen_schema):
        raise NotImplementedError
