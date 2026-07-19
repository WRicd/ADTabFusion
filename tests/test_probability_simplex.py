import numpy as np
import pytest

from src.external.inference import probability_frame


def test_probability_simplex_is_enforced():
    result = probability_frame(np.array([[0.2, 0.5, 0.3]]))
    assert result.loc[0, "predicted_class"] == "MCI"
    with pytest.raises(ValueError):
        probability_frame(np.array([[0.2, 0.2, 0.2]]))
