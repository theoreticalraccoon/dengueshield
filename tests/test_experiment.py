"""The shared Brazil run: split discipline and imbalance weighting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dengue import experiment
from dengue.config import TRAIN_END, VAL_END


def frame(years=(2019, 2021, 2022, 2023, 2024, 2025)):
    return pd.DataFrame({"anio": list(years), "y": [0, 1, 0, 1, 0, 1]})


def test_split_is_temporal_and_disjoint():
    tr, va, te = experiment.split(frame())
    assert set(tr.anio) == {2019, 2021}
    assert set(va.anio) == {2022, 2023}
    assert set(te.anio) == {2024, 2025}
    assert len(tr) + len(va) + len(te) == 6


def test_split_boundaries_follow_config():
    tr, va, te = experiment.split(frame())
    assert tr.anio.max() <= TRAIN_END
    assert va.anio.min() > TRAIN_END
    assert va.anio.max() <= VAL_END
    assert te.anio.min() > VAL_END


def test_no_row_appears_in_two_splits():
    df = frame()
    tr, va, te = experiment.split(df)
    seen = pd.concat([tr, va, te]).index
    assert len(seen) == len(set(seen)) == len(df)


def test_test_years_are_never_in_training():
    """The locked test set is the project's central protocol commitment."""
    tr, _, te = experiment.split(frame())
    assert set(tr.anio) & set(te.anio) == set()


@pytest.mark.parametrize(
    ("y", "expected"),
    [([0, 0, 0, 1], 3.0), ([0, 1], 1.0), ([0, 0], 2.0)],
)
def test_pos_weight(y, expected):
    assert experiment.pos_weight(np.array(y)) == expected


def test_pos_weight_survives_a_fold_with_no_positives():
    """A spatial or temporal fold can legitimately contain a single class.

    The max(..., 1) guard means the weight degrades to the negative count rather
    than dividing by zero. Preserved from the original inline expression.
    """
    assert experiment.pos_weight(np.array([0, 0, 0])) == 3.0


def test_run_applies_the_weight_to_fit_params():
    run = experiment.Run(
        sup=pd.DataFrame(),
        feats=["a"],
        tr=pd.DataFrame({"y": [0, 0, 0, 1]}),
        va=pd.DataFrame(),
        te=pd.DataFrame(),
        horizon=2,
        outbreak_inc=100.0,
        params={"objective": "binary"},
    )
    assert run.spw == 3.0
    assert run.fit_params()["scale_pos_weight"] == 3.0
    assert run.fit_params(num_leaves=7)["num_leaves"] == 7


def test_run_describes_its_configuration():
    run = experiment.Run(
        sup=pd.DataFrame(),
        feats=["a", "b"],
        tr=pd.DataFrame({"y": [0, 1]}),
        va=pd.DataFrame({"y": [0]}),
        te=pd.DataFrame({"y": [1]}),
        horizon=2,
        outbreak_inc=100.0,
    )
    text = run.describe()
    assert "horizon=2w" in text
    assert "features=2" in text
