import numpy as np
import pandas as pd
import pytest

from cellshape3d.config import Config, FibresCfg
from cellshape3d.fibres import align_to_fibres, axial_diff, fibre_fields, signed_rel_angle
from cellshape3d.stats import anova_tests, compact_letters, group_tests, holm


def stripes(angle_deg, n=160, period=16, width=6):
    t = np.radians(angle_deg)
    yy, xx = np.mgrid[:n, :n]
    d = (xx * np.sin(t) + yy * np.cos(t)) % period
    return np.where(d < width, 80.0, 200.0).astype(np.float32)


@pytest.mark.parametrize("angle", [0, 45, 120])
def test_fibre_angle_and_coherence(angle):
    flat, mask, fib, coh = fibre_fields(stripes(angle), FibresCfg(flat_sigma_px=30, min_fibre_area_px=20))
    c = slice(30, 130)
    assert mask[c, c].mean() == pytest.approx(6 / 16, abs=0.15)
    assert np.median(axial_diff(fib[c, c], angle)) < 3
    assert np.median(coh[c, c]) > 0.8


def test_crossing_fibres_have_low_coherence():
    tl = np.minimum(stripes(0), stripes(90))
    _, _, _, coh = fibre_fields(tl, FibresCfg(flat_sigma_px=30, min_fibre_area_px=20))
    assert np.median(coh[30:130, 30:130]) < 0.5


def test_angle_helpers():
    np.testing.assert_allclose(axial_diff([10, 170, 0], [170, 10, 90]), [20, 20, 90])
    np.testing.assert_allclose(signed_rel_angle([10, 170, 100], [170, 10, 10]), [20, -20, 90])


def test_align_to_fibres():
    mask = np.zeros((50, 50), bool)
    mask[:, 20:24] = True
    fib = np.full((50, 50), 90.0, np.float32)
    coh = np.ones((50, 50), np.float32)
    df = pd.DataFrame(dict(y_um=[25 * 1.26, 25 * 1.26], x_um=[21 * 1.26, 40 * 1.26], orientation_deg=[80.0, 0.0]))
    out = align_to_fibres(df, mask, fib, coh, 1.26, 10.0)
    assert out.on_fibre.tolist() == [True, False]
    np.testing.assert_allclose(out.align_deg, [10, 90])
    assert out.dist_to_fibre_um.iloc[1] == pytest.approx(17 * 1.26)


def test_holm_matches_statsmodels():
    from statsmodels.stats.multitest import multipletests
    p = [0.01, 0.04, 0.03, 0.2]
    np.testing.assert_allclose(holm(p), multipletests(p, method="holm")[1])


def test_compact_letters():
    g = ["G", "GS", "S"]
    assert compact_letters(g, set(), g) == {"G": "a", "GS": "a", "S": "a"}
    assert compact_letters(g, {frozenset({"G", "GS"}), frozenset({"G", "S"})}, g) == {"G": "a", "GS": "b", "S": "b"}
    L = compact_letters(g, {frozenset({"G", "S"})}, g)
    assert L == {"G": "a", "GS": "ab", "S": "b"}


def synthetic_summary(rng, effect=None, teams=8):
    effect = effect or {"G": 1.2, "GS": 1.4, "S": 1.8}
    rows = []
    for t in range(teams):
        tb = rng.normal(0, 0.02)
        for g, ar in effect.items():
            for line in ("HEK", "HeLa"):
                rows.append(dict(image=f"{g}-{line}-{t}", group=g, cell_line=line, team=t,
                                 cell_AR_median=ar * np.exp(tb + rng.normal(0, 0.03))))
    return pd.DataFrame(rows)


def test_anova_recovers_ratios(rng):
    cfg = Config()
    rows = pd.DataFrame(anova_tests(synthetic_summary(rng), "cell_AR_median", "cell", cfg))
    om = rows[rows.test.str.startswith("two-way")].set_index("term")
    assert om.loc["scaffold", "p"] < 1e-10
    assert om.loc["scaffold", "df"] == 2 and om.loc["team (block)", "df"] == 7
    c = rows[(rows.term == "S vs G") & (rows.cell_line == "HEK")].iloc[0]
    assert c.AR_ratio == pytest.approx(1.5, rel=0.05)
    assert c.ratio_ci_lo < 1.5 < c.ratio_ci_hi
    assert set(rows.term[rows.test.str.startswith("ANOVA contrast")]) == {"GS vs G", "S vs G", "S vs GS"}


def test_anova_without_block(rng):
    cfg = Config()
    cfg.design.block = None
    rows = pd.DataFrame(anova_tests(synthetic_summary(rng).drop(columns="team"), "cell_AR_median", "cell", cfg))
    assert "team (block)" not in set(rows.term)


def test_group_tests(rng):
    cfg = Config()
    rows = pd.DataFrame(group_tests(synthetic_summary(rng), "cell_AR_median", "cell", cfg))
    kw = rows[rows.test.str.startswith("Kruskal")]
    assert len(kw) == 2 and (kw.p < 0.001).all()
    assert rows.p_holm.dropna().between(0, 1).all()
