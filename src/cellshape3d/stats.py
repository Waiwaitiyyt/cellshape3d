"""Stage 6: inclusion, per-image summaries and group statistics.

Statistical unit = image (one construct). Objects within an image are not independent, so per-object pooling is only
used descriptively. Per image: median AR, mean AR, fraction AR > 2 and alignment summaries.
Primary analysis: two-way ANOVA on log(per-image median AR), group x cell line with interaction, plus a blocking
factor (e.g. team) when configured; type II sums of squares (unbalanced designs). Pairwise group contrasts within each
cell line come from the same model (pooled residual variance; block terms cancel), Holm-adjusted per cell line, and
are reported as AR ratios exp(diff) with 95 % CI. Diagnostics: Shapiro-Wilk on residuals, Levene across cells.
Sensitivity analysis: Kruskal-Wallis per cell line + pairwise Mann-Whitney U with Holm.
Alignment: per-image median |object axis - fibre axis| (0 = parallel, 45 = random) of elongated objects on a fibre with
unambiguous direction; two-sided one-sample Wilcoxon of (median - 45).
"""
from __future__ import annotations

import re
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from .config import Config


def holm(p) -> np.ndarray:
    """Holm step-down adjusted P values."""
    p = np.asarray(p, float)
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i]))
        adj[i] = run
    return adj


def apply_inclusion(nuc: pd.DataFrame, cel: pd.DataFrame, cfg: Config):
    """Nuclei: not on the xy border, projected area within range. Cells: not on the border, not a sheet, area >= min."""
    s = cfg.stats
    nuc = nuc.copy()
    cel = cel.copy()
    nuc["included"] = ~nuc.touches_xy_border.astype(bool) & nuc.proj_area_um2.between(*s.nuc_area_um2)
    cel["included"] = (~cel.touches_xy_border.astype(bool) & ~cel.is_sheet.fillna(False).astype(bool)
                       & (cel.proj_area_um2 >= s.cell_min_area_um2))
    return nuc, cel


def aligned_mask(df: pd.DataFrame, cfg: Config, elongated: bool = True) -> pd.Series:
    """Objects on a fibre at positions with coherence >= threshold (and AR > elong_ar if ``elongated``)."""
    if "on_fibre" not in df:
        return pd.Series(False, index=df.index)
    m = df.on_fibre.fillna(False).astype(bool) & (df.fibre_coherence >= cfg.stats.align_coherence_min)
    return m & (df.AR_xy > cfg.stats.elong_ar) if elongated else m


def summarise(df: pd.DataFrame, prefix: str, cfg: Config) -> pd.DataFrame:
    """Per-image summary of included objects."""
    s = cfg.stats
    g = df.groupby("image")
    out = pd.DataFrame({f"{prefix}_n": g.size(), f"{prefix}_AR_median": g.AR_xy.median(),
                        f"{prefix}_AR_mean": g.AR_xy.mean(),
                        f"{prefix}_frac_AR_gt2": g.AR_xy.apply(lambda a: (a > 2).mean())})
    if "align_deg" in df:
        a = df[aligned_mask(df, cfg, elongated=False)]
        ga = a.groupby("image")
        out[f"{prefix}_align_n"] = ga.size()
        out[f"{prefix}_align_median_deg"] = ga.align_deg.median()
        out[f"{prefix}_frac_aligned"] = ga.align_deg.apply(lambda v: (v <= s.aligned_deg).mean())
        # orientation of near-round objects is meaningless -> restrict to elongated objects
        e = a[a.AR_xy > s.elong_ar].groupby("image")
        out[f"{prefix}_align_elong_median_deg"] = e.align_deg.median()
        out[f"{prefix}_align_elong_n"] = e.size()
    return out


def per_image_summary(nuc: pd.DataFrame, cel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    meta_cols = [c for c in ("group", "cell_line", "team") if c in nuc]
    meta = pd.concat([nuc, cel]).groupby("image")[meta_cols].first()
    summ = meta.join(summarise(nuc[nuc.included], "nuc", cfg)).join(summarise(cel[cel.included], "cell", cfg))
    sheet_area = cel.proj_area_um2.where(cel.is_sheet.fillna(False).astype(bool), 0.0).groupby(cel.image).sum()
    summ["cell_sheet_frac_area"] = sheet_area / cel.groupby("image").proj_area_um2.sum().clip(lower=1e-9)
    for p in ["nuc", "cell"]:
        summ[f"{p}_ok"] = summ[f"{p}_n"].fillna(0) >= cfg.stats.min_objects_per_image
    return summ.reset_index()


def group_tests(summary: pd.DataFrame, col: str, label: str, cfg: Config) -> list[dict]:
    """Kruskal-Wallis across groups per cell line + pairwise Mann-Whitney U (Holm within cell line)."""
    rows = []
    groups = cfg.design.groups
    for line in cfg.design.cell_lines:
        s = summary[(summary.cell_line == line) & summary[col].notna()]
        gv = {g: s[s.group == g][col].values for g in groups}
        if len(groups) > 2 and all(len(v) >= 2 for v in gv.values()):
            H, p = stats.kruskal(*gv.values())
            rows.append(dict(metric=label, cell_line=line, test=f"Kruskal-Wallis {'/'.join(groups)}", stat=H, p=p,
                             p_holm=np.nan, **{f"n_{g}": len(v) for g, v in gv.items()},
                             **{f"median_{g}": np.median(v) for g, v in gv.items()}))
        pair = []
        for a, b in combinations(groups, 2):
            if len(gv[a]) >= 2 and len(gv[b]) >= 2:
                U, p = stats.mannwhitneyu(gv[a], gv[b], alternative="two-sided")
                pair.append(dict(metric=label, cell_line=line, test=f"Mann-Whitney {a} vs {b}", stat=U, p=p,
                                 median_a=np.median(gv[a]), median_b=np.median(gv[b])))
        if pair:
            for r, pa in zip(pair, holm([r["p"] for r in pair])):
                r["p_holm"] = pa
            rows += pair
    return rows


def anova_tests(summary: pd.DataFrame, col: str, label: str, cfg: Config) -> list[dict]:
    """Two-way ANOVA (group x cell line [+ block]) on log AR, within-line Holm contrasts and diagnostics."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    des = cfg.design
    d = summary[summary[col].notna() & summary.group.isin(des.groups) & summary.cell_line.isin(des.cell_lines)].copy()
    d["y"] = np.log(d[col])
    lines = [l for l in des.cell_lines if (d.cell_line == l).any()]
    terms = {"C(group)": "scaffold"}
    rhs = "C(group)"
    if len(lines) > 1:
        rhs = "C(group) * C(cell_line)"
        terms["C(cell_line)"] = "cell line"
    block = des.block if des.block and des.block in d and d[des.block].nunique() > 1 else None
    if block:
        d["_block"] = d[block].astype(str)
        rhs += " + C(_block)"
        terms["C(_block)"] = f"{block} (block)"
    if len(lines) > 1:
        terms["C(group):C(cell_line)"] = "scaffold x cell line"
    m = smf.ols(f"y ~ {rhs}", d).fit()
    a = sm.stats.anova_lm(m, typ=2)
    rows = [dict(metric=label, test="two-way ANOVA (type II), log AR", term=name, df=a.loc[t, "df"],
                 df_resid=m.df_resid, F=a.loc[t, "F"], p=a.loc[t, "PR(>F)"], n_images=len(d))
            for t, name in terms.items()]
    cells = [v.values for _, v in d.groupby(["group", "cell_line"]).y if len(v) >= 2]
    rows.append(dict(metric=label, test="diagnostics", term="Shapiro-Wilk residuals",
                     p=stats.shapiro(m.resid).pvalue if len(d) >= 3 else np.nan, n_images=len(d)))
    rows.append(dict(metric=label, test="diagnostics", term=f"Levene ({len(cells)} scaffold x line cells)",
                     p=stats.levene(*cells).pvalue if len(cells) >= 2 else np.nan, n_images=len(d)))
    names = list(m.model.exog_names)
    term_re = re.compile(r"C\((\w+)\)\[T\.(.+)\]")

    def x(g, line):
        """Treatment-coded design row for (group, line) at the reference block (block terms cancel in contrasts)."""
        vals = {"group": g, "cell_line": line}
        v = np.zeros(len(names))
        for k, n in enumerate(names):
            if n == "Intercept":
                v[k] = 1
                continue
            parts = [term_re.fullmatch(s) for s in n.split(":")]
            v[k] = float(all(p is not None and vals.get(p.group(1)) == p.group(2) for p in parts))
        return v

    for line in lines:
        con = []
        for g1, g2 in combinations(des.groups, 2):
            n_a = int((d.cell_line.eq(line) & d.group.eq(g1)).sum())
            n_b = int((d.cell_line.eq(line) & d.group.eq(g2)).sum())
            if n_a == 0 or n_b == 0:
                continue
            r = m.t_test(x(g2, line) - x(g1, line))
            est = float(np.squeeze(r.effect))
            lo, hi = np.squeeze(r.conf_int())
            con.append(dict(metric=label, test="ANOVA contrast (within cell line)", term=f"{g2} vs {g1}", cell_line=line,
                            t=float(np.squeeze(r.tvalue)), p=float(np.squeeze(r.pvalue)), AR_ratio=np.exp(est),
                            ratio_ci_lo=np.exp(lo), ratio_ci_hi=np.exp(hi), n_a=n_a, n_b=n_b))
        for r, pa in zip(con, holm([c["p"] for c in con])):
            r["p_holm"] = pa
        rows += con
    return rows


def alignment_tests(summ: pd.DataFrame, cfg: Config) -> list[dict]:
    """One-sample Wilcoxon of per-image median alignment (elongated objects) against 45 degrees."""
    rows = []
    for p in ["nuc", "cell"]:
        col = f"{p}_align_elong_median_deg"
        if col not in summ:
            continue
        for line in cfg.design.cell_lines:
            for g in cfg.design.fibre_groups:
                v = summ[(summ.cell_line == line) & (summ.group == g)][col].dropna().values
                if len(v) >= 3:
                    w, pv = stats.wilcoxon(v - 45.0)
                    rows.append(dict(metric=f"{p} alignment to fibre (elongated, median deg)", cell_line=line,
                                     test=f"Wilcoxon {g}: median vs 45 deg", stat=w, p=pv, n=len(v), median=np.median(v)))
    return rows


METRICS = [("nuc", "nucleus AR (per-image median)"), ("cell", "cell AR (per-image median)")]


def run_stats(nuc: pd.DataFrame, cel: pd.DataFrame, cfg: Config):
    """Returns (nuclei table, cells table, per-image summary, group_tests, anova_tests) as DataFrames."""
    nuc, cel = apply_inclusion(nuc, cel, cfg)
    summ = per_image_summary(nuc, cel, cfg)
    rows, arows = [], []
    for p, lab in METRICS:
        ok = summ[summ[f"{p}_ok"]]
        rows += group_tests(ok, f"{p}_AR_median", lab, cfg)
        if ok[f"{p}_AR_median"].notna().sum() > len(cfg.design.groups) * len(cfg.design.cell_lines):
            arows += anova_tests(ok, f"{p}_AR_median", lab, cfg)
    rows += alignment_tests(summ, cfg)
    return nuc, cel, summ, pd.DataFrame(rows), pd.DataFrame(arows)


def compact_letters(groups, sig_pairs, order) -> dict:
    """Compact letter display (insert-absorb). sig_pairs = set of frozenset({a, b}) that differ significantly;
    ``order`` = groups sorted by their summary value (letter 'a' goes to the first)."""
    sets = [set(groups)]
    for pair in sig_pairs:
        i, j = tuple(pair)
        new = []
        for s in sets:
            if i in s and j in s:
                new += [s - {i}, s - {j}]
            else:
                new.append(s)
        sets = [s for s in new if not any(s < t for t in new)]
        sets = [s for k, s in enumerate(sets) if s not in sets[:k]]
    sets.sort(key=lambda s: min(order.index(g) for g in s))
    return {g: "".join("abcdefghij"[k] for k, s in enumerate(sets) if g in s) for g in groups}
