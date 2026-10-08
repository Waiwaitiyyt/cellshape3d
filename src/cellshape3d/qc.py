"""Quality-control images for every stage (PNG, written next to the stage outputs in ``qc/``)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from skimage.segmentation import find_boundaries  # noqa: E402


def _norm(x, lo=1, hi=99.8):
    a, b = np.percentile(x, [lo, hi])
    return np.clip((x - a) / max(b - a, 1e-6), 0, 1)


def mip_rgb(mip_cyx, ch_dapi, ch_actin):
    """Red = actin, blue/cyan = DAPI."""
    dapi, act = _norm(mip_cyx[ch_dapi].astype(float)), _norm(mip_cyx[ch_actin].astype(float))
    return np.dstack([act, 0.4 * dapi, dapi])


def contact_sheet(mips: dict, path, render, title_size=8, cols=6):
    n = len(mips)
    rows = max(int(np.ceil(n / cols)), 1)
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.1), squeeze=False)
    for a in ax.ravel():
        a.axis("off")
    for a, (name, m) in zip(ax.ravel(), mips.items()):
        im = render(m)
        a.imshow(im, cmap=None if im.ndim == 3 else "gray")
        a.set_title(name, fontsize=title_size)
    plt.tight_layout()
    plt.savefig(path, dpi=90)
    plt.close(fig)


def preprocess_qc(path, title, dapi, actin, tl, zprofile):
    zm = int(np.argmax(np.array(zprofile)))
    panels = [(dapi.max(0), "DAPI MIP (rel.)"), (actin.max(0), "actin MIP (rel.)"), (actin[zm], f"actin z={zm} (rel.)")]
    if tl is not None:
        panels.append((tl, "TL mean"))
    fig, ax = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5.3), squeeze=False)
    for i, (im, t) in enumerate(panels):
        lo, hi = (0, max(np.percentile(im, 99.5), 0.05)) if i < 3 else np.percentile(im, [1, 99])
        h = ax[0, i].imshow(im, cmap="magma" if i < 3 else "gray", vmin=lo, vmax=hi)
        ax[0, i].set_title(t)
        ax[0, i].axis("off")
        plt.colorbar(h, ax=ax[0, i], fraction=0.046)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=70)
    plt.close(fig)


def _outline(ax, lab2d, colour):
    b = find_boundaries(lab2d)
    ov = np.zeros((*b.shape, 4))
    if callable(colour):
        ov[b, :3] = colour(lab2d[b])
        ov[b, 3] = 1
    else:
        ov[b] = colour
    ax.imshow(ov)


def nuclei_qc(path, title, x, lab):
    zc = int(np.argmax((lab > 0).sum((1, 2))))
    fig, ax = plt.subplots(1, 3, figsize=(21, 7.4))
    mip = x.max(0)
    for a in ax[:2]:
        a.imshow(mip, cmap="gray", vmax=max(np.percentile(mip, 99.7), 1e-6))
    ax[0].set_title("DAPI MIP")
    _outline(ax[1], lab.max(0), (1, 0.2, 0.2, 1))
    ax[1].set_title(f"MIP + nuclei (max-projected labels), n={lab.max()}")
    sl = x[zc]
    ax[2].imshow(sl, cmap="gray", vmax=max(np.percentile(sl, 99.8), 1e-6))
    _outline(ax[2], lab[zc], (0, 1, 1, 1))
    ax[2].set_title(f"z={zc}")
    for a in ax:
        a.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=75)
    plt.close(fig)


def cells_qc(path, title, xs, cells, nuc):
    rng = np.random.default_rng(0)
    cmap = rng.random((int(cells.max()) + 1, 3))
    cmap[0] = 0
    zc = int(np.argmax((cells > 0).sum((1, 2))))
    mip = xs.max(0)
    vmax = max(np.percentile(mip, 99.7), 1e-6)
    fig, ax = plt.subplots(1, 3, figsize=(21, 7.4))
    ax[0].imshow(mip, cmap="gray", vmax=vmax)
    ax[0].set_title("actin MIP (smoothed)")
    ax[1].imshow(mip, cmap="gray", vmax=vmax)
    _outline(ax[1], cells.max(0), lambda v: cmap[v])
    ax[1].set_title(f"projected cells n={len(np.unique(cells)) - 1}")
    sl = xs[zc]
    ax[2].imshow(sl, cmap="gray", vmax=max(np.percentile(sl, 99.7), 1e-6))
    _outline(ax[2], cells[zc], lambda v: cmap[v])
    _outline(ax[2], (nuc[zc] > 0).astype(np.uint8), (0, 1, 1, 0.8))
    ax[2].set_title(f"z={zc}: cells (colour) + nuclei (cyan)")
    for a in ax:
        a.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=75)
    plt.close(fig)


def cell_crops_qc(path, title, actin, cells, nuc):
    """Four zoomed quadrants: actin MIP | filled, randomly coloured projected cells + nucleus outlines."""
    m, p, pn = actin.max(0), cells.max(0), nuc.max(0) > 0
    rng = np.random.default_rng(1)
    cm = rng.random((int(cells.max()) + 1, 3)) * 0.8 + 0.2
    cm[0] = 0
    H, W = m.shape
    fig, ax = plt.subplots(2, 4, figsize=(24, 12))
    for k, (y, x) in enumerate([(0, 0), (0, W // 2), (H // 2, 0), (H // 2, W // 2)]):
        sl = (slice(y, y + H // 2), slice(x, x + W // 2))
        mm = m[sl]
        vmax = max(np.percentile(mm, 99.7), 1e-6)
        A, B = ax[k // 2, (k % 2) * 2], ax[k // 2, (k % 2) * 2 + 1]
        A.imshow(mm, cmap="gray", vmax=vmax)
        B.imshow(mm, cmap="gray", vmax=vmax)
        pp = p[sl]
        ov = np.zeros((*mm.shape, 4))
        ov[..., :3] = cm[pp]
        ov[..., 3] = 0.45 * (pp > 0)
        B.imshow(ov)
        if pn[sl].any():
            B.contour(pn[sl], [0.5], colors="cyan", linewidths=0.4)
    for A in ax.ravel():
        A.axis("off")
    fig.suptitle(f"{title}: actin MIP | projected cells (colour) + nuclei (cyan), 4 quadrants")
    plt.tight_layout()
    plt.savefig(path, dpi=60)
    plt.close(fig)


def fibres_qc(path, title, flat, mask, fib, coh, step=16):
    fig, ax = plt.subplots(1, 2, figsize=(14, 7.3))
    ax[0].imshow(flat, cmap="gray")
    if mask.any():
        ax[0].contour(mask, [0.5], colors="y", linewidths=0.5)
    ax[0].set_title("TL (flat-field) + fibre mask")
    ax[1].imshow(flat, cmap="gray")
    Y, X = np.mgrid[step // 2:flat.shape[0]:step, step // 2:flat.shape[1]:step]
    a = np.radians(fib[Y, X])
    c = coh[Y, X]
    m = mask[Y, X]
    # quiver's default angles="uv" draws v towards screen-up, matching the image-up angle convention
    ax[1].quiver(X[m], Y[m], np.cos(a[m]), np.sin(a[m]), c[m], cmap="autumn", pivot="mid", headwidth=0,
                 headlength=0, headaxislength=0, scale=40, width=0.003)
    ax[1].set_title("local fibre axis (colour = coherence)")
    for A in ax:
        A.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=75)
    plt.close(fig)
