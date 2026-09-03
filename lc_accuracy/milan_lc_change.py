"""
Milan land-cover change 2000 -> 2020 (ESA CCI / C3S LCCS, 300 m).

Clips the two annual LCCS maps to the Milan AOI polygon, writes the clipped
GeoTIFFs, and computes:
  * a pixel-wise change map (unchanged / changed)
  * a class transition matrix (pixel counts and hectares)
  * per-class area for each year and the net change

Outputs, each saved as its own slide-ready PNG in ../output:
  milan_lc_map_2000.png / milan_lc_map_2020.png   LCCS maps
  milan_lc_change_map.png                          changed vs unchanged
  milan_lc_change_by_class.png                     changed cells, by 2020 class
  milan_lc_transition_matrix.png                   transition heatmap (ha)
  milan_lc_net_change.png                          net area change per class

Run with an environment that has rasterio, numpy, pandas, matplotlib:
    & C:/ProgramData/anaconda3/python.exe lc_accuracy/milan_lc_change.py
"""

from __future__ import annotations

import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.patches import Patch

# --- configuration ----------------------------------------------------------

LC_DIR = (
    r"C:\Users\user\OneDrive - Politecnico di Milano"
    r"\File di Daniele Oxoli - GLANCE_working_folder\data\ESA_CCI_Land_Cover"
)
SRC = {
    2000: os.path.join(
        LC_DIR,
        "ESACCI-LC-L4-LCCS-Map-300m-P1Y-2000-v2.0.7cds.area-subset.48.40.30.-10.nc",
    ),
    2020: os.path.join(
        LC_DIR,
        "C3S-LC-L4-LCCS-Map-300m-P1Y-2020-v2.1.1.area-subset.48.40.30.-10.nc",
    ),
}
VARIABLE = "lccs_class"

# Milan AOI polygon. Its bounds (reprojected to the LCCS grid CRS, EPSG:4326)
# define the read window; the polygon itself masks pixels outside the AOI.
AOI_SHP = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "milano_aoi", "milano_aoi.shp")
)

OUT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "output"))
NODATA = 0  # ESA CCI "no_data" flag value

YEAR_A, YEAR_B = 2000, 2020


# --- helpers --------------------------------------------------------------------

def subdataset(path: str) -> str:
    return f'NETCDF:"{path}":{VARIABLE}'


def load_aoi() -> tuple[dict, list]:
    """AOI bounds {left, bottom, right, top} and geometries, in EPSG:4326."""
    g = gpd.read_file(AOI_SHP).to_crs("EPSG:4326")
    left, bottom, right, top = g.total_bounds
    bounds = dict(left=left, bottom=bottom, right=right, top=top)
    return bounds, list(g.geometry)


AOI, AOI_GEOM = load_aoi()


def read_legend(path: str) -> tuple[dict[int, str], dict[int, str]]:
    """Return {value: short_name} and {value: '#rrggbb'} from the band tags."""
    with rasterio.open(subdataset(path)) as src:
        tags = src.tags(1)
    values = [int(v) for v in tags["flag_values"].strip("{}").split(",")]
    names = tags["flag_meanings"].split()
    colors = tags["flag_colors"].split()
    # ESA CCI drops the colour for the leading "no_data" value, so flag_colors
    # is one shorter than flag_values / flag_meanings; align it to the tail.
    color_values = values[len(values) - len(colors):]
    return dict(zip(values, names)), dict(zip(color_values, colors))


def clip(path: str) -> tuple[np.ndarray, rasterio.Affine, dict]:
    """Read the AOI window from a LCCS subdataset."""
    with rasterio.open(subdataset(path)) as src:
        win = from_bounds(**AOI, transform=src.transform)
        win = Window(
            int(round(win.col_off)),
            int(round(win.row_off)),
            int(round(win.width)),
            int(round(win.height)),
        )
        arr = src.read(1, window=win)
        transform = src.window_transform(win)
        # blank out pixels whose centre falls outside the AOI polygon
        outside = geometry_mask(
            AOI_GEOM, out_shape=arr.shape, transform=transform, invert=False
        )
        arr = np.where(outside, NODATA, arr).astype(arr.dtype)
        profile = {
            k: v for k, v in src.profile.items()
            if k not in ("blockxsize", "blockysize", "tiled", "interleave")
        }
        profile.update(
            driver="GTiff",
            height=arr.shape[0],
            width=arr.shape[1],
            transform=transform,
            crs="EPSG:4326",
            count=1,
            dtype=arr.dtype,
            nodata=NODATA,
            compress="deflate",
        )
    return arr, transform, profile


def pixel_area_ha(transform: rasterio.Affine, mean_lat_deg: float) -> float:
    """Approximate area of one grid cell in hectares at the given latitude."""
    deg = 111_320.0  # metres per degree latitude (spherical approx.)
    dx = abs(transform.a) * deg * np.cos(np.radians(mean_lat_deg))
    dy = abs(transform.e) * deg
    return dx * dy / 10_000.0


# --- main ---------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    names, colors = read_legend(SRC[YEAR_A])

    a, ta, pa = clip(SRC[YEAR_A])
    b, tb, pb = clip(SRC[YEAR_B])
    assert a.shape == b.shape, f"grid mismatch: {a.shape} vs {b.shape}"
    print(f"AOI window: {a.shape[1]} cols x {a.shape[0]} rows")

    for year, arr, prof in ((YEAR_A, a, pa), (YEAR_B, b, pb)):
        dst = os.path.join(OUT_DIR, f"milan_lc_{year}.tif")
        with rasterio.open(dst, "w", **prof) as out:
            out.write(arr, 1)
        print(f"wrote {dst}")

    mean_lat = (AOI["bottom"] + AOI["top"]) / 2
    cell_ha = pixel_area_ha(ta, mean_lat)
    print(f"cell area ~ {cell_ha:.3f} ha")

    valid = (a != NODATA) & (b != NODATA)
    fa, fb = a[valid], b[valid]

    # ---- transition matrix ----
    classes = sorted(set(np.unique(fa)) | set(np.unique(fb)))
    labels = [f"{c} {names.get(c, '?')}" for c in classes]
    tm = pd.DataFrame(
        0, index=pd.Index(labels, name=f"{YEAR_A}"),
        columns=pd.Index(labels, name=f"{YEAR_B}"), dtype=int,
    )
    idx = {c: i for i, c in enumerate(classes)}
    for f_, t_ in zip(fa, fb):
        tm.iat[idx[f_], idx[t_]] += 1

    tm.to_csv(os.path.join(OUT_DIR, "milan_lc_transition_pixels.csv"))
    (tm * cell_ha).round(2).to_csv(
        os.path.join(OUT_DIR, "milan_lc_transition_hectares.csv")
    )

    # ---- per-class area + net change ----
    area = pd.DataFrame(index=labels)
    area[f"{YEAR_A}_ha"] = [np.sum(fa == c) * cell_ha for c in classes]
    area[f"{YEAR_B}_ha"] = [np.sum(fb == c) * cell_ha for c in classes]
    area["net_ha"] = area[f"{YEAR_B}_ha"] - area[f"{YEAR_A}_ha"]
    area.round(2).to_csv(os.path.join(OUT_DIR, "milan_lc_class_area.csv"))

    changed = (a != b) & valid
    n_changed = int(changed.sum())
    n_valid = int(valid.sum())
    print(f"changed pixels: {n_changed}/{n_valid} "
          f"({100 * n_changed / n_valid:.1f}%), {n_changed * cell_ha:.1f} ha")

    print("\nper-class area (ha):")
    print(area.round(1).to_string())
    print("\ntop transitions (ha):")
    off = tm.where(~np.eye(len(tm), dtype=bool))
    tops = (off.stack() * cell_ha).sort_values(ascending=False).head(10)
    for (frm, to), ha in tops.items():
        if ha > 0:
            print(f"  {frm:<28s} -> {to:<28s} {ha:8.1f}")

    # ---- standalone figures (one PNG each, sized for slides) ----
    extent = [AOI["left"], AOI["right"], AOI["bottom"], AOI["top"]]
    aspect = (AOI["right"] - AOI["left"]) / (AOI["top"] - AOI["bottom"])
    remap = {int(c): i for i, c in enumerate(classes)}
    lc_cmap = ListedColormap([colors[c] for c in classes])

    def to_index(arr: np.ndarray) -> np.ma.MaskedArray:
        out = np.full(arr.shape, np.nan)
        for c, i in remap.items():
            out[arr == c] = i
        return np.ma.masked_invalid(out)

    def save(fig: plt.Figure, name: str) -> None:
        dst = os.path.join(OUT_DIR, name)
        fig.savefig(dst, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {dst}")

    # 1-2: LCCS maps, one per year, shared legend
    legend_handles = [
        Patch(facecolor=colors[c], edgecolor="0.4", label=f"{c}  {names.get(c, '?')}")
        for c in classes
    ]
    for year, arr in ((YEAR_A, a), (YEAR_B, b)):
        fig, ax = plt.subplots(figsize=(11 * min(aspect, 2.2), 9),
                               layout="constrained")
        ax.imshow(to_index(arr), cmap=lc_cmap, vmin=-0.5, vmax=len(classes) - 0.5,
                  extent=extent, interpolation="nearest")
        ax.set_title(f"Milan land cover (LCCS) \u2013 {year}", fontsize=17)
        ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
        ax.tick_params(labelsize=11)
        ax.legend(handles=legend_handles, loc="center left",
                  bbox_to_anchor=(1.02, 0.5), fontsize=10, frameon=False,
                  title="LCCS class")
        save(fig, f"milan_lc_map_{year}.png")

    # 3: change mask
    fig, ax = plt.subplots(figsize=(11 * min(aspect, 2.2), 9),
                           layout="constrained")
    disp = np.full(a.shape, np.nan)
    disp[valid & (a == b)] = 0
    disp[changed] = 1
    ax.imshow(np.ma.masked_invalid(disp),
              cmap=ListedColormap(["#dcdcdc", "#e6194b"]), vmin=-0.5, vmax=1.5,
              extent=extent, interpolation="nearest")
    ax.set_title(f"Land-cover change {YEAR_A}\u2192{YEAR_B}\n"
                 f"{n_changed} cells changed "
                 f"({100 * n_changed / n_valid:.1f}% of AOI, "
                 f"{n_changed * cell_ha:,.0f} ha)", fontsize=16)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.tick_params(labelsize=11)
    ax.legend(handles=[Patch(facecolor="#dcdcdc", label="unchanged"),
                       Patch(facecolor="#e6194b", label="changed")],
              loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=12,
              frameon=False)
    save(fig, "milan_lc_change_map.png")

    # 4: changed pixels coloured by their 2020 class
    fig, ax = plt.subplots(figsize=(11 * min(aspect, 2.2), 9),
                           layout="constrained")
    ax.set_facecolor("#dcdcdc")
    b_changed = np.where(changed, b, NODATA)
    ax.imshow(to_index(b_changed), cmap=lc_cmap, vmin=-0.5,
              vmax=len(classes) - 0.5, extent=extent, interpolation="nearest")
    to_classes = sorted(set(np.unique(b[changed])) - {NODATA})
    ax.set_title(f"Where land cover changed, coloured by {YEAR_B} class",
                 fontsize=16)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.tick_params(labelsize=11)
    ax.legend(handles=[Patch(facecolor=colors[c], edgecolor="0.4",
                             label=f"{c}  {names.get(c, '?')}")
                       for c in to_classes],
              loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=11,
              frameon=False, title=f"{YEAR_B} class")
    save(fig, "milan_lc_change_by_class.png")

    # 5: transition matrix (off-diagonal, hectares, log colour)
    off_ha = (off.fillna(0) * cell_ha)
    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * len(labels), 0.8 + 0.9 * len(labels)),
                           layout="constrained")
    vmax = max(off_ha.values.max(), cell_ha * 2)
    im = ax.imshow(np.where(off_ha.values > 0, off_ha.values, np.nan),
                   cmap="magma_r", norm=LogNorm(vmin=cell_ha, vmax=vmax))
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel(f"{YEAR_B} class", fontsize=13)
    ax.set_ylabel(f"{YEAR_A} class", fontsize=13)
    ax.set_title(f"Class transitions {YEAR_A}\u2192{YEAR_B} (ha, unchanged removed)",
                 fontsize=16)
    thr = np.sqrt(cell_ha * vmax)  # geometric midpoint for text contrast
    for (i, j), v in np.ndenumerate(off_ha.values):
        if v > 0:
            ax.text(j, i, f"{v:,.0f}", ha="center", va="center", fontsize=10,
                    color="white" if v > thr else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="hectares")
    save(fig, "milan_lc_transition_matrix.png")

    # 6: net area change per class
    net = area["net_ha"].sort_values()
    fig, ax = plt.subplots(figsize=(11, 0.6 + 0.5 * len(net)), layout="constrained")
    ax.barh(range(len(net)), net.values,
            color=["#c0392b" if v < 0 else "#27ae60" for v in net.values])
    ax.set_yticks(range(len(net))); ax.set_yticklabels(net.index, fontsize=11)
    ax.axvline(0, color="0.3", lw=0.8)
    span = net.max() - net.min()
    ax.set_xlim(net.min() - 0.28 * span, net.max() + 0.12 * span)
    ax.set_xlabel("net area change 2000\u21922020 (ha)", fontsize=13)
    ax.set_title("Net land-cover area change per class", fontsize=16)
    pad = 0.01 * span
    for i, v in enumerate(net.values):
        ax.text(v + (pad if v >= 0 else -pad), i, f"{v:,.0f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=10)
    save(fig, "milan_lc_net_change.png")


if __name__ == "__main__":
    main()
