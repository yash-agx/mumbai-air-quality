"""One standalone chart: the average day's PM2.5 shape at three monitors.

Writes reports/figures/diurnal_peak_hours.png, sized for reading on a phone.

The point of the chart is timing, not level, so each curve is divided by its own
24-hour average and plotted as a percentage of it. Three monitors whose average
day peaks in the morning, the afternoon and late at night, which is the finding:
"Mumbai's worst hour" is not one hour.

Built on the window every one of the three reports over (1 Mar 2025 onward),
following scripts/03_diurnal_landuse.py: monitors that came online later would
otherwise average a different, monsoon-weighted slice of the year, which is a
seasonal difference wearing a spatial costume. Here it makes no difference -
all three span the same months and the peak hours are identical either way -
but the window is what the rest of the repo reports on.

Colours are the project's categorical slots 1-3, which clear the contrast and
colour-vision separation gates as a set. There is no separate legend box: each
curve is named at its own peak, next to a dot in its colour, which is a
stronger tie than a legend and costs no width on a phone screen. Nothing here
depends on telling the colours apart.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patheffects import withStroke

ROOT = Path(__file__).resolve().parent.parent
CLEAN_PATH = ROOT / "data" / "processed" / "pm25_clean.parquet"
OUT_PATH = ROOT / "reports" / "figures" / "diurnal_peak_hours.png"
TZ = "Asia/Kolkata"

# site in the data -> how it should read on the chart
SERIES = [
    ("Powai, Mumbai", "Powai", "#2a78d6"),
    ("Kurla, Mumbai", "Kurla", "#eb6834"),
    ("Malad West, Mumbai", "Malad West", "#1baf7a"),
]

INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_FAINT = "#8c8a84"
SURFACE = "#fcfcfb"
GRID = "#e4e3df"

# Hand-wrapped, not auto-wrapped: the break is part of the layout, and
# matplotlib's own wrapping does not reserve the height it then uses.
TITLE = "Mumbai's dirtiest hour is not\nthe same hour everywhere"
SUBTITLE = ("Average PM2.5 by hour of day at three monitors. Each curve is "
            "shown as a\npercentage of that area's own daily average, so the "
            "shapes compare directly.")
SOURCE = ("Hourly PM2.5 via OpenAQ, from CPCB / MPCB / IITM monitors. "
          "1 Mar 2025 - 27 Aug 2026.\nThree of 39 monitors in the region; "
          "times are IST.")


def hour_profiles():
    """Mean PM2.5 by hour of day for each series, on the common window."""
    c = pd.read_parquet(CLEAN_PATH)
    c["ist"] = c["timestamp"].dt.tz_convert(TZ)
    c["hour"] = c["ist"].dt.hour

    sites = [s for s, _, _ in SERIES]
    missing = set(sites) - set(c["site"].unique())
    if missing:
        sys.exit(f"sites not in {CLEAN_PATH.name}: {sorted(missing)}")

    d = c[c["site"].isin(sites)]
    start = d.groupby("site")["ist"].min().max()
    end = d.groupby("site")["ist"].max().min()
    d = d[(d["ist"] >= start) & (d["ist"] <= end)]

    out = {}
    for site, label, colour in SERIES:
        g = d[d["site"] == site].groupby("hour")["value"]
        mean, count = g.mean(), g.size()
        if len(mean) != 24:
            sys.exit(f"{site}: only {len(mean)} of 24 hours present")
        out[label] = {"pct": 100 * mean.to_numpy() / mean.mean(),
                      "colour": colour, "n": int(count.sum()),
                      "min_per_hour": int(count.min())}
    return out, start, end


def clock(hour):
    """0 -> "12 am", 15 -> "3 pm" - how a reader says the hour."""
    suffix = "am" if hour < 12 else "pm"
    h = hour % 12 or 12
    return f"{h} {suffix}"


def draw(profiles):
    fig = plt.figure(figsize=(5.6, 6.0), dpi=200, facecolor=SURFACE)
    # Title block, plot, source line. The plot keeps a right-hand margin inside
    # the axes (x runs past 23) so the last curve's label has somewhere to go
    # that is not on top of another curve.
    ax = fig.add_axes([0.115, 0.145, 0.855, 0.555])
    ax.set_facecolor(SURFACE)

    fig.text(0.04, 0.975, TITLE, color=INK, fontsize=15.5, fontweight="bold",
             va="top", ha="left", linespacing=1.25)
    fig.text(0.04, 0.862, SUBTITLE, color=INK_SOFT, fontsize=9.2, va="top",
             ha="left", linespacing=1.5)

    hours = np.arange(24)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    # The axis is the day; the right-hand margin is not part of it.
    ax.spines["bottom"].set_bounds(0, 23)

    ax.set_ylim(68, 140)
    ax.set_xlim(-0.8, 27.6)
    ax.set_yticks([80, 100, 120])
    ax.set_yticklabels(["80%", "100%", "120%"], fontsize=9.5, color=INK_SOFT)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_xticklabels([clock(0), clock(6), "noon", clock(18), clock(23)],
                       fontsize=9.5, color=INK_SOFT)
    ax.tick_params(length=0, pad=6)
    # Gridlines at 80 and 120 only, drawn by hand: 100 carries the dashed
    # reference line below, and a solid gridline underneath it would double it.
    for y in (80, 120):
        ax.axhline(y, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # The 100% line is what every curve is measured against, so it is drawn as
    # a reference rather than left as one gridline among three. Its label sits
    # out in the margin, clear of the curves.
    ax.axhline(100, color=INK_FAINT, linewidth=0.9, linestyle=(0, (4, 3)),
               zorder=1)
    ax.text(23.6, 100, "that area's\ndaily average", fontsize=8.2,
            color=INK_FAINT, ha="left", va="center", linespacing=1.4)

    # Where each label sits relative to its peak dot, in data units. Above the
    # dot for the two peaks inside the day. The 11 pm one is right-aligned and
    # anchored just past the end of the day, so it opens leftwards from beside
    # its own dot: left-aligned there it runs off the figure, and moved further
    # left it starts to read as belonging to the 3 pm curve.
    offsets = {"Powai": (0.0, 4.5, "center", "bottom"),
               "Kurla": (-0.5, 4.5, "center", "bottom"),
               "Malad West": (1.5, 3.0, "right", "bottom")}

    for label, series in profiles.items():
        pct, colour = series["pct"], series["colour"]
        ax.plot(hours, pct, color=colour, linewidth=2.4, solid_capstyle="round",
                zorder=3)
        peak = int(np.argmax(pct))
        # Surface ring on the marker so it stays legible where curves cross.
        ax.plot([peak], [pct[peak]], marker="o", markersize=8.5,
                markerfacecolor=colour, markeredgecolor=SURFACE,
                markeredgewidth=2.0, zorder=5)
        dx, dy, ha, va = offsets[label]
        ax.text(peak + dx, pct[peak] + dy,
                f"{label}\npeaks at {clock(peak)}", fontsize=10.2, color=INK,
                ha=ha, va=va, linespacing=1.35, zorder=6, fontweight="bold",
                path_effects=[withStroke(linewidth=3.0, foreground=SURFACE)])

    fig.text(0.04, 0.072, SOURCE, color=INK_FAINT, fontsize=7.8, va="top",
             ha="left", linespacing=1.5)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, facecolor=SURFACE)
    plt.close(fig)


def main():
    profiles, start, end = hour_profiles()
    draw(profiles)
    print(f"window {start:%Y-%m-%d} .. {end:%Y-%m-%d}")
    for label, s in profiles.items():
        pct = s["pct"]
        print(f"  {label:12s} peak {clock(int(np.argmax(pct))):>6s}  "
              f"trough {clock(int(np.argmin(pct))):>6s}  "
              f"swing {pct.max() - pct.min():4.0f}% of its mean  "
              f"n={s['n']:,} (>={s['min_per_hour']} per hour)")
    size = OUT_PATH.stat().st_size
    print(f"wrote {OUT_PATH.relative_to(ROOT)} ({size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
