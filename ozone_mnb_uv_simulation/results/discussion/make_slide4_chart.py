# Slide-4 chart: JOB-055/056 fixed-line scan, energy + directional derivative a(t)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PRIMARY = "#0E5E6F"; ACCENT = "#E8590C"; MUTED = "#6B7A87"; TEXT = "#1E293B"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})

t      = [0.0,   0.04,   0.045,  0.050,  0.055,  0.060]
dE_uEh = [0.0,   -1.635, -1.711, -1.725, -1.665, -1.517]     # x1e-7 Eh vs 055 centre (actually 1e-7 units: values are -1.635e-7 Eh)
a_uEhB = [-4.486,-2.054, -0.935,  0.420,  2.029,  3.907]     # x1e-6 Eh/Bohr

fig, ax1 = plt.subplots(figsize=(6.4, 4.4), dpi=200)
ax1.axvspan(0.045, 0.050, color=ACCENT, alpha=0.12, zorder=0)
ax1.plot(t, dE_uEh, "o-", color=PRIMARY, lw=2, ms=6, zorder=3)
ax1.set_xlabel("displacement along fixed direction q,  t  (Bohr)")
ax1.set_ylabel("E(t) − E(centre)   (10$^{-7}$ Eh)", color=PRIMARY)
ax1.tick_params(axis="y", labelcolor=PRIMARY)
ax1.set_ylim(-2.4, 0.5)
ax1.axhline(0, color=MUTED, lw=0.8, ls="--", alpha=0.6)

ax2 = ax1.twinx()
ax2.axhline(0, color=MUTED, lw=0.8, ls="--", alpha=0.6)
ax2.plot(t, a_uEhB, "s--", color=ACCENT, lw=1.8, ms=5.5, zorder=3)
ax2.set_ylabel("directional derivative a(t) = g·q   (10$^{-6}$ Eh/Bohr)", color=ACCENT)
ax2.tick_params(axis="y", labelcolor=ACCENT)
ax2.set_ylim(-5.4, 4.9)

# zero-crossing interpolation mark on a(t)
ax2.plot([0.0484], [0], marker="D", color=ACCENT, ms=8, mec="white", mew=1.2, zorder=4)
ax2.text(0.008, 3.5, "shaded band: a(t) sign-change interval 0.045 – 0.050 Bohr\n"
                     "interpolated zero ≈ t = 0.0484 (◇);  lowest sampled E at t = 0.050",
         fontsize=10, color=TEXT, ha="left", va="center",
         bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=MUTED, lw=0.7, alpha=0.95))

ax1.set_title("JOB-055/056 merged line scan (6 points, all SCF + full gradient)", fontsize=11.5, color=TEXT, pad=10)
ax1.grid(alpha=0.18, lw=0.5)
fig.tight_layout()
fig.savefig(r"E:\dft-service\ozone_mnb_uv_simulation\results\discussion\assets\scan055_056_slide.png",
            facecolor="white", bbox_inches="tight")
print("saved")
