"""Shared constants: bands, change classes, colours and training windows."""

BANDS = ["blue", "green", "red", "nir", "swir16", "swir22"]

CLASS_NAMES = ["No change", "Urban expansion", "Deforestation", "Flooding", "Land-use change"]
N_CLASSES = len(CLASS_NAMES)
# RGB 0-255 (class 0 is never drawn)
CLASS_COLORS = [(0, 0, 0), (230, 25, 75), (255, 150, 0), (0, 110, 255), (170, 60, 230)]

# Reflectance normalisation applied before the network
NORM_MEAN = 0.15
NORM_STD = 0.12

# (before window, after window, max cloud %) per preset AOI - used for weak-label training
# and as sensible defaults in the CLI / app. Windows are chosen in the same season so that
# phenology does not masquerade as change (flood is the deliberate exception: dry vs monsoon).
PRESET_WINDOWS = {
    "urban_bengaluru": ("2019-01-01/2019-03-31", "2026-01-01/2026-03-31", 30),
    "deforest_rondonia": ("2019-06-15/2019-09-15", "2025-06-15/2025-09-15", 30),
    "flood_assam": ("2025-01-01/2025-02-28", "2025-06-20/2025-07-31", 70),
    "landuse_coimbatore": ("2019-01-01/2019-03-31", "2026-01-01/2026-03-31", 30),
}
