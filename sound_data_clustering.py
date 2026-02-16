import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
import hdbscan
from skimage.util import view_as_windows
import scipy.signal as sig
from scipy.ndimage import median_filter

import umap

# ====== Params ======
audio_path = 'YOUR_AUDIO_FILE.wav'


#const parameters
RE_SR = 16000

FMIN, FMAX = 250, 2000
NFFT_MULTI = 2000//FMAX 
N_FFT = 2048*1*NFFT_MULTI
HOP = N_FFT // 4

DB_FLOOR = -32.0

PATCH_SIZE = 6
PATCH_FRAMES = PATCH_SIZE
PATCH_BINS = PATCH_SIZE
STEP_T = PATCH_SIZE
STEP_F = PATCH_SIZE
N_PCA = 30

UM_N_COMPONENTS = 40
UM_N_NEIGHBORS = 10
UM_MIN_DIST = 0.2

MEDIAN_F_SIZE = 1
MIN_CLUSTER_SIZE = 80
MIN_SAMPLES = 80

CMAP_SPEC = "gray"

# definition of notch filter band. Frequency, Strength
NOTCH = [(50,60.0),(150,200.0),(250,60.0),(350,300.0),(450,320.0),(750,420.0),(1050,520.0),(1350,520.0),(1650,520.0)]

# time segment (seconds)
y_sub_start_sec = 30*6 + 30
y_sub_end_sec = y_sub_start_sec + 60

# ===== Load and trim (no resample) =====
y_raw, sr = librosa.load(audio_path, sr=RE_SR, res_type='soxr_vhq')
#y_raw, sr = librosa.load(audio_path, sr=RE_SR, res_type='kaiser_best')
y_sub_start_frame = int(y_sub_start_sec * sr)
y_sub_end_frame = int(y_sub_end_sec * sr)
y = y_raw[y_sub_start_frame:y_sub_end_frame].astype(float)

# ===== Compute original limited STFT (for plotting original) =====
S_orig_complex = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
S_orig = np.abs(S_orig_complex)
freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
mask = (freqs >= FMIN) & (freqs <= FMAX)
band_freqs = freqs[mask]

# ===== Build band-limited STFT (crop rows) and reconstruct time-domain band signal =====
# Two ways: (A) crop rows and work on S_band for clustering, (B) create full STFT with zeros outside band then istft.
# We'll create full masked STFT for accurate inverse transform, and also S_band (cropped) for patching.
S_full_masked = np.zeros_like(S_orig_complex)
S_full_masked[mask, :] = S_orig_complex[mask, :]
S_band = np.abs(S_full_masked[mask, :])       # shape: (n_band_bins, n_frames)

# inverse STFT of band-limited signal -> then notch filter (time-domain)
y_band = librosa.istft(S_full_masked, hop_length=HOP, length=len(y))

def design_notch(fs, f0, Q):
    w0 = f0 / (fs / 2)
    b, a = sig.iirnotch(w0, Q)
    return b, a

y_filt = y_band.copy()
for f0, Q in NOTCH:
    b, a = design_notch(sr, f0, Q)
    y_filt = sig.filtfilt(b, a, y_filt)

# normalize (optional)
y_filt = y_filt / np.max(np.abs(y_filt) + 1e-12) * 0.99

# STFT of notch-filtered band signal (for plotting & clustering)
S_filt_complex = librosa.stft(y_filt, n_fft=N_FFT, hop_length=HOP)
S_filt = np.abs(S_filt_complex)
# Crop to band for clustering/patching
S_filt_band = S_filt[mask, :]

# ===== Convert to dB / normalize for patch extraction =====
S_db_band_fulldb = librosa.amplitude_to_db(S_filt_band, ref=np.max, top_db=None)

##get max/min db in original
peak = S_filt_band.max()
S_db_rel = 20 * np.log10(S_filt_band / (peak + 1e-12))
global_min_db = float(np.min(S_db_rel))
global_max_db = float(np.max(S_db_rel))  # should be ~0.0
print("global_min_db =", global_min_db, "dB")
print("global_max_db =", global_max_db, "dB")

this_db_floor = global_min_db*0.58
# apply floor: everything below DB_FLOOR becomes DB_FLOOR
S_db_band= np.maximum(S_db_band_fulldb, this_db_floor)

S_min, S_max = S_db_band.min(), S_db_band.max()
S_norm = (S_db_band - S_min) / (S_max - S_min + 1e-9)

# ===== Patch extraction (on band-limited normalized dB spectrogram) =====
pad_f = PATCH_BINS // 2
pad_t = PATCH_FRAMES // 2
S_padded = np.pad(S_norm, ((pad_f, pad_f), (pad_t, pad_t)), mode='reflect')
win_shape = (PATCH_BINS, PATCH_FRAMES)
patches = view_as_windows(S_padded, win_shape, step=(STEP_F, STEP_T))
n_band_pos, n_time_pos = patches.shape[:2]
patches_reshaped = patches.reshape(-1, PATCH_BINS * PATCH_FRAMES)

X = patches_reshaped.astype(float)
X -= X.mean(axis=0, keepdims=True)
X /= (X.std(axis=0, keepdims=True) + 1e-9)

# optional PCA
pca = PCA(n_components=min(N_PCA, X.shape[1], X.shape[0]))
X_pca = pca.fit_transform(X)


# UMAP
um = umap.UMAP(n_components=UM_N_COMPONENTS, n_neighbors=UM_N_NEIGHBORS, min_dist=UM_MIN_DIST, random_state=42)
X_umap = um.fit_transform(X_pca)

# ===== HDBSCAN =====
n_samples = X_umap.shape[0]
min_size = min(max(2, MIN_CLUSTER_SIZE), max(2, n_samples))
if n_samples < MIN_CLUSTER_SIZE:
    min_size = max(2, int(0.05 * n_samples))

clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size, min_samples=MIN_SAMPLES, allow_single_cluster=False, cluster_selection_method='eom')
labels = clusterer.fit_predict(X_umap)

if np.all(labels == -1):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=max(2, int(0.02*n_samples)), min_samples=1)
    labels = clusterer.fit_predict(X_umap)
    print("couldn't find proper clusters")

# remap -1 to -1 (keep noise as -1) but ensure contiguous non-negative labels for colors later
unique = np.unique(labels)

# reshape to grid
labels_grid = labels.reshape(n_band_pos, n_time_pos)
#print("len(unique)",len(unique))
#print(unique)


# median smoothing on label grid (works with small integer labels; ignore -1 carefully)
# map -1 to a special value for smoothing, then map back
if MEDIAN_F_SIZE > 0:
    temp = labels_grid.copy()
    neg_mask = (temp == -1)
    temp[neg_mask] = temp.max() + 1
    temp = median_filter(temp, size=(MEDIAN_F_SIZE, MEDIAN_F_SIZE))
    temp[temp == temp.max()] = -1
    labels_grid = temp

    
# ===== Expand labels_grid to band spectrogram pixels (direct step-based assignment) =====
n_band_bins, n_frames = S_filt_band.shape
label_map_band = np.full((n_band_bins, n_frames), fill_value=-1, dtype=int)

for i in range(n_band_pos):
    row0 = i * STEP_F
    row1 = row0 + PATCH_BINS
    row1 = min(row1, n_band_bins)
    for j in range(n_time_pos):
        col0 = j * STEP_T
        col1 = col0 + PATCH_FRAMES
        col1 = min(col1, n_frames)
        label_map_band[row0:row1, col0:col1] = labels_grid[i, j]

# ===== Map band label map back to full-frequency spectrogram shape =====
n_bins_full = S_orig.shape[0]
label_map_full = np.full((n_bins_full, n_frames), fill_value=-1, dtype=int)
label_map_full[mask, :] = label_map_band


#### cluster statistics
time_axis = np.arange(S_filt_band.shape[1]) * (HOP / sr)

labels_flat = label_map_band.flatten()
unique_labels = np.unique(labels_flat)
unique_labels = unique_labels[unique_labels != -1]  # skip noise if desired

cluster_stats = {}
for lab in unique_labels:
    mask = (label_map_band == lab)   # boolean mask of pixels in this cluster
    if not np.any(mask):
        continue

    # Magnitudes (linear) and convert to dB if needed
    mags = S_filt_band[mask]                     # linear amplitude values
    mags_db = 20 * np.log10(mags + 1e-12)        # dB

    # Pixel coordinates
    rows, cols = np.nonzero(mask)                # row indices -> freq bins, col indices -> frames

    freqs_in_cluster = band_freqs[rows]
    times_in_cluster = time_axis[cols]

    stats = {
        "n_pixels": int(mask.sum()),
        "freq_min_hz": float(freqs_in_cluster.min()),
        "freq_max_hz": float(freqs_in_cluster.max()),
        "freq_mean_hz": float(freqs_in_cluster.mean()),
        "time_min_s": float(times_in_cluster.min()),
        "time_max_s": float(times_in_cluster.max()),
        "time_mean_s": float(times_in_cluster.mean()),
        "mag_min": float(mags.min()),
        "mag_max": float(mags.max()),
        "mag_mean": float(mags.mean()),
        "mag_min_db": float(mags_db.min()),
        "mag_max_db": float(mags_db.max()),
        "mag_mean_db": float(mags_db.mean()),
    }
    cluster_stats[int(lab)] = stats

# Example: print a summary
for lab, s in cluster_stats.items():
    #print(f"Cluster {lab}: pixels={s['n_pixels']}, freq {s['freq_min_hz']:.1f}-{s['freq_max_hz']:.1f} Hz (mean {s['freq_mean_hz']:.1f}), mag_db mean {s['mag_mean_db']:.1f} dB")
    #print("Cluter",lab, "freq min/max/mean:", int(s['freq_min_hz']),int(s['freq_max_hz']),int(s['freq_mean_hz']), "mag min/max/mean", s['mag_min'],s['mag_max'],s['mag_mean'])
    print(f"Cluster {lab}:, freq {s['freq_min_hz']:.1f}-{s['freq_max_hz']:.1f} {s['freq_mean_hz']:.1f}, mag {s['mag_min']:.1f}/{s['mag_max']:.1f}/{s['mag_mean']:.1f}")

###HDBSCAN Attributes
# now access fit-produced attributes:
persistence = clusterer.cluster_persistence_        # numpy array
probs = getattr(clusterer, "probabilities_", None) # may be None if not computed
outlier = getattr(clusterer, "outlier_scores_", None)

# simple print
n_clusters_hdb = len(persistence)
print(f"HDBSCAN: {n_clusters_hdb} clusters (persistence length), labels shape {labels.shape}")
for lab in range(n_clusters_hdb):
    size = int((labels == lab).sum())
    pers = float(persistence[lab])
    mean_prob = float(probs[labels == lab].mean()) if (probs is not None and size>0) else None
    mean_outlier = float(outlier[labels == lab].mean()) if (outlier is not None and size>0) else None
    print(f"cluster {lab}: size={size}, persistence={pers:.4f}, mean_prob={mean_prob:.4f}, mean_outlier={mean_outlier:.4f}")


####END cluster statistics

# ===== Build RGB overlay (alpha where label != -1) =====
cmap = plt.get_cmap("tab10")
max_label = int(label_map_full.max()) if label_map_full.max() >= 0 else -1
n_clusters_found = max_label + 1 if max_label >= 0 else 0
cluster_colors = np.array([cmap(i % 10) for i in range(max(1, n_clusters_found))])[:, :3]

print("n_clusters_found",n_clusters_found)

# create RGBA image (freq x time x 4), default transparent
label_rgba = np.zeros((label_map_full.shape[0], label_map_full.shape[1], 4), dtype=float)
for lab in range(n_clusters_found):
    m = (label_map_full == lab)
    label_rgba[m, :3] = cluster_colors[lab]
    label_rgba[m, 3] = 1.0  # alpha for clusters

# flip rows so low freq is at bottom for imshow with origin='lower'
#label_rgba_flipped = label_rgba[::-1, :, :]
label_rgba_flipped = label_rgba

# ===== Compute dB spectrograms for plotting full-band originals =====
S_db_orig = librosa.amplitude_to_db(S_orig, ref=np.max, top_db=80.0)
S_db_filt_full = librosa.amplitude_to_db(S_filt, ref=np.max, top_db=80.0)


# ===== Plot: original, notch-filtered, notch-filtered + clusters =====
time_axis = np.arange(n_frames) * HOP / sr
extent = (time_axis[0], time_axis[-1] + HOP/sr, 0, sr/2)

fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
librosa.display.specshow(S_db_orig, sr=sr, hop_length=HOP, x_axis='time', y_axis='hz', ax=axs[0], cmap=CMAP_SPEC)
axs[0].set_ylim(band_freqs[0], band_freqs[-1])
axs[0].set_title("Original limited spectrogram")

librosa.display.specshow(S_db_filt_full, sr=sr, hop_length=HOP, x_axis='time', y_axis='hz', ax=axs[1], cmap=CMAP_SPEC)
axs[1].set_ylim(band_freqs[0], band_freqs[-1])
axs[1].set_title("Notch-filtered limited spectrogram")

librosa.display.specshow(S_db_filt_full, sr=sr, hop_length=HOP, x_axis='time', y_axis='hz', ax=axs[2], cmap=CMAP_SPEC)
# overlay clusters; origin='lower' so label_rgba_flipped aligns with specshow
axs[2].imshow(label_rgba_flipped, extent=extent, aspect='auto', origin='lower', zorder=10)
axs[2].set_ylim(band_freqs[0], band_freqs[-1])
axs[2].set_title("Notch-filtered with cluster overlay")



# Build legend entries for clusters (skip noise)
legend_patches = []
for lab in range(n_clusters_found):
    color = cluster_colors[lab]
    # convert to matplotlib-friendly tuple
    color_tuple = tuple(color)
    patch = mpatches.Patch(color=color_tuple, label=f"cluster {lab}")
    legend_patches.append(patch)

# Optionally place legend outside the plot on the right
axs[2].legend(handles=legend_patches, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title="Clusters", fontsize=8)

plt.tight_layout()
plt.show()
