import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt

# =============================
# 1) READ IMAGE FROM URL
# =============================
image_url = input("Paste image URL: ").strip('"')
response = requests.get(image_url)

image_array = np.asarray(bytearray(response.content), dtype=np.uint8)

image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

if image is None:
    raise ValueError("Image not loaded.")

h, w = image.shape[:2]

if h != w:
    raise ValueError("Image must be square.")

if len(image.shape) == 2:
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

channels = cv2.split(image)

encrypted_channels = []
decrypted_channels = []

# =============================
# PROCESS EACH CHANNEL
# =============================
for channel_data in channels:

    # =============================
    # 2) Arnold Cat Map
    # =============================
    def acm(img, iterations):
        N = img.shape[0]
        scrambled = img.copy()
        for _ in range(iterations):
            nxt = np.zeros_like(scrambled)
            for x in range(N):
                for y in range(N):
                    xn = (x + y) % N
                    yn = (x + 2*y) % N
                    nxt[xn, yn] = scrambled[x, y]
            scrambled = nxt
        return scrambled

    scrambled_img = acm(channel_data, 80)

    # =============================
    # 3) SVD Encryption
    # =============================
    class Encryption:
        def __init__(self, seed_u=1234, seed_s=5678, seed_v=9999):
            self.seed_u = seed_u
            self.seed_s = seed_s
            self.seed_v = seed_v

        def decomposition(self, img):
            imgf = img.astype(np.float64)
            U, S, VT = np.linalg.svd(imgf)
            return U, np.diag(S), VT

        def encrypt_u(self, U):
            rng = np.random.default_rng(self.seed_u)
            perm = rng.permutation(U.shape[1])
            return U[:, perm], perm

        def encrypt_s(self, S):
            rng = np.random.default_rng(self.seed_s)
            diag = np.diag(S)
            perm = rng.permutation(len(diag))
            return np.diag(diag[perm]), perm

        def encrypt_vt(self, VT):
            rng = np.random.default_rng(self.seed_v)
            perm = rng.permutation(VT.shape[0])
            return VT[perm, :], perm

    enc = Encryption()
    U, S, VT = enc.decomposition(scrambled_img)
    Ue, pu = enc.encrypt_u(U)
    Se, ps = enc.encrypt_s(S)
    VTe, pvt = enc.encrypt_vt(VT)

    svd_encrypted = Ue @ Se @ VTe
    svd_encrypted = np.mod(np.round(svd_encrypted), 256)
    svd_encrypted_u8 = svd_encrypted.astype(np.uint8)

    # =============================
    # 4) CHAOTIC FLOAT SEQUENCE
    # =============================
    def sine_map(n, x0=0.347654, r=0.99):
        x = x0
        for _ in range(1000):          # discard transient values
            x = r * np.sin(np.pi * x)
        arr = np.zeros(n)
        arr[0] = x
        for i in range(1, n):
            arr[i] = r * np.sin(np.pi * arr[i-1])
        return arr

    rows, cols = channel_data.shape
    total = rows * cols

    chaotic_float = sine_map(total)
    chaotic_uint8 = np.uint8(chaotic_float * 255)
    key_matrix = chaotic_uint8.reshape(rows, cols)

    # =============================
    # 5) XOR
    # =============================
    xor_output = np.bitwise_xor(svd_encrypted_u8, key_matrix)

    # =============================
    # 6) DIFFUSION LAYER (FIX for chi-square)
    # Each pixel depends on previous → spreads distribution uniformly
    # =============================
    def diffuse(img_flat, seed=7777):
        rng  = np.random.default_rng(seed)
        diff = rng.integers(0, 256, size=len(img_flat), dtype=np.uint8)
        out  = np.zeros_like(img_flat)
        out[0] = (int(img_flat[0]) + int(diff[0])) % 256
        for i in range(1, len(img_flat)):
            out[i] = int(img_flat[i]) ^ ((int(diff[i]) + int(out[i-1])) % 256)
        return out.astype(np.uint8)

    def inverse_diffuse(img_flat, seed=7777):
        rng  = np.random.default_rng(seed)
        diff = rng.integers(0, 256, size=len(img_flat), dtype=np.uint8)
        out  = np.zeros_like(img_flat)
        out[0] = (int(img_flat[0]) - int(diff[0])) % 256
        for i in range(1, len(img_flat)):
            out[i] = int(img_flat[i]) ^ ((int(diff[i]) + int(img_flat[i-1])) % 256)
        return out.astype(np.uint8)

    flat_xor   = xor_output.flatten()
    diffused   = diffuse(flat_xor)
    final_enc  = diffused.reshape(rows, cols)

    encrypted_channels.append(final_enc)

    # =============================
    # DECRYPTION
    # =============================
    # Step 1: inverse diffusion
    flat_enc      = final_enc.flatten()
    undiffused    = inverse_diffuse(flat_enc)
    undiffused_2d = undiffused.reshape(rows, cols)

    # Step 2: inverse XOR
    svd_decrypted_u8 = np.bitwise_xor(undiffused_2d, key_matrix)

    # Step 3: inverse SVD
    inv_pvt = np.argsort(pvt)
    VT_rec  = VTe[inv_pvt, :]

    inv_ps  = np.argsort(ps)
    S_diag  = np.diag(Se)
    S_rec   = np.diag(S_diag[inv_ps])

    inv_pu  = np.argsort(pu)
    U_rec   = Ue[:, inv_pu]

    scrambled_recovered = U_rec @ S_rec @ VT_rec
    scrambled_recovered = np.clip(scrambled_recovered, 0, 255)

    # Step 4: inverse ACM — must match encryption iterations
    def inverse_acm(img, iterations):
        N = img.shape[0]
        recovered = img.copy()
        for _ in range(iterations):
            nxt = np.zeros_like(recovered)
            for x in range(N):
                for y in range(N):
                    xn = (2*x - y) % N
                    yn = (-x + y) % N
                    nxt[xn, yn] = recovered[x, y]
            recovered = nxt
        return recovered

    final_decrypted = inverse_acm(scrambled_recovered, 80)
    final_decrypted = np.uint8(np.round(final_decrypted))

    decrypted_channels.append(final_decrypted)


# =============================
# MERGE CHANNELS
# =============================
encrypted_image = cv2.merge(encrypted_channels)
decrypted_image = cv2.merge(decrypted_channels)

cv2.imwrite("final_encrypted_xor.png", encrypted_image)

# =============================
# DISPLAY
# =============================
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original", fontsize=12, fontweight="bold")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(encrypted_image, cv2.COLOR_BGR2RGB))
plt.title("Encrypted", fontsize=12, fontweight="bold")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(decrypted_image, cv2.COLOR_BGR2RGB))
plt.title("Decrypted", fontsize=12, fontweight="bold")
plt.axis("off")

plt.tight_layout()
plt.show()

# =============================
# ENTROPY
# =============================
def entropy(img):
    total = img.size
    hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 256))
    probs = hist / total
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

original_entropy  = entropy(image)
encrypted_entropy = entropy(encrypted_image)
decrypted_entropy = entropy(decrypted_image)

print(f"Entropy of Original Image:  {original_entropy:.4f}")
print(f"Entropy of Encrypted Image: {encrypted_entropy:.4f}")
print(f"Entropy of Decrypted Image: {decrypted_entropy:.4f}")




# =============================
# HISTOGRAM ANALYSIS
# =============================

import matplotlib.pyplot as plt
import numpy as np
import cv2

image_name = "Image 1"

# Check if 'image' and 'encrypted_image' are defined from previous cells.
# If not, create placeholder variables to allow the cell to run without NameError.
# For meaningful results, ensure cell IShBFVAzGSkg is executed first.
try:
    image # Attempt to access 'image' to check if it's defined
    encrypted_image # Attempt to access 'encrypted_image'
except NameError:
    print("Warning: 'image' or 'encrypted_image' not found. Creating placeholder data. Please run the image encryption cell (IShBFVAzGSkg) above for actual results.")
    # Create dummy images (e.g., a 256x256 black image) for demonstration/error prevention
    dummy_size = 256
    image = np.zeros((dummy_size, dummy_size, 3), dtype=np.uint8)
    encrypted_image = np.zeros((dummy_size, dummy_size, 3), dtype=np.uint8)


# Convert images BGR → RGB
orig_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
enc_rgb  = cv2.cvtColor(encrypted_image, cv2.COLOR_BGR2RGB)

# Create figure layout
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(f"Histogram Analysis — {image_name}", fontsize=15, fontweight="bold")

# -----------------------------
# IMAGE PREVIEWS
# -----------------------------

axes[0, 0].imshow(orig_rgb)
axes[0, 0].set_title("Original Image", fontweight="bold")
axes[0, 0].axis("off")

axes[0, 1].imshow(enc_rgb)
axes[0, 1].set_title("Encrypted Image", fontweight="bold")
axes[0, 1].axis("off")

# Channel colors
channel_colors = [("#e74c3c","R"), ("#2ecc71","G"), ("#3498db","B")]

# -----------------------------
# ORIGINAL HISTOGRAM
# -----------------------------

for ch, (color, label) in enumerate(channel_colors):
    axes[1,0].hist(orig_rgb[:,:,ch].ravel(),
                   bins=256,
                   range=(0,256),
                   color=color,
                   alpha=0.6,
                   label=label)

axes[1,0].set_title("Original Histogram", fontweight="bold")
axes[1,0].set_xlim(0,255)
axes[1,0].set_xlabel("Pixel Value")
axes[1,0].set_ylabel("Frequency")
axes[1,0].legend()
axes[1,0].spines[["top","right"]].set_visible(False)

# -----------------------------
# ENCRYPTED HISTOGRAM
# -----------------------------

for ch, (color, label) in enumerate(channel_colors):
    axes[1,1].hist(enc_rgb[:,:,ch].ravel(),
                   bins=256,
                   range=(0,256),
                   color=color,
                   alpha=0.6,
                   label=label)

# Compute mean frequency
total_pixels = enc_rgb.shape[0] * enc_rgb.shape[1]
mean_freq = total_pixels / 256



axes[1,1].set_title("Encrypted Histogram", fontweight="bold")
axes[1,1].set_xlim(0,255)
axes[1,1].set_ylim(0,800)   # Y-axis forced to 800
axes[1,1].set_xlabel("Pixel Value")
axes[1,1].set_ylabel("Frequency")
axes[1,1].spines[["top","right"]].set_visible(False)

# -----------------------------
# SAVE & SHOW
# -----------------------------

plt.tight_layout()
plt.savefig(f"{image_name.replace(' ','_')}_histogram.png",
            dpi=150,
            bbox_inches="tight")

plt.show()

print(f"Saved: {image_name.replace(' ','_')}_histogram.png")

# =============================
# CHI-SQUARE ANALYSIS — FIXED
# =============================

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chisquare

image_name = "Image 1"

orig_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
enc_rgb  = cv2.cvtColor(encrypted_image, cv2.COLOR_BGR2RGB)

channel_names  = ["Red", "Green", "Blue"]
channel_colors = ["#e74c3c", "#2ecc71", "#3498db"]

def chi_square(img_rgb):
    results = []
    for ch in range(3):
        observed, _ = np.histogram(img_rgb[:, :, ch].ravel(), bins=256, range=(0, 256))
        expected = np.full(256, observed.sum() / 256)
        chi2, p = chisquare(observed, f_exp=expected)
        results.append((chi2, p))
    return results

orig_chi = chi_square(orig_rgb)
enc_chi  = chi_square(enc_rgb)

print(f"\n{'─'*55}")
print(f"  Chi-Square Analysis — {image_name}")
print(f"{'─'*55}")
print(f"  {'Channel':<10} {'Original χ²':>15} {'Encrypted χ²':>15}")
print(f"{'─'*55}")
for i, ch in enumerate(channel_names):
    print(f"  {ch:<10} {orig_chi[i][0]:>15.2f} {enc_chi[i][0]:>15.2f}")
print(f"{'─'*55}")
print(f"  Ideal encrypted χ² ≈ 255–280\n")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(f"Chi-Square Analysis — {image_name}", fontsize=15, fontweight="bold")

axes[0, 0].imshow(orig_rgb)
axes[0, 0].set_title("Original Image", fontsize=12, fontweight="bold")
axes[0, 0].axis("off")

axes[0, 1].imshow(enc_rgb)
axes[0, 1].set_title("Encrypted Image", fontsize=12, fontweight="bold")
axes[0, 1].axis("off")

# NO clipping — real values only
orig_vals = [orig_chi[i][0] for i in range(3)]
enc_vals  = [enc_chi[i][0] for i in range(3)]
y_max     = max(max(orig_vals), max(enc_vals)) * 1.15

x     = np.arange(3)
bar_w = 0.35

bars1 = axes[1, 0].bar(x - bar_w/2, orig_vals, bar_w,
                        color=channel_colors, alpha=0.7, label="Original")
bars2 = axes[1, 0].bar(x + bar_w/2, enc_vals,  bar_w,
                        color=channel_colors, edgecolor="black", linewidth=0.8, label="Encrypted")


axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(channel_names)
axes[1, 0].set_ylabel("χ² Value", fontsize=10)
axes[1, 0].set_ylim(0, y_max)
axes[1, 0].set_title("Chi-Square per Channel", fontsize=12, fontweight="bold")
axes[1, 0].spines[["top", "right"]].set_visible(False)

for bar in bars1:
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                    f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=7)
for bar in bars2:
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                    f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=7)

axes[1, 1].axis("off")
table_data = [["Channel", "Original χ²", "Encrypted χ²"]]
for i, ch in enumerate(channel_names):
    table_data.append([ch, f"{orig_chi[i][0]:.2f}", f"{enc_chi[i][0]:.2f}"])


table = axes[1, 1].table(cellText=table_data[1:], colLabels=table_data[0],
                          loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2.0)
for j in range(3):
    table[0, j].set_facecolor("#2c3e50")
    table[0, j].set_text_props(color="white", fontweight="bold")


axes[1, 1].set_title("Chi-Square Summary Table", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{image_name.replace(' ', '_')}_chisquare.png", dpi=150, bbox_inches="tight")
plt.show()
print(f" Saved: {image_name.replace(' ', '_')}_chisquare.png")

# =====================================================
# NPCR & UACI — DIFFERENTIAL ATTACK TEST
# Change 1 pixel in ORIGINAL → re-encrypt → compare
# encrypted_image vs modified_encrypted_image
# (paste this after your existing code)
# =====================================================

import numpy as np
import matplotlib.pyplot as plt

# Change this label each time
image_name = "Image 1"

# ── Step 1: Copy original & change exactly 1 pixel ───────────
modified_image = image.copy()
cy, cx = image.shape[0] // 2, image.shape[1] // 2  # center pixel
modified_image[cy, cx] = (modified_image[cy, cx].astype(np.int16) + 1) % 256

print(f"  Original pixel at ({cy},{cx}): {image[cy, cx]}")
print(f"  Modified pixel at ({cy},{cx}): {modified_image[cy, cx]}")

# ── Step 2: Re-encrypt the modified original ─────────────────
mod_channels = cv2.split(modified_image)
mod_encrypted_channels = []

for channel_data in mod_channels:

    def acm(img, iterations):
        N = img.shape[0]
        scrambled = img.copy()
        for _ in range(iterations):
            nxt = np.zeros_like(scrambled)
            for x in range(N):
                for y in range(N):
                    xn = (x + y) % N
                    yn = (x + 2 * y) % N
                    nxt[xn, yn] = scrambled[x, y]
            scrambled = nxt
        return scrambled

    scrambled_img = acm(channel_data, 80)

    class Encryption:
        def __init__(self, seed_u=1234, seed_s=5678, seed_v=9999):
            self.seed_u = seed_u
            self.seed_s = seed_s
            self.seed_v = seed_v

        def decomposition(self, img):
            imgf = img.astype(np.float64)
            U, S, VT = np.linalg.svd(imgf)
            return U, np.diag(S), VT

        def encrypt_u(self, U):
            rng = np.random.default_rng(self.seed_u)
            perm = rng.permutation(U.shape[1])
            return U[:, perm], perm

        def encrypt_s(self, S):
            rng = np.random.default_rng(self.seed_s)
            diag = np.diag(S)
            perm = rng.permutation(len(diag))
            return np.diag(diag[perm]), perm

        def encrypt_vt(self, VT):
            rng = np.random.default_rng(self.seed_v)
            perm = rng.permutation(VT.shape[0])
            return VT[perm, :], perm

    enc = Encryption()
    U, S, VT = enc.decomposition(scrambled_img)
    Ue, pu  = enc.encrypt_u(U)
    Se, ps  = enc.encrypt_s(S)
    VTe, pvt = enc.encrypt_vt(VT)

    svd_encrypted   = Ue @ Se @ VTe
    svd_encrypted = np.mod(np.round(svd_encrypted), 256)
    svd_encrypted_u8 = svd_encrypted.astype(np.uint8)

    def sine_map(n, x0=0.102341, r=0.965):
        arr = np.zeros(n)
        arr[0] = x0
        for i in range(1, n):
            arr[i] = r * np.sin(np.pi * arr[i - 1])
        return arr

    rows, cols = channel_data.shape
    chaotic_float = sine_map(rows * cols)
    key_matrix    = np.uint8(chaotic_float * 255).reshape(rows, cols)

    xor_output = np.bitwise_xor(svd_encrypted_u8, key_matrix)
    mod_encrypted_channels.append(xor_output)

modified_encrypted_image = cv2.merge(mod_encrypted_channels)

# ── Step 3: Compute NPCR & UACI ──────────────────────────────
# Comparing: encrypted_image (from original)
#        vs: modified_encrypted_image (from 1-pixel-changed original)

IDEAL_NPCR = 99.6094
IDEAL_UACI = 33.4635

def compute_npcr_uaci(img1, img2):
    channel_names = ["Blue", "Green", "Red"]
    results = {}
    for ch in range(3):
        c1 = img1[:, :, ch].astype(np.float64)
        c2 = img2[:, :, ch].astype(np.float64)
        D    = (c1 != c2).astype(np.float64)
        npcr = (D.sum() / D.size) * 100
        uaci = (np.abs(c1 - c2).sum() / (255 * D.size)) * 100
        results[channel_names[ch]] = {"npcr": npcr, "uaci": uaci}

    all1  = img1.astype(np.float64)
    all2  = img2.astype(np.float64)
    D_all = (all1 != all2).astype(np.float64)
    results["Overall"] = {
        "npcr": (D_all.sum() / D_all.size) * 100,
        "uaci": (np.abs(all1 - all2).sum() / (255 * D_all.size)) * 100
    }
    return results

results = compute_npcr_uaci(encrypted_image, modified_encrypted_image)

# ── Print ─────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print(f"  NPCR & UACI Differential Test — {image_name}")
#print(f"  (1 pixel changed in original before encryption)")
print(f"{'─'*60}")
print(f"  {'Channel':<10} {'NPCR (%)':>12} {'UACI (%)':>12}")
print(f"{'─'*60}")
for ch, vals in results.items():
    print(f"  {ch:<10} {vals['npcr']:>12.4f} {vals['uaci']:>12.4f}")
print(f"{'─'*60}")
print(f"  {'Ideal':<10} {IDEAL_NPCR:>12.4f} {IDEAL_UACI:>12.4f}\n")

# ── Plot ──────────────────────────────────────────────────────
channels    = list(results.keys())
npcr_vals   = [results[ch]["npcr"] for ch in channels]
uaci_vals   = [results[ch]["uaci"] for ch in channels]
bar_colors  = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(f"NPCR & UACI Differential Test — {image_name}",
             fontsize=13, fontweight="bold")

# Left: side-by-side encrypted images
orig_enc_rgb = cv2.cvtColor(encrypted_image, cv2.COLOR_BGR2RGB)
mod_enc_rgb  = cv2.cvtColor(modified_encrypted_image, cv2.COLOR_BGR2RGB)
combined     = np.concatenate([orig_enc_rgb, mod_enc_rgb], axis=1)
axes[0].imshow(combined)
axes[0].set_title("Encrypted (original)  |  Encrypted (modified)", fontsize=9, fontweight="bold")
axes[0].axis("off")

# Middle: NPCR bars
bars1 = axes[1].bar(channels, npcr_vals, color=bar_colors, alpha=0.85,
                    edgecolor="black", linewidth=0.6)

axes[1].set_title("NPCR per Channel (%)", fontsize=12, fontweight="bold")
axes[1].set_ylabel("NPCR (%)", fontsize=10)
axes[1].set_ylim(0, 105)
axes[1].spines[["top", "right"]].set_visible(False)
for bar, val in zip(bars1, npcr_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val:.2f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

# Right: UACI bars
bars2 = axes[2].bar(channels, uaci_vals, color=bar_colors, alpha=0.85,
                    edgecolor="black", linewidth=0.6)

axes[2].set_title("UACI per Channel (%)", fontsize=12, fontweight="bold")
axes[2].set_ylabel("UACI (%)", fontsize=10)
axes[2].set_ylim(0, 50)
axes[2].spines[["top", "right"]].set_visible(False)
for bar, val in zip(bars2, uaci_vals):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{val:.2f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{image_name.replace(' ', '_')}_differential_npcr_uaci.png", dpi=150, bbox_inches="tight")
plt.show()
print(f" Saved: {image_name.replace(' ', '_')}_differential_npcr_uaci.png")


# =====================================================
# SSIM — Original vs Encrypted
# =====================================================

from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Change this label each time
image_name = "Image 1"

# Convert BGR → RGB
orig_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
enc_rgb  = cv2.cvtColor(encrypted_image, cv2.COLOR_BGR2RGB)

# ── Compute SSIM per channel + overall ───────────────────────
channel_names = ["Red", "Green", "Blue"]
ssim_values = []

for ch in range(3):
    val = ssim(orig_rgb[:, :, ch], enc_rgb[:, :, ch], data_range=255)
    ssim_values.append(val)

overall_ssim = ssim(orig_rgb, enc_rgb, data_range=255, channel_axis=2)

# ── Print ─────────────────────────────────────────────────────
print(f"\n{'─'*45}")
print(f"  SSIM — Original vs Encrypted — {image_name}")
print(f"{'─'*45}")
for ch, val in zip(channel_names, ssim_values):
    print(f"  {ch:<10} SSIM: {val:.6f}")
print(f"{'─'*45}")
print(f"  {'Overall':<10} SSIM: {overall_ssim:.6f}")
print(f"{'─'*45}")
print(f"  (Closer to 0 → better encryption, 1 = identical)\n")

# ── Plot ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f"SSIM — Original vs Encrypted — {image_name}", fontsize=13, fontweight="bold")

# Left: images side by side
combined = np.concatenate([orig_rgb, enc_rgb], axis=1)
axes[0].imshow(combined)
axes[0].set_title("Original  |  Encrypted", fontsize=10, fontweight="bold")
axes[0].axis("off")

# Middle: SSIM bar chart per channel
bar_colors = ["#e74c3c", "#2ecc71", "#3498db"]
bars = axes[1].bar(channel_names, ssim_values, color=bar_colors, alpha=0.85,
                   edgecolor="black", linewidth=0.6)

axes[1].set_title("SSIM per Channel", fontsize=12, fontweight="bold")
axes[1].set_ylabel("SSIM Value", fontsize=10)
axes[1].set_ylim(-0.1, 1.05)
axes[1].spines[["top", "right"]].set_visible(False)
for bar, val in zip(bars, ssim_values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

# Right: summary table
axes[2].axis("off")
table_data = [["Channel", "SSIM"]]
for ch, val in zip(channel_names, ssim_values):
    table_data.append([ch, f"{val:.6f}"])
table_data.append(["Overall", f"{overall_ssim:.6f}"])

table = axes[2].table(cellText=table_data[1:], colLabels=table_data[0],
                      loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2.2)
for j in range(2):
    table[0, j].set_facecolor("#2c3e50")
    table[0, j].set_text_props(color="white", fontweight="bold")
# Highlight overall row
for j in range(2):
    table[len(channel_names)+1, j].set_facecolor("#ecf0f1")
    table[len(channel_names)+1, j].set_text_props(fontweight="bold")

axes[2].set_title("SSIM Summary Table", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{image_name.replace(' ', '_')}_ssim_orig_vs_enc.png", dpi=150, bbox_inches="tight")
plt.show()
print(f" Saved: {image_name.replace(' ', '_')}_ssim_orig_vs_enc.png")


# =====================================================
# SSIM — Original vs Decrypted
# =====================================================

from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Change this label each time
image_name = "Image 1"

# Convert BGR → RGB
orig_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
dec_rgb  = cv2.cvtColor(decrypted_image, cv2.COLOR_BGR2RGB)

# ── Compute SSIM per channel + overall ───────────────────────
channel_names = ["Red", "Green", "Blue"]
ssim_values = []

for ch in range(3):
    val = ssim(orig_rgb[:, :, ch], dec_rgb[:, :, ch], data_range=255)
    ssim_values.append(val)

overall_ssim = ssim(orig_rgb, dec_rgb, data_range=255, channel_axis=2)

# ── Print ─────────────────────────────────────────────────────
print(f"\n{'─'*45}")
print(f"  SSIM — Original vs Decrypted — {image_name}")
print(f"{'─'*45}")
for ch, val in zip(channel_names, ssim_values):
    print(f"  {ch:<10} SSIM: {val:.6f}")
print(f"{'─'*45}")
print(f"  {'Overall':<10} SSIM: {overall_ssim:.6f}")
print(f"{'─'*45}")
print(f"  (Closer to 1 → perfect decryption, 0 = no similarity)\n")

# ── Plot ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f"SSIM — Original vs Decrypted — {image_name}", fontsize=13, fontweight="bold")

# Left: images side by side
combined = np.concatenate([orig_rgb, dec_rgb], axis=1)
axes[0].imshow(combined)
axes[0].set_title("Original  |  Decrypted", fontsize=10, fontweight="bold")
axes[0].axis("off")

# Middle: SSIM bar chart per channel
bar_colors = ["#e74c3c", "#2ecc71", "#3498db"]
bars = axes[1].bar(channel_names, ssim_values, color=bar_colors, alpha=0.85,
                   edgecolor="black", linewidth=0.6)

axes[1].set_title("SSIM per Channel", fontsize=12, fontweight="bold")
axes[1].set_ylabel("SSIM Value", fontsize=10)
axes[1].set_ylim(-0.1, 1.1)
axes[1].spines[["top", "right"]].set_visible(False)
for bar, val in zip(bars, ssim_values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

# Right: summary table
axes[2].axis("off")
table_data = [["Channel", "SSIM"]]
for ch, val in zip(channel_names, ssim_values):
    table_data.append([ch, f"{val:.6f}"])
table_data.append(["Overall", f"{overall_ssim:.6f}"])

table = axes[2].table(cellText=table_data[1:], colLabels=table_data[0],
                      loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2.2)
for j in range(2):
    table[0, j].set_facecolor("#2c3e50")
    table[0, j].set_text_props(color="white", fontweight="bold")
# Highlight overall row
for j in range(2):
    table[len(channel_names)+1, j].set_facecolor("#ecf0f1")
    table[len(channel_names)+1, j].set_text_props(fontweight="bold")

axes[2].set_title("SSIM Summary Table", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{image_name.replace(' ', '_')}_ssim_orig_vs_dec.png", dpi=150, bbox_inches="tight")
plt.show()
print(f" Saved: {image_name.replace(' ', '_')}_ssim_orig_vs_dec.png")


# =====================================================
# CORRELATION COEFFICIENT — H, V, D
# Original vs Encrypted — Scatter Plots + Table
# =====================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

# Change this label each time
image_name = "Image 1"

# ── Correlation coefficient formula ──────────────────────────
def correlation_coefficient(img, direction):
    """
    Compute correlation coefficient between adjacent pixels.
    direction: 'H' = horizontal, 'V' = vertical, 'D' = diagonal
    """
    if direction == 'H':
        x = img[:, :-1].flatten()
        y = img[:, 1:].flatten()
    elif direction == 'V':
        x = img[:-1, :].flatten()
        y = img[1:, :].flatten()
    elif direction == 'D':
        x = img[:-1, :-1].flatten()
        y = img[1:, 1:].flatten()

    # Sample 3000 random pairs for scatter plot visibility
    np.random.seed(42)
    idx = np.random.choice(len(x), size=min(3000, len(x)), replace=False)
    x_sample, y_sample = x[idx], y[idx]

    # Correlation coefficient
    cov  = np.cov(x.astype(np.float64), y.astype(np.float64))[0, 1]
    std1 = np.std(x.astype(np.float64))
    std2 = np.std(y.astype(np.float64))
    cc   = cov / (std1 * std2 + 1e-10)

    return cc, x_sample, y_sample

# ── Convert to grayscale for correlation ─────────────────────
orig_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
enc_gray  = cv2.cvtColor(encrypted_image, cv2.COLOR_BGR2GRAY)

directions = ['H', 'V', 'D']
dir_names  = {'H': 'Horizontal', 'V': 'Vertical', 'D': 'Diagonal'}
colors     = {'H': '#3498db', 'V': '#2ecc71', 'D': '#e74c3c'}

# Compute all coefficients and samples
orig_results = {}
enc_results  = {}

for d in directions:
    cc, xs, ys           = correlation_coefficient(orig_gray, d)
    orig_results[d]      = {'cc': cc, 'x': xs, 'y': ys}

    cc, xs, ys           = correlation_coefficient(enc_gray, d)
    enc_results[d]       = {'cc': cc, 'x': xs, 'y': ys}

# ── Print table ───────────────────────────────────────────────
print(f"\n{'─'*58}")
print(f"  Correlation Coefficients — {image_name}")
print(f"{'─'*58}")
print(f"  {'Direction':<14} {'Original':>14} {'Encrypted':>14}")
print(f"{'─'*58}")
for d in directions:
    print(f"  {dir_names[d]:<14} {orig_results[d]['cc']:>14.6f} {enc_results[d]['cc']:>14.6f}")
print(f"{'─'*58}")
print(f"  (Closer to 0 in encrypted → better encryption)\n")

# ── Plot: 3 rows (H, V, D) × 2 cols (Original, Encrypted) + table ──
fig = plt.figure(figsize=(16, 13))
fig.suptitle(f"Correlation Coefficient Scatter Plots — {image_name}",
             fontsize=14, fontweight="bold", y=1.01)

gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35,
                       height_ratios=[1, 1, 1, 0.6])

for row, d in enumerate(directions):
    # Original scatter
    ax_o = fig.add_subplot(gs[row, 0])
    ax_o.scatter(orig_results[d]['x'], orig_results[d]['y'],
                 s=1.5, alpha=0.4, color=colors[d])
    ax_o.set_title(f"Original — {dir_names[d]}  (r = {orig_results[d]['cc']:.4f})",
                   fontsize=10, fontweight="bold")
    ax_o.set_xlabel("Pixel (i)", fontsize=8)
    ax_o.set_ylabel("Pixel (i+1)", fontsize=8)
    ax_o.set_xlim(0, 255)
    ax_o.set_ylim(0, 255)
    ax_o.tick_params(labelsize=7)
    ax_o.spines[["top", "right"]].set_visible(False)

    # Encrypted scatter
    ax_e = fig.add_subplot(gs[row, 1])
    ax_e.scatter(enc_results[d]['x'], enc_results[d]['y'],
                 s=1.5, alpha=0.4, color="#7f8c8d")
    ax_e.set_title(f"Encrypted — {dir_names[d]}  (r = {enc_results[d]['cc']:.4f})",
                   fontsize=10, fontweight="bold")
    ax_e.set_xlabel("Pixel (i)", fontsize=8)
    ax_e.set_ylabel("Pixel (i+1)", fontsize=8)
    ax_e.set_xlim(0, 255)
    ax_e.set_ylim(0, 255)
    ax_e.tick_params(labelsize=7)
    ax_e.spines[["top", "right"]].set_visible(False)

# ── Summary table (bottom row, spans both columns) ───────────
ax_table = fig.add_subplot(gs[3, :])
ax_table.axis("off")

table_data = [
    [dir_names[d], f"{orig_results[d]['cc']:.6f}", f"{enc_results[d]['cc']:.6f}"]
    for d in directions
]
col_labels = ["Direction", "Original (r)", "Encrypted (r)"]

table = ax_table.table(cellText=table_data, colLabels=col_labels,
                       loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.0, 2.2)

# Style header
for j in range(3):
    table[0, j].set_facecolor("#2c3e50")
    table[0, j].set_text_props(color="white", fontweight="bold")

# Alternating row colors
row_colors = ["#ecf0f1", "#ffffff"]
for i in range(1, len(directions) + 1):
    for j in range(3):
        table[i, j].set_facecolor(row_colors[i % 2])

ax_table.set_title("Correlation Coefficient Summary", fontsize=12,
                   fontweight="bold", pad=12)

plt.tight_layout()
plt.savefig(f"{image_name.replace(' ', '_')}_correlation.png", dpi=150, bbox_inches="tight")
plt.show()
print(f" Saved: {image_name.replace(' ', '_')}_correlation.png")
