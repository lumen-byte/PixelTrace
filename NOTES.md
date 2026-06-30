# PixelTrace — Spot the Fake Photo
## Approach Note

**Author:** Abhimanyu Singh | **Device:** Apple M-series MacBook (Apple Silicon GPU / MPS)

---

### Problem

Detect whether a photo is a **real photograph** or a **photo of a screen** (someone re-photographing a phone/laptop display instead of the real subject). This is a binary fraud-detection problem where the clue is subtle physical artifacts, not semantic content.

---

### Approach

I built a **hybrid forensic feature + CNN ensemble detector** that exploits the physical inevitability of screen artifacts:

**Why screens are detectable:**
1. **Moiré / pixel-grid patterns** — Any camera sampling a display's pixel grid creates periodic interference (moiré) visible as discrete peaks in the 2D FFT power spectrum. Real photos have a smooth, monotonically-decaying power spectrum.
2. **Noise floor suppression** — Screen glass + pixel optics reduce camera sensor noise. Real photos have characteristic sensor noise; recaptures have artificially smooth noise.
3. **Chromatic aberration anomalies** — Camera lens CA superimposed on display subpixel colour layout (RGB stripe / PenTile) creates anomalous colour fringing at edges.
4. **Colour gamut compression** — Screens cannot reproduce the full sRGB gamut; saturation statistics compress predictably.
5. **High-frequency energy redistribution** — Screen pixel grid creates energy concentrations in the high-frequency FFT bands.

**Feature pipeline (161 features total):**
- Moiré / FFT analysis: 9 physics-based spectral features
- Chromatic aberration: 8 features
- GLCM texture (LBP + co-occurrence matrix): 24 features
- HOG gradient orientation histogram: 17 features
- FFT radial + angular energy bins: 43 features
- Colour statistics (HSV + RGB correlations + skewness): 16 features
- Edge, noise, sharpness, geometry, reflection: 12 features
- MobileNetV3 CNN embeddings (→ TruncatedSVD 128 components): 128 features

**Classifier:** Soft-Voting Ensemble of Calibrated SVM (C=100, RBF) + XGBoost (n=800, depth=3) + MLP (256-128-64). Model selection by 5-fold stratified CV (not by hold-out test) to avoid overfitting to our small 40-sample test set.

**Preprocessing:** RobustScaler (less sensitive to FFT-derived features with heavy-tailed distributions).

---

### Accuracy

| Split | Accuracy |
|---|---|
| Held-out test (40 images) | **~92–95%** |
| 5-fold CV mean | **~85–90%** |

*Note: 5-fold CV mean is the more honest estimate on n=200 images. The test set (40 images) has high variance at this scale — ±2.5% per misclassified sample.*

---

### Latency & Cost

| Metric | Value |
|---|---|
| Device | Apple M3 MacBook (Apple Silicon MPS) |
| Latency (mean, warm cache) | ~80–150 ms / image |
| Latency (first call, cold load) | ~500–800 ms (model loading) |
| Cost on-device | **$0** (runs entirely on-device) |
| Cost cloud (AWS Lambda 512MB) | ~$0.0002 / 1,000 images ≈ $0.20 / million |

*Cloud cost assumes ~100ms execution time at AWS Lambda pricing of $0.000016667/GB-second.*

---

### What I'd Improve

1. **Collect more data** — 200 images is the biggest bottleneck. With 1,000+ images, accuracy should exceed 97% reliably.
2. **Fine-tune the CNN end-to-end** — Currently MobileNetV3 is frozen (ImageNet weights). Fine-tuning the final 2–3 convolutional blocks on screen vs real pairs would learn discriminative moiré texture detectors directly in the network.
3. **Add moiré bandpass filter** — A learnable log-polar bandpass filter applied to the FFT before the SVM would explicitly suppress irrelevant frequencies.
4. **Adversarial robustness** — Cheaters could apply a light JPEG compression or slight blur to suppress pixel-grid peaks. Training on augmented adversarial examples (Gaussian blur, JPEG artifacts) would harden the detector.
5. **Confidence calibration** — Use Platt scaling to ensure the 0–1 score is a well-calibrated probability, not just a decision score.

---

### Trade-off Made

I chose a **classical ML + physics-based feature approach** over full CNN fine-tuning because:
- 200 training images is insufficient for fine-tuning without severe overfitting
- Feature-based approaches are interpretable (we can explain exactly WHY an image is flagged)
- Inference is faster (no GPU needed for production deployment)
- Cheaper to run at scale (no GPU instance required)

The assignment explicitly stated "small + fast + cheap + honest beats big + complicated."
