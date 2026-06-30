# PixelTrace

## Overview
Screens lie. A photograph of a display can look just as crisp and convincing as the real thing — until you know exactly what to look for. PixelTrace was built to know what to look for.

It's a system-level solution for detecting screen recaptures and image fraud. Rather than leaning entirely on brute-force deep learning, PixelTrace is built around a **hybrid feature extraction pipeline** that pairs lightweight, handcrafted computer-vision descriptors with a deep convolutional neural network (CNN) embedding. The payoff is detection accuracy that doesn't need a GPU farm to run — it's fast, predictable, and comfortable in CPU-only environments, which matters far more in a real production system than it ever does in a research notebook.

## Architecture & Hybrid Mode
At the core of PixelTrace sits the **FeatureFusionEngine** — an orchestrator that merges handcrafted visual descriptors (moiré patterns, chromatic aberration, texture inconsistencies, and noise profiles) with deep CNN embeddings before routing everything to the final classifier. This combination is what gives the system resilience across such a wide spectrum of manipulation techniques.

**Hybrid Mode** is the cornerstone of that flexibility. You can switch seamlessly between the highly optimized handcrafted CV stack and the heavier CNN-based extractor, toggled dynamically at runtime through the `PT_NO_CNN` environment variable — speed when latency is critical, depth when accuracy is paramount.

- **Stateless API** — A FastAPI endpoint (`api/index.py`) ready to deploy serverless on Vercel or containerized on Render, with no hidden state to manage between requests.
- **Optimized Runtime** — Every dependency earns its place. The pipeline is stripped of heavy training-time libraries, leaning on highly optimized OpenCV and NumPy routines to stay within strict serverless bundle limits while still responding in under 70ms on warm local runs.
- **Docker Ready** — A production-grade `Dockerfile` built on `python:3.12-slim`, so deployments behave the same way everywhere.
- **Cross-Platform** — Runs reliably on macOS, Linux, and Windows, with optional GPU acceleration for the CNN path.

## Visual Evidence

### Authentic Photo Detection
The system correctly recognizes a genuine, camera-captured photograph as Authentic, assigning a very low fraud probability while surfacing the forensic indicators that informed the decision.

![Authentic Photo Detection](https://github.com/lumen-byte/PixelTrace/blob/main/ScreenShots/Screenshot%202026-07-01%20at%204.27.09%E2%80%AFAM.png)

<br/>

### Screen Recapture (Fraud) Detection
The system flags a photograph of a screen or display as a Screen Recapture (Fraud) with high confidence, drawing on moiré patterns, chromatic edge artifacts, texture inconsistencies, and other display-related forensic signals.

![Screen Recapture Detection](https://github.com/lumen-byte/PixelTrace/blob/main/ScreenShots/Screenshot%202026-07-01%20at%204.29.46%E2%80%AFAM.png)

## Model Performance
Careful optimization of both the feature extractors and the classification model gives PixelTrace a strong balance between accuracy and inference speed:

- **Accuracy** — **85.0%** test accuracy on unseen data.
- **Cross-Validation** — A rigorous 5-fold stratified cross-validation run shows a stable **83.87% (± 6.67%)** mean accuracy across folds, evidence that the model is genuinely generalizing rather than memorizing the training set.
- **Inference Latency** — The handcrafted CV pipeline has been heavily optimized using vectorization, hardware-accelerated transforms, and integer-shift local binary patterns. The full pipeline runs in roughly **60–70ms** on local machines and scales efficiently to constrained cloud environments without cold-start penalties.
- **Classifier** — Final classification is handled by an optimized XGBoost model, chosen for how well it separates complex, non-linear relationships within the handcrafted forensic features while staying extremely fast at inference time.

## Live Demo
- **Vercel:** [pixel-trace-8966vpegb-lumenbyte1.vercel.app](https://pixel-trace-8966vpegb-lumenbyte1.vercel.app/)
- **Render:** [pixeltrace.onrender.com](https://pixeltrace.onrender.com/)

## Installation

```bash
# Clone the repository
git clone https://github.com/lumen-byte/PixelTrace.git
cd PixelTrace

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install runtime dependencies using uv for deterministic resolution
uv pip install -r <(uv pip compile pyproject.toml --no-dev)
```

## Usage

**Command Line Interface (Offline Inference)**
```bash
.venv/bin/python predict.py path/to/image.jpg
```

**Development Server (API)**
```bash
.venv/bin/python -m uvicorn app.main:app --reload
```
The API accepts a `multipart/form-data` POST request at `/predict` and returns a JSON payload containing the authenticity score and key forensic features.

## Deployment

### Vercel (Serverless)
1. Ensure `vercel.json` routes to `api/index.py`.
2. Set `PT_NO_CNN=1` in your environment variables to bypass the deep learning model and stay within the 500MB function size limit.
3. Deploy via the Vercel CLI:
   ```bash
   vercel --prod
   ```

### Render (Docker)
1. Push the repository to Render and configure a Docker service.
2. The provided `Dockerfile` builds the production image automatically.
3. Expose port `8000` (FastAPI default) in the Render service settings.

## License
This project is licensed under the MIT License.

## Developers

**Abhimanyu Pratap Singh**
E23CSEU0193
B.Tech Computer Science and Engineering
Bennett University
