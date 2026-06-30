# PixelTrace

## Overview
Screens lie. A photo of a photo can look just as crisp, just as convincing, as the real thing — until you know what to look for. PixelTrace was built to know what to look for.

It's a system-level solution for catching screen recaptures and image fraud, built around a **hybrid feature extraction pipeline** that pairs lightweight handcrafted computer-vision descriptors with a deep convolutional neural network (CNN). The result is detection accuracy that doesn't demand a GPU farm to run — it's comfortable on CPU-only environments, which matters a lot more in production than in a research notebook.

## Features
- **Hybrid Mode** — Switch seamlessly between a fast handcrafted CV stack and a heavier CNN-based extractor, toggled at runtime with the `PT_NO_CNN` environment variable. Speed when you need it, depth when you don't.
- **Stateless API** — A FastAPI endpoint (`api/index.py`) ready to deploy serverless on Vercel or containerized on Render, with no hidden state to manage between requests.
- **Optimized Runtime** — Every runtime dependency earns its place. No pandas, no XGBoost, no heavy training-time libraries — just what's needed to stay under Vercel's 500 MB bundle limit.
- **Docker Ready** — A production-grade `Dockerfile` built on `python:3.12-slim` for deployments that behave the same way everywhere.
- **Cross-Platform** — Runs on macOS, Linux, and Windows, with GPU acceleration available for the CNN path when you want it, optional when you don't.

## Architecture
```
PixelTrace
├─ cv/
│   ├─ feature_fusion.py   # Orchestrates 11 handcrafted extractors
│   └─ *.py                # Individual CV modules (e.g., chromatic, noise)
├─ ml/
│   ├─ models/
│   │   └─ best_model.pkl  # Pre-trained SVM classifier
│   └─ cnn/                # Optional PyTorch/Timm models
├─ api/
│   └─ index.py            # FastAPI entry point for Vercel/Render
├─ predict.py               # CLI for offline inference
├─ pyproject.toml           # Minimal runtime dependencies
└─ Dockerfile               # Production container image
```
At the heart of it all sits the **FeatureFusionEngine**, which merges handcrafted descriptors with CNN embeddings before handing them off to the final classifier — giving the system resilience against a wide range of manipulation techniques, not just the obvious ones.

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

# Install runtime dependencies
uv pip install -r <(uv pip compile pyproject.toml --no-dev)
```
> `uv` is used here deliberately — it gives fast, deterministic dependency resolution, so the environment you build today behaves the same way next month.

## Usage
### CLI
```bash
# Run inference on a local image
.venv/bin/python predict.py path/to/image.jpg
```
### API
```bash
# Start the development server
.venv/bin/python -m uvicorn app.main:app --reload
```
The API accepts a `multipart/form-data` POST request at `/predict` and returns a JSON payload with the authenticity score.

## Deployment

### Vercel (Serverless)
1. Make sure `vercel.json` routes to `api/index.py`.
2. Set `PT_NO_CNN=1` if you need to stay within the function size limit.
3. Deploy with the Vercel CLI:
   ```bash
   vercel --prod
   ```

### Render (Docker)
1. Push the repository to Render and configure a **Docker** service.
2. Use the provided `Dockerfile` — Render builds the image automatically.
3. Expose port `8000` (FastAPI default) in the Render settings.

## Screenshots

### ✅ Authentic Photo Detection
The system correctly recognizes a genuine, camera-captured photograph as **Authentic**, assigning a very low fraud probability while surfacing the forensic indicators that informed the decision.

![Authentic Photo Detection](https://github.com/lumen-byte/PixelTrace/blob/main/ScreenShots/Screenshot%202026-07-01%20at%202.26.46%E2%80%AFAM.png)

---

### 🚨 Screen Recapture (Fraud) Detection
The system flags a photograph of a screen or display as a **Screen Recapture (Fraud)** with high confidence, drawing on moiré patterns, chromatic edge artifacts, texture inconsistencies, and other display-related forensic signals.

![Screen Recapture Detection](https://github.com/lumen-byte/PixelTrace/blob/main/ScreenShots/Screenshot%202026-07-01%20at%202.27.18%E2%80%AFAM.png)

## Contributing
Contributions are genuinely welcome. Fork the repository, create a feature branch, and open a pull request. Please follow the existing code style and run the test suite before submitting — it keeps the project trustworthy for everyone who depends on it.

## License
This project is licensed under the MIT License.
