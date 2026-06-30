# PixelTrace

## Overview
PixelTrace is a system‑level solution for detecting screen recaptures and image fraud. It combines a **hybrid feature extraction pipeline**—lightweight handcrafted computer‑vision descriptors and a deep convolutional neural network (CNN)—to deliver high‑accuracy detection while remaining performant on CPU‑only environments.

## Features
- **Hybrid Mode**: Switch seamlessly between a fast handcrafted CV stack and a powerful CNN‑based extractor. The mode can be toggled at runtime (environment variable `PT_NO_CNN`).
- **Stateless API**: Exposes a FastAPI endpoint (`api/index.py`) that can be deployed server‑less on Vercel or containerised on Render.
- **Optimised Runtime**: Runtime dependencies have been trimmed to stay under Vercel’s 500 MB bundle limit (no pandas, XGBoost, or heavy training‑time libraries).
- **Docker Ready**: A production‑grade `Dockerfile` based on `python:3.12‑slim` for consistent deployments.
- **Cross‑Platform**: Works on macOS, Linux and Windows; GPU acceleration is optional for the CNN path.

## Architecture
```
PixelTrace
├─ cv/
│   ├─ feature_fusion.py   # Orchestrates 11 handcrafted extractors
│   └─ *.py                # Individual CV modules (e.g., chromatic, noise)
├─ ml/
│   ├─ models/
│   │   └─ best_model.pkl  # Pre‑trained SVM classifier
│   └─ cnn/                # Optional PyTorch/Timm models
├─ api/
│   └─ index.py            # FastAPI entry point for Vercel/Render
├─ predict.py               # CLI for offline inference
├─ pyproject.toml           # Minimal runtime dependencies
└─ Dockerfile               # Production container image
```
The **FeatureFusionEngine** merges handcrafted descriptors with CNN embeddings before feeding them to the final classifier, providing robustness against a wide range of manipulations.

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
> The `uv` tool is used for deterministic, fast dependency resolution.

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
The API accepts a multipart/form‑data POST request at `/predict` and returns a JSON payload with the authenticity score.

## Deployment
### Vercel (Server‑less)
1. Ensure `vercel.json` routes to `api/index.py`.
2. Set the environment variable `PT_NO_CNN=1` if you need to stay within the function size limit.
3. Deploy with the Vercel CLI:
   ```bash
   vercel --prod
   ```
### Render (Docker)
1. Push the repository to Render and configure a **Docker** service.
2. Use the provided `Dockerfile`; Render will build the image automatically.
3. Expose port `8000` (FastAPI default) in the Render settings.

## Screenshots

### ✅ Authentic Photo Detection

The system correctly identifies a genuine camera-captured photograph as **Authentic**, assigning a very low fraud probability while showing the extracted forensic indicators used during the decision-making process.

![Authentic Photo Detection](https://github.com/lumen-byte/PixelTrace/blob/main/ScreenShots/Screenshot%202026-07-01%20at%202.26.46%E2%80%AFAM.png)

---

### 🚨 Screen Recapture (Fraud) Detection

The system successfully detects a photograph of a screen/display as a **Screen Recapture (Fraud)** with high confidence by analyzing moiré patterns, chromatic edge artifacts, texture inconsistencies, and display-related forensic features.

![Screen Recapture Detection](https://github.com/lumen-byte/PixelTrace/blob/main/ScreenShots/Screenshot%202026-07-01%20at%202.27.18%E2%80%AFAM.png)

*Replace the above placeholders with the actual screenshots of the live application.*

## Contributing
Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request. Follow the existing code style and run the test suite before submitting.

## License
This project is licensed under the MIT License.
