# PixelTrace

## Overview
Screens lie. A photograph of a display can look just as crisp and convincing as the real thing—until you know exactly what to look for. PixelTrace is built to know what to look for.

It is a system-level solution for detecting screen recaptures and image fraud. Rather than relying solely on brute-force deep learning, PixelTrace is built around a **Hybrid Feature Extraction Pipeline**. It intelligently pairs lightweight, handcrafted computer-vision descriptors with a deep convolutional neural network (CNN) embedding. The result is detection accuracy that doesn't demand a GPU farm to run—it is fast, predictable, and comfortable in CPU-only environments, which matters significantly more in real-world production systems than in a research notebook.

## The Architecture & Hybrid Mode
At the core of PixelTrace is the **FeatureFusionEngine**, an orchestrator that merges handcrafted visual descriptors (extracting moiré patterns, chromatic aberration, texture inconsistencies, and noise profiles) with deep CNN embeddings before routing them to the final classifier. This design provides resilience against a wide spectrum of manipulation techniques.

**Hybrid Mode** is the cornerstone of this system's flexibility. You can switch seamlessly between the highly optimized handcrafted CV stack and the heavier CNN-based extractor, toggled dynamically at runtime with the `PT_NO_CNN` environment variable. This allows you to choose speed when latency is critical, and depth when accuracy is paramount.

- **Stateless API** — A FastAPI endpoint (`api/index.py`) ready to deploy serverless on Vercel or containerized on Render, with no hidden state to manage between requests.
- **Optimized Runtime** — Every runtime dependency earns its place. The pipeline is stripped of heavy training-time libraries, using highly optimized OpenCV and NumPy routines to stay under strict serverless bundle limits and deliver responses in under 70ms on warm local runs.
- **Docker Ready** — A production-grade `Dockerfile` built on `python:3.12-slim` ensures that deployments behave the same way everywhere.
- **Cross-Platform** — Runs reliably on macOS, Linux, and Windows, with optional GPU acceleration for the CNN path.

## Visual Evidence

### Authentic Photo Detection
The system correctly recognizes a genuine, camera-captured photograph as Authentic, assigning a very low fraud probability while surfacing the forensic indicators that informed the decision.

![Authentic Photo Detection](https://github.com/lumen-byte/PixelTrace/blob/main/ScreenShots/Screenshot%202026-07-01%20at%202.26.46%E2%80%AFAM.png)

<br/>

### Screen Recapture (Fraud) Detection
The system flags a photograph of a screen or display as a Screen Recapture (Fraud) with high confidence, drawing on moiré patterns, chromatic edge artifacts, texture inconsistencies, and other display-related forensic signals.

![Screen Recapture Detection](https://github.com/lumen-byte/PixelTrace/blob/main/ScreenShots/Screenshot%202026-07-01%20at%202.27.18%E2%80%AFAM.png)

## Model Performance
Through rigorous optimization of both the feature extractors and the classification model, PixelTrace achieves an excellent balance between accuracy and inference latency:

- **Accuracy**: Achieves an **85.0%** test accuracy on unseen data.
- **Cross-Validation**: Rigorous 5-Fold Stratified Cross-Validation shows a highly stable **81.87% (± 6.67%)** mean accuracy across folds, proving the model generalizes well rather than just memorizing the training set.
- **Inference Latency**: The handcrafted CV pipeline has been heavily optimized (using vectorization, hardware-accelerated transforms, and integer-shift local binary patterns). The entire pipeline executes in approximately **60-70ms** on local machines, scaling efficiently to constrained cloud environments without cold-start penalties.
- **Classifier**: The final classification relies on an optimized XGBoost model, chosen for its superior performance in separating complex, non-linear relationships within handcrafted forensic features, while remaining extremely fast during inference.

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
2. The provided `Dockerfile` will automatically build the production image.
3. Expose port `8000` (FastAPI default) in the Render service settings.

## License
This project is licensed under the MIT License.

## Developers

**Abhimanyu Pratap Singh**  
E23CSEU0193  
B.tech Computer Science and Engineering  
Bennett University  
