import os
import shutil
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from predict import predict

app = FastAPI(
    title="PixelTrace API",
    description="Evidence-Driven Computer Vision for Screen Recapture Detection",
    version="1.0.0",
)

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static Frontend HTML ──────────────────────────────────────────────────────
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PixelTrace — Screen Recapture Forensic Analysis</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    fontFamily: {
                        sans: ['Plus Jakarta Sans', 'sans-serif'],
                        outfit: ['Outfit', 'sans-serif'],
                    },
                    colors: {
                        brand: {
                            purple: '#a855f7',
                            cyan: '#06b6d4',
                            dark: '#030712',
                            card: '#0f172a'
                        }
                    }
                }
            }
        }
    </script>
    <style>
        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: #030712;
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(168, 85, 247, 0.15) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(6, 116, 212, 0.12) 0%, transparent 45%);
        }
        .radial-progress {
            transform: rotate(-90deg);
        }
        .glass-card {
            background: rgba(15, 23, 42, 0.45);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        .glass-card:hover {
            border-color: rgba(255, 255, 255, 0.1);
        }
    </style>
</head>
<body class="text-slate-100 min-h-screen flex flex-col antialiased">

    <!-- Header -->
    <header class="border-b border-white/5 bg-slate-950/40 backdrop-blur-md sticky top-0 z-50">
        <div class="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="h-9 w-9 bg-gradient-to-tr from-brand-purple to-brand-cyan rounded-xl flex items-center justify-center font-bold text-lg text-slate-900 shadow-lg shadow-purple-500/20">
                    P
                </div>
                <span class="text-xl font-extrabold tracking-tight bg-gradient-to-r from-brand-purple to-brand-cyan bg-clip-text text-transparent font-outfit">
                    PixelTrace
                </span>
            </div>
            <div class="flex items-center space-x-3">
                <span class="text-[10px] tracking-wider uppercase font-semibold border border-white/10 bg-white/5 px-2.5 py-1 text-slate-300 rounded-lg font-outfit">Production Model</span>
                <span class="relative flex h-2.5 w-2.5">
                    <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                    <span class="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500"></span>
                </span>
                <span class="text-xs text-slate-400 font-semibold font-outfit">Active</span>
            </div>
        </div>
    </header>

    <!-- Main Container -->
    <main class="flex-grow max-w-6xl w-full mx-auto px-4 py-8 grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        <!-- Left Side: Upload & Control -->
        <section class="lg:col-span-5 flex flex-col space-y-6">
            
            <div class="glass-card rounded-2xl p-6 shadow-xl transition-all duration-300">
                <h2 class="text-lg font-bold mb-2 tracking-tight text-white font-outfit">Forensic Upload</h2>
                <p class="text-xs text-slate-400 mb-6 leading-relaxed">Submit an image to run a forensic visual scan for moiré lines, chromatic fringing, and display-grid texture distortions.</p>
                
                <!-- Drop Zone -->
                <div id="dropzone" class="border border-slate-800 hover:border-brand-purple/50 transition-all duration-300 rounded-xl p-8 flex flex-col items-center justify-center cursor-pointer bg-slate-950/20 group relative min-h-[220px]">
                    <input type="file" id="fileInput" accept="image/*" class="hidden">
                    <div class="p-3 bg-purple-500/10 rounded-2xl mb-4 group-hover:scale-110 transition-transform duration-300">
                        <svg class="w-8 h-8 text-brand-purple" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.75" d="M4 16l4.586-4.586a2.5 2.5 0 013.414 0L18 16m-2-2l1.586-1.586a2.5 2.5 0 013.414 0L22 12m-2-2v10a1 1 0 01-1 1H3a1 1 0 01-1-1V5a1 1 0 011-1h9m4 0h2m4 0h2m-4 0v2m0-6V4m0 0L9 9m4-5L9 9"></path>
                        </svg>
                    </div>
                    <p class="text-sm font-semibold text-slate-300 mb-1">Drag and drop image here</p>
                    <p class="text-[11px] text-slate-500 font-sans">Supports JPG, PNG, WEBP (Max 15MB)</p>
                </div>
            </div>

            <!-- Preview Panel -->
            <div id="previewCard" class="hidden glass-card rounded-2xl p-6 flex flex-col items-center justify-center shadow-xl">
                <h3 class="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-4 self-start font-outfit">Image Preview</h3>
                <img id="imagePreview" class="max-h-[300px] w-auto rounded-xl object-contain border border-white/5 shadow-inner" src="" alt="Preview">
                <button id="analyzeBtn" class="mt-6 w-full py-3.5 bg-gradient-to-r from-brand-purple to-brand-cyan hover:from-purple-600 hover:to-cyan-500 text-slate-950 font-bold rounded-xl transition-all duration-300 shadow-lg shadow-purple-500/20 hover:scale-[1.01] active:scale-[0.99] uppercase tracking-wider text-xs font-outfit">
                    Run Forensic Analysis
                </button>
            </div>
        </section>

        <!-- Right Side: Results & Forensic Evidence -->
        <section class="lg:col-span-7 flex flex-col space-y-6">
            
            <!-- Default Empty State -->
            <div id="emptyState" class="glass-card border border-dashed border-slate-800 rounded-2xl p-12 flex flex-col items-center justify-center text-center h-full min-h-[400px]">
                <div class="h-16 w-16 bg-slate-900/50 border border-white/5 rounded-2xl flex items-center justify-center mb-6">
                    <svg class="w-8 h-8 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                    </svg>
                </div>
                <h3 class="text-slate-200 font-bold mb-2 font-outfit text-base">Awaiting Forensic Scan</h3>
                <p class="text-xs text-slate-500 max-w-xs leading-relaxed font-sans">Upload an image file in the left panel and click 'Run Forensic Scan' to display recaptured fraud diagnostics.</p>
            </div>

            <!-- Loading State -->
            <div id="loadingState" class="hidden glass-card rounded-2xl p-12 flex flex-col items-center justify-center text-center h-full min-h-[400px]">
                <div class="relative w-14 h-14 mb-6">
                    <div class="absolute inset-0 rounded-full border-4 border-brand-purple/20"></div>
                    <div class="absolute inset-0 rounded-full border-4 border-brand-purple border-t-transparent animate-spin"></div>
                </div>
                <h3 class="text-slate-200 font-bold mb-1 font-outfit text-base">Computing Forensic Signatures</h3>
                <p class="text-xs text-brand-cyan animate-pulse font-mono tracking-wider uppercase">Running feature extractors...</p>
            </div>

            <!-- Results Report -->
            <div id="resultsReport" class="hidden glass-card rounded-2xl p-6 space-y-8 shadow-xl transition-all duration-500">
                
                <!-- Main Header (Gauge & Verdict) -->
                <div class="flex items-center space-x-8">
                    <!-- Radial Gauge -->
                    <div class="relative flex items-center justify-center">
                        <svg class="radial-progress w-24 h-24" viewBox="0 0 100 100">
                            <circle class="text-slate-800/80" stroke-width="8" stroke="currentColor" fill="transparent" r="40" cx="50" cy="50"/>
                            <circle id="gaugeFill" class="text-brand-purple transition-all duration-1000" stroke-width="8" stroke-dasharray="251.2" stroke-dashoffset="251.2" stroke-linecap="round" stroke="currentColor" fill="transparent" r="40" cx="50" cy="50"/>
                        </svg>
                        <div class="absolute flex flex-col items-center justify-center">
                            <span id="verdictPercent" class="text-2xl font-extrabold text-white font-outfit">0%</span>
                            <span class="text-[8px] text-slate-500 uppercase tracking-widest font-bold">Fraud</span>
                        </div>
                    </div>

                    <!-- Verdict Texts -->
                    <div class="flex-grow">
                        <h3 class="text-[10px] font-bold text-slate-400 uppercase tracking-widest font-outfit">Analysis Verdict</h3>
                        <div id="verdictBadge" class="inline-flex items-center mt-2 px-3 py-1 rounded-xl font-bold text-xs uppercase tracking-wider">
                            Analyzing...
                        </div>
                        <p id="verdictDesc" class="text-xs text-slate-400 mt-3 leading-relaxed font-sans"></p>
                    </div>
                </div>

                <!-- Latency Stats -->
                <div class="grid grid-cols-2 gap-4 border-t border-b border-white/5 py-4 font-outfit text-xs text-slate-400">
                    <div>
                        <span class="block text-slate-500 uppercase tracking-wider text-[9px]">Scanning speed</span>
                        <span id="latencyText" class="font-bold text-slate-200 text-sm">- ms</span>
                    </div>
                    <div>
                        <span class="block text-slate-500 uppercase tracking-wider text-[9px]">Analysis Model</span>
                        <span class="font-bold text-slate-200 text-sm">Calibrated SVM</span>
                    </div>
                </div>

                <!-- Forensic Evidence Breakdown -->
                <div>
                    <h4 class="text-xs font-bold text-slate-400 uppercase tracking-widest mb-6 font-outfit">// Forensic Evidence Indicators</h4>
                    <div class="space-y-5">
                        
                        <!-- Moiré / FFT -->
                        <div>
                            <div class="flex justify-between text-xs mb-1.5 font-outfit">
                                <span class="text-slate-300">Moiré frequency peaks (FFT)</span>
                                <span id="moireVal" class="text-slate-400 font-mono">-</span>
                            </div>
                            <div class="h-2 bg-slate-950 rounded-full overflow-hidden border border-white/5">
                                <div id="moireBar" class="h-full bg-brand-purple transition-all duration-1000" style="width: 0%"></div>
                            </div>
                        </div>

                        <!-- Chromatic Aberration -->
                        <div>
                            <div class="flex justify-between text-xs mb-1.5 font-outfit">
                                <span class="text-slate-300">Chromatic edge misalignment</span>
                                <span id="caVal" class="text-slate-400 font-mono">-</span>
                            </div>
                            <div class="h-2 bg-slate-950 rounded-full overflow-hidden border border-white/5">
                                <div id="caBar" class="h-full bg-brand-cyan transition-all duration-1000" style="width: 0%"></div>
                            </div>
                        </div>

                        <!-- Texture Contrast -->
                        <div>
                            <div class="flex justify-between text-xs mb-1.5 font-outfit">
                                <span class="text-slate-300">GLCM texture contrast</span>
                                <span id="textureVal" class="text-slate-400 font-mono">-</span>
                            </div>
                            <div class="h-2 bg-slate-950 rounded-full overflow-hidden border border-white/5">
                                <div id="textureBar" class="h-full bg-brand-purple transition-all duration-1000" style="width: 0%"></div>
                            </div>
                        </div>

                        <!-- Noise Level -->
                        <div>
                            <div class="flex justify-between text-xs mb-1.5 font-outfit">
                                <span class="text-slate-300">Image noise-floor level</span>
                                <span id="noiseVal" class="text-slate-400 font-mono">-</span>
                            </div>
                            <div class="h-2 bg-slate-950 rounded-full overflow-hidden border border-white/5">
                                <div id="noiseBar" class="h-full bg-brand-cyan transition-all duration-1000" style="width: 0%"></div>
                            </div>
                        </div>

                    </div>
                </div>

            </div>

        </section>
    </main>

    <!-- Footer -->
    <footer class="border-t border-white/5 py-6 text-center text-xs text-slate-500 bg-slate-950/20 font-outfit">
        PixelTrace Screen Recapture Classifier &copy; 2026. Made with Tailwind CSS & FastAPI.
    </footer>

    <!-- Logic -->
    <script>
        const dropzone = document.getElementById('dropzone');
        const fileInput = document.getElementById('fileInput');
        const previewCard = document.getElementById('previewCard');
        const imagePreview = document.getElementById('imagePreview');
        const analyzeBtn = document.getElementById('analyzeBtn');
        
        const emptyState = document.getElementById('emptyState');
        const loadingState = document.getElementById('loadingState');
        const resultsReport = document.getElementById('resultsReport');

        let selectedFile = null;

        // Click to choose
        dropzone.addEventListener('click', () => fileInput.click());

        // File chosen
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });

        // Drag Over
        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropzone.classList.add('border-brand-purple');
        });

        // Drag Leave
        dropzone.addEventListener('dragleave', () => {
            dropzone.classList.remove('border-brand-purple');
        });

        // Drop
        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('border-brand-purple');
            if (e.dataTransfer.files.length > 0) {
                handleFile(e.dataTransfer.files[0]);
            }
        });

        function handleFile(file) {
            selectedFile = file;
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
                previewCard.classList.remove('hidden');
                emptyState.classList.remove('hidden');
                resultsReport.classList.add('hidden');
            };
            reader.readAsDataURL(file);
        }

        // Run analysis
        analyzeBtn.addEventListener('click', async () => {
            if (!selectedFile) return;

            // Update loading state
            emptyState.classList.add('hidden');
            resultsReport.classList.add('hidden');
            loadingState.classList.remove('hidden');

            const formData = new FormData();
            formData.append('file', selectedFile);

            const t0 = performance.now();
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    throw new Error('Server returned an error');
                }

                const data = await response.json();
                const latency = Math.round(performance.now() - t0);

                showResults(data, latency);
            } catch (err) {
                alert('Analysis failed: ' + err.message);
                emptyState.classList.remove('hidden');
                loadingState.classList.add('hidden');
            }
        });

        function showResults(data, latency) {
            loadingState.classList.add('hidden');
            resultsReport.classList.remove('hidden');

            const score = data.score; // float [0, 1]
            const pct = Math.round(score * 100);

            // Gauge setup
            document.getElementById('verdictPercent').textContent = pct + '%';
            const circle = document.getElementById('gaugeFill');
            const circumference = 2 * Math.PI * 40; // ~251.2
            const offset = circumference - (score * circumference);
            circle.style.strokeDashoffset = offset;

            // Verdict setting
            const verdictBadge = document.getElementById('verdictBadge');
            const verdictDesc = document.getElementById('verdictDesc');

            if (score > 0.5) {
                verdictBadge.textContent = 'Screen Recapture (Fraud)';
                verdictBadge.className = 'inline-block mt-2 px-3 py-1.5 rounded-xl font-bold text-xs uppercase tracking-wider bg-rose-500/10 text-rose-400 border border-rose-500/20';
                verdictDesc.textContent = 'This image displays high-confidence indicators of being a photograph of a display screen, showing color misalignment and texture interference.';
                circle.className = 'text-rose-500 transition-all duration-1000';
            } else {
                verdictBadge.textContent = 'Authentic Photo (Real)';
                verdictBadge.className = 'inline-block mt-2 px-3 py-1.5 rounded-xl font-bold text-xs uppercase tracking-wider bg-emerald-500/10 text-emerald-400 border border-emerald-500/20';
                verdictDesc.textContent = 'This image matches natural camera noise and lens profiles, with no evidence of subpixel grids or refresh distortion.';
                circle.className = 'text-emerald-500 transition-all duration-1000';
            }

            document.getElementById('latencyText').textContent = latency + ' ms';

            // Forensic details (normalised between 0 and 100 for display)
            const features = data.features;
            
            // 1. Moiré Peak Ratio
            const moire = features.moire_peak_ratio || 0.0;
            const moirePct = Math.min(100, Math.round(moire * 1500)); // scale for visual representation
            document.getElementById('moireVal').textContent = moire.toFixed(4);
            document.getElementById('moireBar').style.width = moirePct + '%';

            // 2. CA edge color variance
            const ca = features.ca_edge_color_var || 0.0;
            const caPct = Math.min(100, Math.round(ca / 10)); // scale for visuals
            document.getElementById('caVal').textContent = ca.toFixed(2);
            document.getElementById('caBar').style.width = caPct + '%';

            // 3. GLCM Texture Contrast
            const texture = features.glcm_contrast || 0.0;
            const texturePct = Math.min(100, Math.round(texture * 50));
            document.getElementById('textureVal').textContent = texture.toFixed(4);
            document.getElementById('textureBar').style.width = texturePct + '%';

            // 4. Noise Mean
            const noise = features.noise_mean || 0.0;
            const noisePct = Math.min(100, Math.round(noise * 300));
            document.getElementById('noiseVal').textContent = noise.toFixed(4);
            document.getElementById('noiseBar').style.width = noisePct + '%';
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse(content=HTML_TEMPLATE)


@app.post("/predict")
def predict_endpoint(file: UploadFile = File(...)):
    # 1. Validate file extension
    suffix = Path(file.filename).suffix.lower()
    if suffix not in [".jpg", ".jpeg", ".png", ".webp"]:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload JPG, PNG, or WEBP."
        )

    # 2. Save file temporarily in workspace
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        try:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process upload: {e}"
            )

    # 3. Predict and get both score and features
    try:
        t0 = time.perf_counter()
        score, features = predict(str(tmp_path), return_features=True)
        latency = (time.perf_counter() - t0) * 1000
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {e}"
        )
    finally:
        # Cleanup temporary file
        if tmp_path.exists():
            tmp_path.unlink()

    # Filter features to keep the report lightweight
    report_features = {
        "moire_peak_ratio": features.get("moire_peak_ratio", 0.0),
        "ca_edge_color_var": features.get("ca_edge_color_var", 0.0),
        "glcm_contrast": features.get("glcm_contrast", 0.0),
        "noise_mean": features.get("noise_mean", 0.0),
    }

    return {
        "score": score,
        "latency_ms": round(latency, 2),
        "features": report_features
    }


@app.get("/health")
def health():
    return {"status": "healthy"}