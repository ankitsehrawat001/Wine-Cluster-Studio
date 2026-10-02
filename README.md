# Wine Cluster Atlas

An immersive React Three Fiber and Three.js observatory for the wine clustering project. The Streamlit launcher hosts the website and starts the FastAPI model service; predictions continue to use the saved `wine_model.joblib` and `wine_scaler.joblib` artifacts.

## Run on Streamlit

From the project root in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd frontend
npm install
npm run build
cd ..
streamlit run app.py
```

Open `http://localhost:8501`. Streamlit hosts the built frontend and starts the model API on `127.0.0.1:8000` automatically. Rebuild the frontend with `npm run build` after changing files under `frontend/src`.

## Features

- Five scroll-linked chapters with a persistent Three.js world and camera journey.
- Procedural planet, star field, spiral galaxy, nebula layers, orbital regions, and observatory geometry.
- Real K-Means wine samples, cluster filtering, feature-axis selectors, and hover/click inspection.
- Original model-backed single-sample inference, CSV upload, and clustered CSV export.
- Saved obsidian/pearl theme preference, reduced-motion support, and adaptive mobile particle counts.
