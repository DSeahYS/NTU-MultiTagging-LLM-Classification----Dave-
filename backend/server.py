import asyncio
import json
import os
import time
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from progress_tracker import ProgressTracker, SCHEMA_CATEGORIES
from reranker_core import create_client, classify_paper
from pdf_processor import extract_text_from_pdf

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=BASE_DIR / ".env")
load_dotenv(dotenv_path=ENV_PATH)

PAPERS_FOLDER = BASE_DIR / "uploads"
if not PAPERS_FOLDER.exists() or not any(PAPERS_FOLDER.glob("*.pdf")):
    fallback = PROJECT_ROOT / "QA_Generator (For RAG)" / "uploads"
    if fallback.exists():
        PAPERS_FOLDER = fallback
PROGRESS_DIR = BASE_DIR / "backend" / "data"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

tracker = ProgressTracker(str(PROGRESS_DIR))
tracker_lock = threading.Lock()
active_ws_clients: list[WebSocket] = []
processing_task: Optional[asyncio.Task] = None
global_api_key = os.environ.get("openrouterkey", "")

# All available models for parallel fan-out
PARALLEL_MODELS = [
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "google/gemma-4-26b-a4b-it:free",
    "openai/gpt-oss-120b:free",
]

# Scan papers folder
ALL_PAPERS = []
if PAPERS_FOLDER.exists():
    ALL_PAPERS = [f.name for f in PAPERS_FOLDER.glob("*.pdf")]

class StartRequest(BaseModel):
    api_key: str = ""
    model: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    delay_ms: int = 0
    concurrency: int = 20

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(title="Multi-Tag Paper Sorter Dashboard", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

if (BASE_DIR / "index.html").exists():
    app.mount("/css", StaticFiles(directory=str(BASE_DIR / "css")), name="css")
    app.mount("/js", StaticFiles(directory=str(BASE_DIR / "js")), name="js")

async def broadcast(msg: dict):
    dead = []
    for ws in active_ws_clients:
        try:
            await ws.send_json(msg)
        except:
            dead.append(ws)
    for w in dead:
        active_ws_clients.remove(w)

async def broadcast_status():
    await broadcast({"type": "status_update", "data": tracker.get_stats(len(ALL_PAPERS))})

async def broadcast_log(message: str, level: str = "info"):
    await broadcast({"type": "log", "message": message, "level": level})

completed_count = 0

async def classify_one(sem: asyncio.Semaphore, client, filename: str, model: str, total: int):
    """Classify a single paper under the semaphore concurrency limit."""
    global completed_count
    async with sem:
        try:
            # 1. Extract text
            pdf_path = PAPERS_FOLDER / filename
            pdf_data = await asyncio.to_thread(extract_text_from_pdf, str(pdf_path))
            paper_text = pdf_data["full_text"]
            
            # 2. Call LLM to multi-tag
            tags = await asyncio.to_thread(classify_paper, client, paper_text, model)
            
            # 3. Update tracker
            with tracker_lock:
                tracker.update_progress(filename, tags)
                completed_count += 1
                local_count = completed_count
                
            await broadcast({"type": "progress", "data": {"filename": filename, "tags": tags}})
            if local_count % 10 == 0 or local_count == total:
                await broadcast_status()
                
            return filename, tags
        except Exception as e:
            await broadcast_log(f"Error processing {filename}: {e}", "error")
            return filename, []

async def rerank_worker(api_key: str, model: str, delay_ms: int, concurrency: int):
    global completed_count
    try:
        if not ALL_PAPERS:
            await broadcast_log(f"No PDFs found in {PAPERS_FOLDER}", "error")
            return

        client = create_client(api_key)
        total = len(ALL_PAPERS)

        # Gather all unprocessed papers
        pending_papers = [f for f in ALL_PAPERS if f not in tracker.classified]

        if not pending_papers:
            await broadcast_log("All papers already classified!", "success")
            await broadcast_status()
            return

        completed_count = total - len(pending_papers)
        num_models = len(PARALLEL_MODELS)
        sem = asyncio.Semaphore(concurrency)

        await broadcast_log(
            f"⚡ Parallel mode: {len(pending_papers)} remaining papers across {num_models} models, concurrency={concurrency}",
            "info"
        )

        # Build all tasks, round-robining models
        tasks = []
        for slot, filename in enumerate(pending_papers):
            assigned_model = PARALLEL_MODELS[slot % num_models]
            tasks.append(classify_one(sem, client, filename, assigned_model, total))

        await asyncio.gather(*tasks)

        await broadcast_log("Classification fully completed!", "success")
        await broadcast_status()
    except asyncio.CancelledError:
        await broadcast_log("Processing paused.", "warning")
    except Exception as e:
        await broadcast_log(f"Fatal error: {e}", "error")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_ws_clients.append(websocket)
    await websocket.send_json({"type": "status_update", "data": tracker.get_stats(len(ALL_PAPERS))})
    try:
        while True:
            await websocket.receive_text()
    except:
        active_ws_clients.remove(websocket)

@app.post("/api/start")
async def api_start(req: StartRequest):
    global processing_task
    if processing_task and not processing_task.done():
        return {"status": "already_running"}
    
    key = req.api_key or global_api_key
    if not key:
        return {"status": "error", "message": "No API key provided."}
        
    processing_task = asyncio.create_task(rerank_worker(key, req.model, req.delay_ms, req.concurrency))
    return {"status": "started"}

@app.post("/api/pause")
async def api_pause():
    global processing_task
    if processing_task and not processing_task.done():
        processing_task.cancel()
    return {"status": "paused"}

@app.post("/api/reset")
async def api_reset():
    global processing_task
    if processing_task and not processing_task.done():
        processing_task.cancel()
    tracker.reset()
    await broadcast_status()
    await broadcast_log("Progress tracker reset to 0.", "info")
    return {"status": "reset"}

@app.get("/api/export")
async def api_export():
    if not ALL_PAPERS:
        return {"status": "error", "message": "No data"}
    
    # 1. Generate master JSON
    master_path = OUTPUT_DIR / "categorized_papers.json"
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(tracker.classified, f, indent=4)
        
    # 1.5 Generate master CSV
    import csv
    csv_path = OUTPUT_DIR / "master_tags.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Tags"])
        for fname, tags in tracker.classified.items():
            writer.writerow([fname, ", ".join(tags)])
        
    # 2. Generate 30 individual markdown files
    for cat in SCHEMA_CATEGORIES.keys():
        papers_in_cat = [fname for fname, tags in tracker.classified.items() if cat in tags]
        
        path_md = OUTPUT_DIR / f"{cat}.md"
        with open(path_md, "w", encoding="utf-8") as fm:
            fm.write(f"# {cat}\n")
            fm.write(f"**Domain**: {SCHEMA_CATEGORIES[cat]}\n")
            fm.write(f"**Total Papers**: {len(papers_in_cat)}\n\n")
            fm.write("---\n\n")
            
            for fname in papers_in_cat:
                fm.write(f"- {fname}\n")
            
    return {"status": "success", "message": "Exported master JSON and 30 category lists to output/ folder"}

@app.get("/")
async def serve_index():
    return FileResponse(BASE_DIR / "index.html")

if __name__ == "__main__":
    import uvicorn
    # Use port 8044 to avoid conflict with RAG/QA Generator on 8042/8043
    uvicorn.run(app, host="0.0.0.0", port=8044)
