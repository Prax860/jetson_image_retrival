# Imagify: Comprehensive Project Documentation

## 1. Eraser.io Architecture Diagram Prompt

```
Create a detailed architecture diagram for "Imagify", a semantic image search system for Jetson alert footage:

1. **Actors & Clients**:
   - Jetson Device: pushes alert images via POST /api/v1/ingest (multipart/form-data with image, camera_id, timestamp, alert_type, confidence, location, extra metadata, label, frame_num, object_id, class_id, bbox)
   - Streamlit Chatbot UI: user-facing interface for natural language search
   - Script Clients: send_alert.py (single alert), bulk_ingest.py (directory ingest)

2. **Frontend**:
   - Technology: Streamlit
   - Files: frontend/app.py
   - Key Features:
     - Chat interface for natural language queries
     - Filter controls (top_k, min_score, camera, alert_type)
     - Stats display (total alerts, camera list)
     - Image grid results with bounding boxes
   - API calls: GET /api/v1/collections (stats), POST /api/v1/search

3. **Backend API**:
   - Technology: FastAPI
   - Entry Point: backend/app/main.py
   - Router: backend/app/api/v1/router.py
   - Endpoints:
     - POST /api/v1/ingest
     - POST /api/v1/search
     - GET /api/v1/collections
     - DELETE /api/v1/alerts/{id}
     - GET /api/v1/health
   - Components:
     - Lifespan: pre-initializes vector store on startup
     - CORS middleware: allows all origins for dev

4. **Backend Core (backend/app/core/)**:
   - logging.py: logging setup with console output, quiets httpx/httpcore/urllib3

5. **Models (backend/app/models/)**:
   - alert.py:
     - BBox: left, top, width, height
     - AlertRecord: id, camera_id, timestamp, image_path, image_filename, alert_type, confidence, location_label, extra, label, frame_num, object_id, class_id, bbox, metadata_path, caption, ocr, indexed_at
     - SearchResult: record, score, rank, image_b64
     - AlertResultItem: for API responses (rank, score, id, camera_id, timestamp, alert_type, confidence, location_label, image_filename, image_b64, extra, label, frame_num, object_id, class_id, bbox, caption, ocr)
   - intent.py: IntentFilter (camera_id, label, alert_type, date, time_after, time_before, min_confidence, semantic_query) with validators, has_metadata_filters(), __repr__()

6. **Schemas (backend/app/schemas/)**:
   - alert.py: FastAPI request/response models
     - IngestResponse: id, camera_id, timestamp, alert_type, confidence, location_label, image_filename
     - SearchRequest: query, top_k, camera_id, alert_type, min_score
     - AlertResultItem (API version)
     - SearchResponse: query, total, results
     - CollectionStatsResponse: total_alerts, cameras
     - HealthResponse: status, version, total_indexed, clip_model

7. **Services (backend/app/services/)**:
   - ingest.py: ingest_alert() validates image, saves to data/images/<camera_id>/<record_id>.<ext>, writes metadata to data/metadata/<record_id>.json, returns AlertRecord
   - rag.py:
     - embed_image_file(): loads image, uses CLIP to embed, returns normalized embedding
     - embed_text(): uses CLIP to embed text query
     - index_record(): embeds AlertRecord image, loads/writes metadata JSON, upserts to Chroma with slim metadata
   - retrieval.py:
     - _meta_to_record(): builds AlertRecord from metadata dict (JSON or Chroma fallback)
     - _load_image_b64(): reads image file as base64
   - intent.py:
     - extract_intent(): uses Ollama LLM to extract structured IntentFilter from natural language query, falls back to regex for camera_id if needed
     - _load_llm(): loads Ollama ChatOllama as singleton
     - _load_system_prompt(): loads from backend/app/prompts/query_parser.txt or uses fallback
     - _build_chain(): builds LangChain chain (ChatPromptTemplate â†’ ChatOllama â†’ OutputFixingParser)
   - query_pipeline.py:
     - HybridQueryResult: dataclass with results, intent, where_clause
     - run_query(): main entry point that orchestrates intent extraction, where clause building, retrieval
     - _build_where_clause(): translates IntentFilter to Chroma $and/$or where dict
     - _normalize_intent(): normalizes camera_id to Chroma storage format, ensures semantic_query present
     - _is_trivial_query(): detects greetings to skip search
     - _build_timestamp_range(): builds ISO timestamp range from date + time_after/time_before
     - _normalised_to_bbox(): converts GD's normalized cx/cy/w/h to absolute BBox, clamps to image bounds
   - bbox_overlay.py:
     - annotate_b64(): takes base64 image + BBox, draws box + label, returns annotated base64 JPEG
     - _draw_bbox(): draws rectangle, optionally label with confidence above box (flips below if off-screen)
     - _make_tag(): creates label string from label + confidence
     - _get_font(): tries DejaVu Sans Bold, Arial, then PIL default

8. **Repositories (backend/app/repositories/)**:
   - vector_store.py:
     - VectorStoreRepository class:
       - __init__(): initializes Chroma PersistentClient, gets or creates collection
       - upsert(): stores embedding + sanitized metadata in Chroma
       - query(): takes query embedding, top_k, where, returns (metadata, similarity) list sorted best-first (converts Chroma distance â†’ similarity)
       - count(): returns number of items in collection
       - list_cameras(): returns sorted unique normalized camera IDs from collection
       - camera_id_format(): infers and caches CameraIdFormat from stored data
       - normalize_camera_id(): normalizes camera_id to stored format
       - delete(): removes item by ID
       - reset(): deletes and recreates collection
     - get_vector_store(): singleton accessor
     - _sanitise(): converts metadata values to Chroma-compatible scalars (JSON for non-primitive, empty string for None)

9. **Utils (backend/app/utils/)**:
   - camera_ids.py:
     - CameraIdFormat dataclass: prefix, width, is_numeric, describe()
     - infer_camera_id_format(): infers format from sample camera IDs
     - infer_existing_camera_id_format(): infers format from data/metadata/ and data/images/ on disk
     - normalize_camera_id(): normalizes value to given format
     - parse_camera_id_from_query(): extracts camera ID from user query via regex
     - _split_camera_id(): splits camera ID into (prefix, trailing digits)
   - metadata.py:
     - _meta_path(): returns path to data/metadata/<record_id>.json
     - save_metadata(): writes metadata dict to JSON file
     - load_metadata(): reads metadata dict from JSON file, returns {} if missing/error
     - update_metadata(): merges updates into existing metadata and re-saves
   - image.py: file_to_b64(), b64_data_uri()
   - query_parsers.py: regex-based parsers (parse_camera_id(), parse_time_range(), parse_confidence()) for LLM fallback

10. **Prompts (backend/app/prompts/)**:
    - query_parser.txt: system prompt + examples for Ollama to extract IntentFilter
    - captions.py: CAPTION_SYSTEM_PROMPT, CAPTION_USER_PROMPT for optional LLaVA image captioning

11. **Storage & Data Flow**:
    - Data Directories:
      - data/images/<camera_id>/<record_id>.<ext>: ingested alert images
      - data/metadata/<record_id>.json: canonical alert metadata (source of truth)
      - data/chroma/: ChromaDB persistent storage
    - Chroma Collection:
      - Name: "imagify_alerts"
      - Metadata: hnsw:space = "cosine"
      - Stored Metadata: id, camera_id, timestamp (ISO), label, alert_type, confidence, location_label, image_path, image_filename, caption, indexed_at
    - Ingest Flow:
      Jetson â†’ POST /api/v1/ingest â†’ ingest_alert() (save image/metadata) â†’ index_record() (CLIP embed â†’ Chroma upsert)
    - Search Flow:

12. **Scripts**:
    - scripts/send_alert.py: CLI and importable send_alert() function to push single alert
    - scripts/bulk_ingest.py: walks alerts/<camera_id>/ directories, pushes all images
    - scripts/debug_health.py: simple script to test /health endpoint

13. **Tests**:
    - tests/test_api.py: smoke tests for /health, /collections
    - tests/test_query_pipeline.py: unit tests for camera ID parsing, normalization, run_query()

14. **Dependencies (requirements.txt)**:
    - FastAPI + Uvicorn + python-multipart + Pydantic
    - ChromaDB
    - PyTorch + torchvision + torchaudio
    - Transformers (CLIP) + Pillow + OpenCV + NumPy
    - Streamlit + Requests
    - LangChain + LangChain Core + LangChain Community + LangChain Ollama
    - pytest + httpx

15. **Configuration (config.py Settings)**:
    - APP_NAME, APP_VERSION, DEBUG, LOG_LEVEL
    - HOST, PORT
    - DATA_DIR, IMAGE_STORE_DIR, METADATA_DIR, CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME
    - CLIP_MODEL_NAME, EMBEDDING_DEVICE
    - MAX_IMAGE_SIZE_MB, ALLOWED_IMAGE_EXTENSIONS
    - DEFAULT_TOP_K, MAX_TOP_K
    - LLM_MODEL_NAME, OLLAMA_BASE_URL, LLM_TEMPERATURE, LLM_MAX_TOKENS, QUERY_PROMPT_PATH

16. **Docker**:
    - Dockerfile: builds backend image
    - docker-compose.yml: runs api and frontend services
```


---

## 2. Low-Level Design (LLD)

### 2.1 Project Structure

```
Imagify/
â”œâ”€â”€ backend/
â”‚   â””â”€â”€ app/
â”‚       â”œâ”€â”€ __init__.py
â”‚       â”œâ”€â”€ main.py                          # FastAPI app factory + lifespan
â”‚       â”œâ”€â”€ api/
â”‚       â”‚   â”œâ”€â”€ __init__.py
â”‚       â”‚   â””â”€â”€ v1/
â”‚       â”‚       â”œâ”€â”€ __init__.py
â”‚       â”‚       â””â”€â”€ router.py                # FastAPI endpoints
â”‚       â”œâ”€â”€ core/
â”‚       â”‚   â”œâ”€â”€ __init__.py
â”‚       â”‚   â”œâ”€â”€ config.py                    # Pydantic Settings (app config)
â”‚       â”‚   â”œâ”€â”€ logging.py                   # Logging setup
â”‚       â”‚   â””â”€â”€ exceptions.py                # Custom exceptions
â”‚       â”œâ”€â”€ models/
â”‚       â”‚   â”œâ”€â”€ __init__.py
â”‚       â”‚   â”œâ”€â”€ alert.py                     # Alert domain models
â”‚       â”‚   â”œâ”€â”€ intent.py                    # IntentFilter model
â”‚       â”œâ”€â”€ schemas/
â”‚       â”‚   â”œâ”€â”€ __init__.py
â”‚       â”‚   â””â”€â”€ alert.py                     # FastAPI request/response schemas
â”‚       â”œâ”€â”€ repositories/
â”‚       â”‚   â”œâ”€â”€ __init__.py
â”‚       â”‚   â””â”€â”€ vector_store.py              # ChromaDB repository
â”‚       â”œâ”€â”€ services/
â”‚       â”‚   â”œâ”€â”€ __init__.py
â”‚       â”‚   â”œâ”€â”€ ingest.py                    # Ingest alert image/metadata
â”‚       â”‚   â”œâ”€â”€ rag.py                       # CLIP embedding + indexing
â”‚       â”‚   â”œâ”€â”€ retrieval.py                 # Semantic search + bbox annotation
â”‚       â”‚   â”œâ”€â”€ intent.py                    # LLM intent extraction
â”‚       â”‚   â”œâ”€â”€ query_pipeline.py            # Hybrid RAG orchestration
â”‚       â”‚   â””â”€â”€ bbox_overlay.py              # Bounding box drawing
â”‚       â”œâ”€â”€ utils/
â”‚       â”‚   â”œâ”€â”€ __init__.py
â”‚       â”‚   â”œâ”€â”€ camera_ids.py                # Camera ID normalization/parsing
â”‚       â”‚   â”œâ”€â”€ metadata.py                  # Metadata JSON read/write
â”‚       â”‚   â”œâ”€â”€ image.py                     # Image utilities
â”‚       â”‚   â””â”€â”€ query_parsers.py             # Regex fallback parsers
â”‚       â””â”€â”€ prompts/
â”‚           â”œâ”€â”€ __init__.py
â”‚           â”œâ”€â”€ captions.py                  # Image caption prompts
â”‚           â””â”€â”€ query_parser.txt             # Intent extraction prompt
â”œâ”€â”€ configs/
â”œâ”€â”€ frontend/
â”‚   â””â”€â”€ app.py                               # Streamlit chatbot
â”œâ”€â”€ scripts/
â”‚   â”œâ”€â”€ send_alert.py                        # Single alert sender
â”‚   â”œâ”€â”€ bulk_ingest.py                       # Bulk directory ingest
â”‚   â””â”€â”€ debug_health.py                      # Health check script
â”œâ”€â”€ tests/
â”‚   â”œâ”€â”€ __init__.py
â”‚   â”œâ”€â”€ test_api.py                          # API smoke tests
â”‚   â””â”€â”€ test_query_pipeline.py               # Query pipeline tests
â”œâ”€â”€ .dockerignore
â”œâ”€â”€ .gitignore
â”œâ”€â”€ Dockerfile
â”œâ”€â”€ docker-compose.yml
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ README.md
â””â”€â”€ COMPREHENSIVE_README.md                  # This file
```


---

### 2.2 Core Components LLD

#### 2.2.1 FastAPI Application (`main.py`)
- **Purpose**: Creates and configures the FastAPI application
- **Key Functions**:
  - `lifespan(app: FastAPI)`: Async context manager that sets up logging and pre-initializes the vector store singleton on app startup
  - `create_app() -> FastAPI`: Factory function that creates the FastAPI app, adds CORS middleware, and includes the v1 router
- **Configuration**: Uses `get_settings()` from `config.py` for all app configuration


#### 2.2.2 Configuration (`core/config.py`)
- **Class**: `Settings(BaseSettings)`
  - **Pydantic Config**: Reads from .env file, case-insensitive, ignores extra fields
  - **Fields**:
    - App metadata: `APP_NAME`, `APP_VERSION`, `DEBUG`, `LOG_LEVEL`
    - Server: `HOST`, `PORT`
    - Storage: `DATA_DIR`, `IMAGE_STORE_DIR`, `METADATA_DIR`, `CHROMA_PERSIST_DIR`, `CHROMA_COLLECTION_NAME`
    - CLIP: `CLIP_MODEL_NAME`, `EMBEDDING_DEVICE`
    - Upload: `MAX_IMAGE_SIZE_MB`, `ALLOWED_IMAGE_EXTENSIONS`
    - Retrieval: `DEFAULT_TOP_K`, `MAX_TOP_K`
    - LLM/Ollama: `LLM_MODEL_NAME`, `OLLAMA_BASE_URL`, `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`, `QUERY_PROMPT_PATH`
- **Singleton Accessor**: `get_settings() -> Settings` (cached with `@lru_cache`)


#### 2.2.3 Custom Exceptions (`core/exceptions.py`)
- **Base Class**: `ImagifyError(Exception)`
- **Subclasses**:
  - `IngestError`: Image validation/upload/ingestion failure
  - `EmbeddingError`: CLIP embedding generation failure
  - `VectorStoreError`: ChromaDB operation failure
  - `RetrievalError`: Semantic retrieval failure
  - `IntentExtractionError`: Intent extraction failure
  - `QueryPipelineError`: Query pipeline orchestration failure


#### 2.2.4 Domain Models (`models/alert.py`)
- **`BBox`**:
  - Fields: `left: int = 0`, `top: int = 0`, `width: int = 0`, `height: int = 0`
- **`AlertRecord`**:
  - Core identity: `id: str`, `camera_id: str`, `timestamp: datetime`
  - Image: `image_path: str`, `image_filename: str`
  - Detection metadata: `alert_type: Optional[str]`, `confidence: Optional[float]`, `location_label: Optional[str]`, `extra: Dict[str, Any]`
  - Jetson-specific: `label: Optional[str]`, `frame_num: Optional[int]`, `object_id: Optional[int]`, `class_id: Optional[int]`, `bbox: Optional[BBox]`
  - Canonical metadata: `metadata_path: Optional[str]`
  - Enrichment: `caption: str`, `ocr: str`
  - Bookkeeping: `indexed_at: datetime`
- **`SearchResult`**:
  - Fields: `record: AlertRecord`, `score: float`, `rank: int`, `image_b64: str`
- **`AlertResultItem`**:
  - API response model with all fields exposed to frontend


#### 2.2.5 Intent Model (`models/intent.py`)
- **`IntentFilter`**:
  - Metadata filters: `camera_id: Optional[str]`, `label: Optional[str]`, `alert_type: Optional[str]`, `date: Optional[str]`, `time_after: Optional[str]`, `time_before: Optional[str]`, `min_confidence: Optional[float]`
  - Semantic: `semantic_query: Optional[str]`
  - **Validators**:
    - `_empty_to_none()`: converts empty strings to None
    - `_normalise_time()`: normalizes time to HH:MM
    - `_normalise_date()`: accepts natural language dates via `date_parser` (local import to avoid circular dependency)
  - **Methods**:
    - `has_metadata_filters() -> bool`: returns True if any metadata filter is set
    - `__repr__()`: string representation with only non-None fields


#### 2.2.6 Vector Store Repository (`repositories/vector_store.py`)
- **`VectorStoreRepository` Class**:
  - **Initialization**:
    - Creates `data/chroma/` directory if needed
    - Initializes `chromadb.PersistentClient`
    - Gets or creates collection `cfg.CHROMA_COLLECTION_NAME` with `hnsw:space = "cosine"`
    - Caches `_camera_id_format` for reuse
  - **Methods**:
    - `upsert(record_id: str, embedding: List[float], metadata: Dict[str, Any]) -> None`: stores embedding + sanitized metadata; normalizes camera_id first
    - `query(query_embedding: List[float], top_k: int = 10, where: Optional[Dict[str, Any]] = None) -> List[Tuple[Dict[str, Any], float]]`: returns list of (metadata, similarity) sorted best-first; converts Chroma distance (0-2) â†’ similarity (1-0)
    - `count() -> int`: returns number of items in collection
    - `list_cameras() -> List[str]`: returns sorted unique normalized camera IDs
    - `camera_id_format() -> CameraIdFormat`: infers format from stored metadata
    - `normalize_camera_id(camera_id: object) -> Optional[str]`: normalizes camera_id using stored format
    - `delete(record_id: str) -> None`: removes item by ID
    - `reset() -> None`: deletes and recreates collection
- **Singleton Accessor**: `get_vector_store() -> VectorStoreRepository`
- **Helpers**:
  - `_sanitise(m: Dict[str, Any]) -> Dict[str, Any]`: converts metadata values to Chroma-compatible scalars (JSON for non-primitive, empty string for None)


#### 2.2.7 Ingest Service (`services/ingest.py`)
- **`ingest_alert(...) -> AlertRecord`**:
  1. Validates image extension against `ALLOWED_IMAGE_EXTENSIONS`
  2. Validates image size against `MAX_IMAGE_SIZE_MB`
  3. Verifies image is valid using PIL `Image.verify()`
  4. Generates unique `record_id` with `uuid.uuid4()`
  5. Saves image to `IMAGE_STORE_DIR/<camera_id>/<record_id>.<ext>`
  6. Parses bbox dict into `BBox` model if provided
  7. Writes initial metadata dict to `METADATA_DIR/<record_id>.json`
  8. Re-writes metadata with `metadata_path` added
  9. Returns `AlertRecord`


#### 2.2.8 RAG Service (`services/rag.py`)
- **Module-level Singletons**: `_model: Optional[CLIPModel]`, `_processor: Optional[CLIPProcessor]`
- **`_load_clip() -> Tuple[CLIPModel, CLIPProcessor]`**: loads CLIP model and processor from `cfg.CLIP_MODEL_NAME`, moves to `cfg.EMBEDDING_DEVICE`, sets to eval mode
- **`embed_image_file(image_path: str | Path) -> List[float]`**:
  - Opens image with PIL, converts to RGB
  - Preprocesses with CLIP processor
  - Gets image features with CLIP, normalizes by L2 norm
  - Returns embedding as Python list
- **`embed_text(text: str) -> List[float]`**:
  - Tokenizes text with CLIP processor
  - Gets text features with CLIP, normalizes by L2 norm
  - Returns embedding as Python list
- **`index_record(record: AlertRecord, vector_store: VectorStoreRepository) -> None`**:
  1. Embeds the image file from `record.image_path`
  2. Loads canonical metadata from `METADATA_DIR/<record.id>.json`
  3. Adds `indexed_at` (ISO string) to metadata JSON
  4. Builds slim Chroma metadata dict (only fields needed for search)
  5. Upserts to vector store


#### 2.2.9 Retrieval Service (`services/retrieval.py`)
- **`search_alerts(...) -> List[SearchResult]`**:
  1. Validates query is not empty
  2. Embeds the semantic query with `embed_text()`
  3. Builds Chroma `where` clause (if provided)
  4. Queries Chroma with `top_k * 4` (over-fetch to allow post-filtering)
  5. For each result:
     a. Loads full metadata from `METADATA_DIR/<id>.json`; falls back to Chroma metadata if missing
     b. Converts metadata to `AlertRecord` with `_meta_to_record()`
     c. Applies alert_type filter (legacy)
     d. Loads image as base64 with `_load_image_b64()`
     f. If GD found object, draws its bbox with `annotate_b64()`; else uses stored YOLO bbox
     g. Adds `SearchResult` to results list
  6. Trims to top_k results and returns
- **Helpers**:
  - `_meta_to_record(meta: dict) -> AlertRecord`: constructs AlertRecord from metadata dict; handles confidence sentinel (-1.0), parses extra from JSON string if needed, parses bbox dict to BBox
  - `_load_image_b64(image_path: str) -> str`: reads image file as base64 string; returns empty string if missing/error


#### 2.2.10 Intent Extraction Service (`services/intent.py`)
- **Module-level Singleton**: `_llm: Optional[ChatOllama]`
- **`_load_llm() -> ChatOllama`**: loads Ollama ChatOllama with `cfg.LLM_MODEL_NAME`, `cfg.OLLAMA_BASE_URL`, `cfg.LLM_TEMPERATURE`, `cfg.LLM_MAX_TOKENS`, `format="json"`
- **`_load_system_prompt() -> str`**: loads prompt from `cfg.QUERY_PROMPT_PATH`; uses fallback if file missing
- **`_build_chain()`**: builds LangChain chain:
  1. `ChatPromptTemplate` with system message (from prompt file) + human message (`"User query: {query}"`)
  2. `ChatOllama` LLM
  3. `OutputFixingParser` (wraps `PydanticOutputParser[IntentFilter]`)
- **`extract_intent(query: str) -> IntentFilter`**:
  - Builds chain and invokes with query
  - If semantic_query is missing, fills it from label or original query
  - Falls back to pure semantic search (no filters) on any failure


#### 2.2.11 Query Pipeline Orchestrator (`services/query_pipeline.py`)
- **`HybridQueryResult` Dataclass**:
  - Fields: `results: List[SearchResult]`, `intent: IntentFilter`, `where_clause: Optional[Dict[str, Any]]`
- **`run_query(...) -> HybridQueryResult`**:
  1. Extracts intent with `extract_intent(query)`
  2. If intent has no camera_id, tries regex fallback with `parse_camera_id_from_query(query)`
  3. Normalizes intent with `_normalize_intent()`
  4. Checks if query is trivial (greeting) â†’ returns empty results
  5. Builds Chroma `where` clause with `_build_where_clause()`
  6. Runs search with `search_alerts()` (passes original query to GD)
  7. Returns `HybridQueryResult`
- **Helpers**:
  - `_build_where_clause(intent: IntentFilter, vector_store: Optional[VectorStoreRepository] = None) -> Optional[Dict[str, Any]]`:
    - Builds $and/$or Chroma filter from IntentFilter
    - For camera_id: checks if stored camera IDs are numeric â†’ adds both string and int equality to $or
    - For date + time_after/time_before: builds ISO timestamp range
  - `_normalize_intent(intent: IntentFilter, vector_store: VectorStoreRepository, query: str) -> IntentFilter`:
    - Normalizes camera_id to Chroma format
    - Ensures semantic_query is present (uses label, camera_id, alert_type, or query if needed)
  - `_is_trivial_query(query: str) -> bool`: detects greetings (hi, hello, etc.)
  - `_build_timestamp_range(date_str: str, time_after: Optional[str], time_before: Optional[str]) -> Tuple[Optional[str], Optional[str]]`: builds ISO timestamp range from date + time bounds


- **Module-level Singleton**: `_gdino = None`
  - Validates config/weights files exist
  1. Loads model (singleton)
  4. Selects detection with highest confidence
  5. Converts GD's normalized (cx, cy, w, h) to absolute BBox with `_normalised_to_bbox()`
  6. Clamps BBox to image bounds
  7. Returns BBox or None if no detection
- **`_normalised_to_bbox(box: torch.Tensor, img_w: int, img_h: int) -> BBox`**: converts normalized coords to absolute pixels, clamps to image bounds


#### 2.2.13 Bounding Box Overlay Service (`services/bbox_overlay.py`)
- **Constants**: `_BOX_COLOR = "#00FF41"`, `_BOX_WIDTH = 3`, `_LABEL_BG = "#00FF41"`, `_LABEL_FG = "#000000"`, `_FONT_SIZE = 14`, `_JPEG_QUALITY = 85`
- **`annotate_b64(image_b64: str, bbox: BBox, label: Optional[str] = None, confidence: Optional[float] = None) -> str`**:
  - Decodes base64 image, opens with PIL
  - Draws bbox + label with `_draw_bbox()`
  - Re-encodes as JPEG base64
  - Returns original image on any error
- **Helpers**:
  - `_draw_bbox(img: Image.Image, bbox: BBox, label: Optional[str], confidence: Optional[float]) -> Image.Image`:
    - Draws green rectangle
    - Draws label tag above box (flips below if off-screen)
  - `_make_tag(label: Optional[str], confidence: Optional[float]) -> str`: builds label string
  - `_get_font() -> ImageFont.ImageFont`: tries DejaVu Sans Bold â†’ Arial â†’ PIL default


#### 2.2.14 Camera ID Utilities (`utils/camera_ids.py`)
- **`CameraIdFormat` Dataclass**:
  - Fields: `prefix: str = ""`, `width: int = 0`
  - Properties: `is_numeric: bool` (True if no prefix and width=0)
  - Methods: `describe() -> str`
- **`infer_camera_id_format(samples: Iterable[object]) -> CameraIdFormat`**: infers format from sample camera IDs
- **`infer_existing_camera_id_format() -> CameraIdFormat`**: infers format from `data/metadata/` and `data/images/` on disk
- **`normalize_camera_id(value: object, camera_format: CameraIdFormat | None = None) -> Optional[str]`**: normalizes value to given format
- **`parse_camera_id_from_query(query: str) -> Optional[str]`**: extracts camera ID from query via regex
- **`_split_camera_id(value: object) -> Optional[Tuple[str, str]]`**: splits into (prefix, trailing digits)


#### 2.2.15 Metadata Utilities (`utils/metadata.py`)
- **`_meta_path(record_id: str) -> Path`**: returns `data/metadata/<record_id>.json`
- **`save_metadata(record_id: str, data: Dict[str, Any]) -> Path`**: writes data to JSON file with indent=2, uses `str()` for non-serializable values
- **`load_metadata(record_id: str) -> Dict[str, Any]`**: reads JSON file; returns {} if missing/error
- **`update_metadata(record_id: str, updates: Dict[str, Any]) -> Dict[str, Any]`**: loads existing metadata, merges updates, re-saves


#### 2.2.16 API Router (`api/v1/router.py`)
- **Endpoints**:
  - `POST /api/v1/ingest`:
    - Accepts multipart/form-data: image, camera_id, timestamp, alert_type, confidence, location_label, extra_json, label, frame_num, object_id, class_id, bbox
    - Parses extra_json and bbox from JSON strings
    - Calls `ingest_alert()`, then `index_record()`
    - Returns `IngestResponse` with 201 Created
  - `POST /api/v1/search`:
    - Accepts `SearchRequest` (query, top_k, camera_id, alert_type, min_score)
    - Calls `run_query()` from query_pipeline
    - Maps `SearchResult` to `AlertResultItem`
    - Returns `SearchResponse`
  - `GET /api/v1/collections`:
    - Returns `CollectionStatsResponse` (total_alerts, cameras)
  - `DELETE /api/v1/alerts/{alert_id}`:
    - Deletes alert from vector store
    - Returns `{"deleted": alert_id}`
  - `GET /api/v1/health`:
    - Returns `HealthResponse` (status, version, total_indexed, clip_model)


#### 2.2.17 Frontend (`frontend/app.py`)
- **Configuration**: Uses `IMAGIFY_API_URL` env var (default: http://localhost:8000)
- **Session State**: `st.session_state.messages` stores chat history
- **Sidebar**:
  - App title/caption
  - Stats (calls `GET /api/v1/collections`)
  - Filters: `top_k` slider, `min_score` slider, camera selectbox, alert_type text input
- **Main Chat Area**:
  - Header/examples
  - Renders chat history
  - Chat input for user queries
- **Search Flow**:
  - When user submits query: adds user message to state â†’ calls `POST /api/v1/search` â†’ adds assistant message + results â†’ renders image grid


#### 2.2.18 Scripts
- **`scripts/send_alert.py`**:
  - Importable `send_alert()` function for Jetson pipelines
  - CLI interface for testing
- **`scripts/bulk_ingest.py`**:
  - Walks `alerts/<camera_id>/` directories
  - Pushes all images to API
  - Supports dry-run, camera filter, delay between requests
- **`scripts/debug_health.py`**:
  - Tests `/health` endpoint


#### 2.2.19 Tests
- **`tests/test_api.py`**:
  - `test_health()`: checks /health returns 200 and status "ok"
  - `test_collections()`: checks /collections returns 200 with total_alerts and cameras
- **`tests/test_query_pipeline.py`**:
  - `test_camera_id_parser_supports_common_query_forms()`
  - `test_normalize_camera_id_matches_numeric_storage()`
  - `test_run_query_normalizes_camera_id_before_building_where()`
  - `test_run_query_skips_trivial_greeting()`


---

### 2.3 Data Flow Diagrams

#### 2.3.1 Ingest Flow
```
Jetson Device
    â†“
POST /api/v1/ingest (multipart/form-data)
    â†“
router.ingest_endpoint()
    â†“
ingest_alert()
    â”œâ”€ Validate image (extension, size, validity)
    â”œâ”€ Save image to data/images/<camera_id>/<record_id>.<ext>
    â”œâ”€ Write metadata to data/metadata/<record_id>.json
    â””â”€ Return AlertRecord
    â†“
index_record()
    â”œâ”€ Load CLIP model (singleton)
    â”œâ”€ Embed image with CLIP
    â”œâ”€ Load metadata JSON
    â”œâ”€ Update metadata with indexed_at
    â””â”€ Upsert (embedding + slim metadata) to ChromaDB
    â†“
Return IngestResponse (201 Created)
```

#### 2.3.2 Search Flow
```
User
    â†“
Streamlit UI: type query
    â†“
POST /api/v1/search (SearchRequest)
    â†“
router.search_endpoint()
    â†“
run_query() (query_pipeline)
    â”œâ”€ extract_intent() â†’ IntentFilter
    â”‚   â”œâ”€ Load Ollama LLM (singleton)
    â”‚   â”œâ”€ Build LangChain chain
    â”‚   â”œâ”€ Invoke chain with user query
    â”‚   â””â”€ Regex fallback for camera_id if needed
    â”œâ”€ _normalize_intent()
    â”œâ”€ Check trivial query
    â”œâ”€ _build_where_clause() â†’ Chroma filter dict
    â””â”€ search_alerts()
        â”œâ”€ embed_text() (semantic_query)
        â”œâ”€ vector_store.query() â†’ (metadata, score) list
        â”œâ”€ For each result:
        â”‚   â”œâ”€ Load full metadata from data/metadata/<id>.json
        â”‚   â”œâ”€ _meta_to_record() â†’ AlertRecord
        â”‚   â”œâ”€ Load image as base64
        â”‚   â”œâ”€ annotate_b64() (draw bbox on image)
        â”‚   â””â”€ Add SearchResult
        â””â”€ Return List[SearchResult]
    â†“
Map SearchResult â†’ AlertResultItem
    â†“
Return SearchResponse
    â†“
Streamlit UI: display image grid
```


---

### 2.4 Configuration & Dependencies

#### 2.4.1 Environment Variables
Create a `.env` file in the root directory:
```env
# App
DEBUG=True
LOG_LEVEL=INFO

# Storage (optional)
DATA_DIR=./data
CHROMA_PERSIST_DIR=./data/chroma

# CLIP
CLIP_MODEL_NAME=openai/clip-vit-base-patch32
EMBEDDING_DEVICE=cpu  # or cuda

# LLM / Ollama
LLM_MODEL_NAME=qwen2.5:1.5b
OLLAMA_BASE_URL=http://localhost:11434

```

#### 2.4.2 Dependencies
Install with:
```bash
pip install -r requirements.txt
```


---

## 3. Quick Start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment**
   ```bash
   # Copy .env.example if available, or create .env
   ```

3. **Start FastAPI backend**
   ```bash
   uvicorn backend.app.main:app --reload
   ```

4. **Start Streamlit frontend**
   ```bash
   streamlit run frontend/app.py
   ```

5. **Send a test alert**
   ```bash
   python scripts/send_alert.py --image /path/to/alert.jpg --camera-id cam-01
   ```


---

## 4. Docker

```bash
docker-compose up --build
```

- API: http://localhost:8000
- UI: http://localhost:8501
- Swagger docs: http://localhost:8000/docs
