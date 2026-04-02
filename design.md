# Medical Image Database Search & Analysis Platform

## 1. Project Scope & Objectives
- **Goal**: Build a web application that aggregates publicly available medical imaging datasets, enabling AI-powered search by imaging modality and providing dataset insights.
- **Target Users**: Researchers, clinicians, data scientists, and students interested in medical imaging analysis.
- **Core Functionalities**:
  1. **AI Search**: Users can query images by textual description and filter by modality (e.g., MRI, CT, X‑ray, Ultrasound).
  2. **Dataset Insights**: Automatic generation of statistics (modality distribution, patient demographics, acquisition equipment, quality metrics) and visualizations.
  3. **Dataset Discovery**: Browse collections, view sample images, and access download links or API endpoints.
  4. **Community Contributions**: Allow dataset curators to submit new sources with metadata and licensing information.

## 2. Search & Analysis Strategy
### 2.1. Data Sources
| Source | Modality(s) | Size | Access | License |
|--------|-------------|------|--------|---------|
| NIH ChestX‑ray14 | X‑ray | ~112k images | Public download | Public domain / CC0 |
| The Cancer Imaging Archive (TCIA) | MRI, CT, PET | Multi‑modal, >1M images | Public API & bulk download | Varies (CC0, CC‑BY) |
| OpenNeuro | MRI, fMRI | ~2M sessions | Public API | CC0 |
| MGH MRI Dataset | MRI | ~1.5M images | Direct download | CC‑BY‑NC |
| Public MRI/CT repositories (e.g., UK Biobank) | MRI, CT | Large scale | Requires registration | Varies |

*Strategy*: Identify 5–10 high‑quality, openly licensed repositories covering all major modalities. Prioritize sources with machine‑readable metadata (JSON/CSV) and stable URLs.

### 2.2. Ingestion Pipeline
1. **Metadata Harvesting** – Write scrapers / API clients to collect:
   - Image file URLs or identifiers
   - Modality tags
   - Patient demographics (age, sex, ethnicity) – where available and permissible
   - Acquisition parameters (scanner model, field strength, resolution)
   - Quality scores (e.g., motion artifacts, signal‑to‑noise ratio)
   - Licensing information
2. **Download & Storage** – Store raw images in a tiered storage system (e.g., S3‑compatible object storage) with checksum verification.
3. **Metadata Repository** – Insert records into a centralized metadata database (PostgreSQL or MongoDB) with fields matching the schema below.

### 2.3. Database Schema & Indexing
```sql
-- Simplified relational schema
TABLE images (
    image_id      UUID PRIMARY KEY,
    dataset_id    UUID REFERENCES datasets(dataset_id),
    file_path     TEXT,
    modality      TEXT,          -- e.g., MRI, CT, XRAY
    width_px      INT,
    height_px     INT,
    checksum      TEXT,
    license       TEXT,
    created_at    TIMESTAMP
);

TABLE datasets (
    dataset_id    UUID PRIMARY KEY,
    name          TEXT,
    description   TEXT,
    source_url    TEXT,
    license       TEXT,
    total_images  INT,
    created_at    TIMESTAMP
);

TABLE metadata (
    image_id      UUID PRIMARY KEY REFERENCES images(image_id),
    age           INT,
    sex           TEXT,
    ethnicity     TEXT,
    acquisition_date DATE,
    scanner_model TEXT,
    field_strength TEXT,        -- for MRI
    slice_thickness REAL,
    pixel_spacing REAL,
    quality_score REAL
);
```
- **Vector Embeddings**: Use a pretrained multimodal model (e.g., MedCLIP, CLIP‑Med) to generate 512‑dim embeddings for each image. Store embeddings in a vector database (FAISS, Milvus, or Pinecone) for fast approximate nearest‑neighbor search.
- **Indexing**: Create indexes on `modality`, `dataset_id`, and vector similarity search. Optionally maintain secondary indexes for age, sex, and quality_score for insight queries.

### 2.4. AI Models for Search
| Function | Model | Reasoning |
|----------|-------|-----------|
| **Modality Detection** | Fine‑tuned ResNet‑50 or EfficientNet‑B3 on public modality classification datasets | Provides robust, lightweight modality classification (MRI, CT, X‑ray, US, PET). |
| **Semantic Search** | MedCLIP (CLIP fine‑tuned on medical images) or BioViT | Maps image patches to a shared embedding space enabling text‑to‑image retrieval. |
| **Insight Generation** | Tabular ML models (e.g., LightGBM) on metadata to predict distribution patterns | Helps surface hidden correlations (e.g., scanner model vs. image quality). |

- **Pipeline**: For each newly ingested image, compute embeddings asynchronously and store them in the vector DB. Periodically recompute embeddings for existing images if model upgrades occur.

### 2.5. Search API Design
- **Endpoint**: `GET /api/v1/search`
- **Parameters**:
  - `q` (string) – free‑text query
  - `modality` (enum) – optional filter
  - `dataset_id` (UUID) – optional filter
  - `page` / `size` – pagination
  - `sort_by` – relevance, date, or metadata fields
- **Response**:
  ```json
  {
    "results": [
      {
        "image_id": "uuid",
        "preview_url": "https://cdn.example.com/thumb/uuid.jpg",
        "modality": "MRI",
        "dataset": "TCIA-XXYZ",
        "similarity_score": 0.93
      }
    ],
    "stats": {
      "total_hits": 1245,
      "modality_distribution": {"MRI": 540, "CT": 310, "XRAY": 395}
    }
  }
  ```
- **Additional Endpoints**:
  - `GET /api/v1/dataset/{id}` – dataset metadata and summary statistics.
  - `GET /api/v1/insights` – aggregated dataset insights (e.g., age/sex distribution heatmaps).

### 2.6. Dataset Insight Generation
1. **Statistical Aggregations** – Query the metadata DB for counts, means, and distributions grouped by modality, age bins, sex, etc.
2. **Visualizations** – Generate charts (bar plots, histograms) on‑the‑fly using a charting library (e.g., Chart.js, Plotly) and cache the rendered PNG/SVG.
3. **Report API** – `GET /api/v1/insights?modality=MRI` returns JSON with:
   - Modality count
   - Age‑sex pyramid
   - Scanner model breakdown
   - Quality score histogram
4. **Scheduled Jobs** – Nightly batch jobs recompute heavy statistics and update cached insight artifacts.

## 3. Suggested Architectural Structures
```
frontend/
│─ src/
│   ├─ components/        # SearchBar, ResultGrid, InsightModal
│   ├─ pages/             # Home, Browse, SearchResults, Insights
│   └─ services/          # apiClient.ts, embeddingWorker.ts
│
backend/
│─ src/
│   ├─ ingestion/         # scraper workers, downloader
│   ├─ models/            # modality classifier, embedding generator
│   ├─ db/                # ORM models, migrations
│   ├─ api/               # REST endpoints, search service
│   └─ workers/           # async embedding computation (Kafka/RabbitMQ)
│
infra/
│─ docker-compose.yml
│─ k8s/                   # Helm charts for deployment
│─ terraform/             # Cloud resource definitions (S3, RDS, VPC)
│
tests/
│─ unit/
│─ integration/
│─ e2e/
│
docs/
│─ design.md              # (this file)
│─ api_spec.yaml
│
```

### 3.1. Technology Stack
- **Frontend**: React (TypeScript) + Material‑UI + Redux Toolkit
- **Backend**: FastAPI (Python 3.11) + SQLAlchemy + Pydantic
- **Vector DB**: FAISS (self‑hosted) or Pinecone (managed)
- **Containerization**: Docker + Docker‑Compose for dev; Kubernetes for prod
- **CI/CD**: GitHub Actions (lint, test, build, push images)
- **Monitoring**: Prometheus + Grafana; logging via Loki

### 3.2. Scalability Considerations
- **Horizontal Scaling**: Stateless API servers behind a load balancer; separate worker pool for embedding generation.
- **Cache Layer**: Use Redis to cache recent search queries and embedding results.
- **Asynchronous Processing**: Offload heavy embedding computation to a background queue (e.g., AWS SQS + Lambda workers).

## 4. Next Steps
1. **Finalize Scope** – Confirm target modalities and user stories with stakeholders.
2. **Prototype Ingestion** – Build a scraper for one source (e.g., NIH ChestX‑ray) and store a small metadata set.
3. **Model Proof‑of‑Concept** – Fine‑tune a modality classifier on a labeled subset and evaluate accuracy.
4. **API Mockup** – Implement a minimal `/search` endpoint returning mock results.
5. **UI Wireframes** – Create low‑fidelity mockups of the search interface and insight dashboard.
