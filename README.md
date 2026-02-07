# Facial Recognition API

A high-performance FastAPI backend for employee facial recognition system designed for canteen entry management. Uses deep learning for face detection and vector similarity search for matching.

## Features

- **Employee Registration** - Register employees with face images
- **Face Recognition** - Identify employees from face images using AI
- **Manual Fallback** - Employee ID-based entry when face recognition fails
- **Entry Logging** - Track all entries with method (face/manual) and confidence
- **Multiple Face Images** - Support up to 3 face images per employee for better accuracy
- **Async Architecture** - Non-blocking I/O for high throughput

---

## Technology Stack (All Open Source)

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Web Framework** | FastAPI | High-performance async API framework |
| **Database** | PostgreSQL | Relational database for employee data |
| **Vector Search** | pgvector | PostgreSQL extension for similarity search |
| **Face Detection** | RetinaFace | Deep learning face detector (via DeepFace) |
| **Face Embeddings** | ArcFace | State-of-the-art face recognition model |
| **Deep Learning** | TensorFlow/Keras | Neural network backend |
| **ORM** | SQLAlchemy (async) | Database abstraction with asyncpg driver |
| **Image Processing** | Pillow, OpenCV | Image manipulation and preprocessing |
| **Async File I/O** | aiofiles | Non-blocking file operations |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT REQUEST                                  │
│                         (Image Upload via API)                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FASTAPI ROUTES                                  │
│                    /api/employees, /api/recognize, /api/entry               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            SERVICE LAYER                                     │
│              EmployeeService, FaceService, EntryService                      │
│                      (Business Logic & Orchestration)                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
┌───────────────────────┐ ┌───────────────────┐ ┌───────────────────────────┐
│    FACE SERVICE       │ │   REPOSITORY      │ │    IMAGE UTILS            │
│                       │ │     LAYER         │ │                           │
│  • Face Detection     │ │                   │ │  • Validation             │
│  • Embedding Gen      │ │  • EmployeeRepo   │ │  • Format Conversion      │
│  • Similarity Calc    │ │  • FaceEmbedRepo  │ │  • Async File Storage     │
│                       │ │  • EntryLogRepo   │ │                           │
└───────────────────────┘ └───────────────────┘ └───────────────────────────┘
         │                         │                         │
         ▼                         ▼                         ▼
┌───────────────────────┐ ┌───────────────────┐ ┌───────────────────────────┐
│      DEEPFACE         │ │    POSTGRESQL     │ │      FILE SYSTEM          │
│                       │ │    + PGVECTOR     │ │                           │
│  RetinaFace Detector  │ │                   │ │    face_images/           │
│  ArcFace Embeddings   │ │  Vector Storage   │ │    └── EMP-001/           │
│                       │ │  & Similarity     │ │        └── primary.jpg    │
└───────────────────────┘ └───────────────────┘ └───────────────────────────┘
```

---

## How Face Recognition Works

### 1. Face Detection (RetinaFace)

When an image is uploaded, the system first detects faces using **RetinaFace**, a deep learning-based face detector that:

- Locates face bounding boxes in the image
- Identifies facial landmarks (eyes, nose, mouth)
- Handles multiple faces, varying angles, and lighting conditions
- Returns the largest/most prominent face if multiple detected

### 2. Face Embedding Generation (ArcFace)

Once a face is detected, it's converted into a **512-dimensional vector** (embedding) using the **ArcFace** model:

```
Input Image → Face Detection → Face Alignment → ArcFace CNN → 512-D Vector
     │                                                              │
     │            [0.023, -0.156, 0.089, ..., 0.234]               │
     │            ◄──────── 512 floating-point numbers ──────────► │
```

**Why 512 dimensions?**
- Each dimension captures a specific facial feature
- The vector is a compact numerical representation of the face
- Similar faces produce similar vectors; different faces produce different vectors

### 3. Vector Storage (pgvector)

The embedding is stored in PostgreSQL using the **pgvector** extension:

```sql
-- Table structure
CREATE TABLE face_embeddings (
    id SERIAL PRIMARY KEY,
    employee_id INTEGER REFERENCES employees(id),
    embedding VECTOR(512),  -- pgvector type
    image_path TEXT,
    is_primary BOOLEAN
);

-- Index for fast similarity search
CREATE INDEX ON face_embeddings 
USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);
```

### 4. Face Matching (Cosine Distance)

When recognizing a face, the system:

1. Generates embedding for the input image
2. Searches for similar vectors using **cosine distance**
3. Returns the closest match if within threshold

```sql
-- Similarity search query
SELECT employee_id, embedding <=> '[input_vector]'::vector AS distance
FROM face_embeddings
WHERE embedding <=> '[input_vector]'::vector < 0.55  -- threshold
ORDER BY distance ASC
LIMIT 1;
```

**Cosine Distance Explained:**
- **0.0** = Identical faces (perfect match)
- **0.0 - 0.4** = Same person (high confidence)
- **0.4 - 0.55** = Likely same person (moderate confidence)
- **> 0.55** = Different person (no match)

```
Cosine Distance = 1 - Cosine Similarity

Confidence Score = 1 - Distance
Example: Distance 0.2 → Confidence 80%
```

---

## Data Flow Diagrams

### Employee Registration Flow

```
┌──────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  Client  │────►│  POST /api/  │────►│  Validate   │────►│   Detect     │
│          │     │  employees/  │     │   Image     │     │    Face      │
└──────────┘     │  register    │     └─────────────┘     └──────────────┘
                 └──────────────┘                                │
                                                                 ▼
┌──────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│ Response │◄────│    Save      │◄────│   Store     │◄────│   Generate   │
│   JSON   │     │  to Disk     │     │  Embedding  │     │   512-D      │
└──────────┘     │  (async)     │     │  in pgvector│     │   Vector     │
                 └──────────────┘     └─────────────┘     └──────────────┘
```

### Face Recognition Flow

```
┌──────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  Client  │────►│  POST /api/  │────►│   Detect    │────►│   Generate   │
│  Image   │     │  recognize   │     │    Face     │     │  Embedding   │
└──────────┘     └──────────────┘     └─────────────┘     └──────────────┘
                                                                 │
                                                                 ▼
┌──────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│ Response │◄────│   Return     │◄────│   Compare   │◄────│   Vector     │
│ Employee │     │   Match or   │     │  Distance   │     │   Search     │
│   Info   │     │   Not Found  │     │  < 0.55?    │     │  (pgvector)  │
└──────────┘     └──────────────┘     └─────────────┘     └──────────────┘
```

---

## Project Structure

```
facial-api/
├── app/
│   ├── main.py              # Application entry point
│   ├── core/
│   │   ├── config.py        # Environment configuration
│   │   ├── database.py      # Async database connection
│   │   └── exceptions.py    # Custom exception handlers
│   ├── models/
│   │   ├── employee.py      # Employee ORM model
│   │   ├── face_embedding.py # Face vector storage model
│   │   └── entry_log.py     # Entry tracking model
│   ├── schemas/
│   │   ├── employee.py      # Request/Response schemas
│   │   ├── recognize.py     # Recognition schemas
│   │   └── entry.py         # Entry log schemas
│   ├── repositories/
│   │   ├── employee_repo.py      # Employee CRUD operations
│   │   ├── face_embedding_repo.py # Vector similarity search
│   │   └── entry_log_repo.py     # Entry log operations
│   ├── services/
│   │   ├── employee_service.py   # Employee business logic
│   │   ├── face_service.py       # Face detection & embedding
│   │   └── entry_service.py      # Entry logging logic
│   ├── routes/
│   │   ├── employees.py     # Employee API endpoints
│   │   ├── recognize.py     # Recognition endpoint
│   │   └── entry.py         # Entry management endpoints
│   └── utils/
│       └── image_utils.py   # Image validation & async storage
├── face_images/             # Stored face images (gitignored)
├── requirements.txt         # Python dependencies
├── setup_db.py             # Database initialization script
├── .env                    # Environment variables (gitignored)
└── README.md
```

---

## Prerequisites

- Python 3.11+
- PostgreSQL 14+ with pgvector extension
- ~2GB disk space for face recognition models (downloaded on first use)

### Installing pgvector on PostgreSQL Server

```bash
# Ubuntu/Debian
sudo apt install postgresql-16-pgvector

# macOS with Homebrew
brew install pgvector

# Or compile from source
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
make install
```

---

## Quick Start

```bash
# 1. Clone and enter directory
cd facial-api

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
source venv/bin/activate      # Linux/macOS
# OR
venv\Scripts\activate         # Windows

# 4. Install dependencies
pip install -r requirements.txt

# 5. Configure environment (edit .env file with your database credentials)

# 6. Setup database tables
python setup_db.py

# 7. Run the application
python app/main.py
```

---

## Installation & Setup (Detailed)

### Step 1: Create Virtual Environment

```bash
cd facial-api
python -m venv venv
```

### Step 2: Activate Virtual Environment

**Linux / macOS:**
```bash
source venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```

You should see `(venv)` prefix in your terminal prompt.

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages including:
- FastAPI, Uvicorn (web server)
- DeepFace, TensorFlow (face recognition)
- SQLAlchemy, asyncpg, pgvector (database)
- Pillow, OpenCV (image processing)

> **Note:** First run will download ~500MB of deep learning models.

### Step 4: Configure Environment

Create or edit the `.env` file in the project root:

```env
# Database Configuration
DB_HOST=your_db_host          # e.g., localhost or 192.168.1.100
DB_PORT=5432
DB_NAME=facial_db
DB_USER=your_username
DB_PASSWORD=your_password

# Face Recognition Settings
FACE_MATCH_THRESHOLD=0.55     # Lower = stricter matching
FACE_MODEL=ArcFace            # Best accuracy
FACE_DETECTOR=retinaface      # Most robust detector

# Application Settings
APP_HOST=0.0.0.0              # Listen on all interfaces
APP_PORT=8000
CORS_ORIGINS=*                # Allow all origins (change in production)

# Storage
FACE_IMAGES_DIR=face_images
```

### Step 5: Setup Database

Run the database setup script to create tables and indexes:

```bash
python setup_db.py
```

**Expected output:**
```
Connecting to database...
  ✓ Connected successfully
Creating pgvector extension...
  ✓ pgvector extension created/verified
Creating database tables...
  ✓ employees table created
  ✓ face_embeddings table created
  ✓ entry_logs table created
Creating indexes...
  ✓ Vector similarity index created
Setup complete!
```

### Step 6: Run the Application

**Option 1: Direct Python execution (Recommended)**
```bash
python app/main.py
```

**Option 2: Using Uvicorn directly**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Option 3: Production mode (no auto-reload)**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Step 7: Verify Installation

Once running, verify the API is working:

```bash
# Check health endpoint
curl http://localhost:8000/api/health

# Expected response:
# {"status":"healthy","service":"facial-recognition-api"}
```

**Access the API documentation:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## Running the Application

### Development Mode

```bash
# Activate virtual environment first
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows

# Run with auto-reload (restarts on code changes)
python app/main.py
```

### Production Mode

```bash
# Run with multiple workers for better performance
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

# Or use gunicorn (install separately: pip install gunicorn)
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

### Running in Background

**Linux/macOS:**
```bash
# Start in background
nohup python app/main.py > server.log 2>&1 &

# Check if running
ps aux | grep uvicorn

# Stop the server
pkill -f "uvicorn app.main:app"
```

### Using systemd (Linux Production)

Create `/etc/systemd/system/facial-api.service`:
```ini
[Unit]
Description=Facial Recognition API
After=network.target

[Service]
User=www-data
WorkingDirectory=/path/to/facial-api
Environment="PATH=/path/to/facial-api/venv/bin"
ExecStart=/path/to/facial-api/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable facial-api
sudo systemctl start facial-api
sudo systemctl status facial-api
```

---

## API Documentation

Once running, access the interactive API docs at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### API Endpoints

#### Employee Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/employees/register` | Register new employee with face image |
| POST | `/api/employees/{id}/add-face` | Add additional face image |
| GET | `/api/employees` | List all employees |
| GET | `/api/employees/{id}` | Get employee details |
| PUT | `/api/employees/{id}` | Update employee info |
| DELETE | `/api/employees/{id}` | Soft-delete employee |

#### Face Recognition

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/recognize` | Recognize employee from face image |

#### Entry Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/entry/manual` | Manual entry with employee ID |
| GET | `/api/entry/logs` | Get today's entry logs |
| GET | `/api/entry/logs/date/{date}` | Get entries for specific date |
| GET | `/api/entry/logs/employee/{id}` | Get employee's entry history |
| GET | `/api/entry/stats` | Get today's entry statistics |

---

## Usage Examples

### Register an Employee

```bash
curl -X POST "http://localhost:8000/api/employees/register" \
  -F "employee_id=EMP-001" \
  -F "name=John Doe" \
  -F "department=Engineering" \
  -F "email=john@example.com" \
  -F "face_image=@/path/to/face.jpg"
```

### Recognize a Face

```bash
curl -X POST "http://localhost:8000/api/recognize" \
  -F "face_image=@/path/to/face.jpg" \
  -F "log_entry=true"
```

**Success Response:**
```json
{
  "success": true,
  "employee_id": "EMP-001",
  "name": "John Doe",
  "department": "Engineering",
  "confidence": 0.8542,
  "entry_logged": true,
  "message": "Face recognized successfully"
}
```

**Not Recognized Response:**
```json
{
  "success": false,
  "message": "Face not recognized. Please use manual entry.",
  "suggestion": "Use POST /api/entry/manual with your employee ID"
}
```

### Manual Entry (Fallback)

```bash
curl -X POST "http://localhost:8000/api/entry/manual" \
  -H "Content-Type: application/json" \
  -d '{"employee_id": "EMP-001"}'
```

---

## Configuration

### Face Recognition Threshold

The `FACE_MATCH_THRESHOLD` controls matching strictness:

| Threshold | Behavior |
|-----------|----------|
| 0.40 | Very strict - fewer false positives, may miss valid matches |
| 0.55 | Balanced (default) - good accuracy with reasonable tolerance |
| 0.70 | Lenient - more matches, higher false positive risk |

### Face Detector Options

| Detector | Speed | Accuracy | Best For |
|----------|-------|----------|----------|
| `retinaface` | Medium | Highest | Production (default) |
| `mtcnn` | Fast | High | Good lighting conditions |
| `opencv` | Fastest | Medium | Resource-constrained environments |

### Face Model Options

| Model | Embedding Size | Accuracy |
|-------|----------------|----------|
| `ArcFace` | 512 | Highest (default) |
| `Facenet512` | 512 | Very High |
| `VGG-Face` | 2622 | High |

---

## Performance Considerations

### Why pgvector?

- **Native PostgreSQL** - No separate vector database needed
- **ACID compliance** - Transactional consistency with employee data
- **IVFFlat indexing** - Sub-linear search time for large datasets
- **Cosine distance** - Optimized for normalized embeddings

### Scaling Recommendations

| Employees | Index Type | Expected Search Time |
|-----------|------------|---------------------|
| < 1,000 | Sequential | < 10ms |
| 1,000 - 10,000 | IVFFlat | < 20ms |
| > 10,000 | HNSW | < 5ms |

---

## Troubleshooting

### "pgvector extension not found"
Install pgvector on your PostgreSQL server. See Prerequisites section.

### "No face detected"
- Ensure the image has good lighting
- Face should be clearly visible and frontal
- Try using `retinaface` detector (more robust)

### "Face not recognized" (low confidence)
- Register additional face images with different expressions
- Adjust `FACE_MATCH_THRESHOLD` if needed
- Ensure consistent lighting between registration and recognition

### "Connection refused"
- Check if PostgreSQL is running
- Verify database credentials in `.env`
- Ensure the database exists

---

## Extending the System

The layered architecture supports easy extension:

1. **Adding new modules**: Create files in each layer (models → schemas → repositories → services → routes)
2. **Authentication**: Add `core/security.py` with JWT middleware
3. **Scaling vectors**: Switch to HNSW index or dedicated vector DB (Pinecone, Milvus)
4. **Multiple cameras**: Add device_id tracking in entry logs

---

## License

MIT License
