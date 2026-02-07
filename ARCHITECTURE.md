# Face Recognition System - Technical Architecture

This document explains how the facial recognition system processes images, stores face data, and performs matching.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Image Processing Pipeline](#image-processing-pipeline)
3. [Face Embedding Generation](#face-embedding-generation)
4. [Vector Storage in PostgreSQL](#vector-storage-in-postgresql)
5. [Face Matching Algorithm](#face-matching-algorithm)
6. [Complete Flow Diagrams](#complete-flow-diagrams)

---

## System Overview

```mermaid
graph TB
    subgraph "Client Layer"
        A[Frontend Application]
    end
    
    subgraph "API Layer"
        B[FastAPI Routes]
        C[Request Validation]
    end
    
    subgraph "Service Layer"
        D[Employee Service]
        E[Face Service]
        F[Entry Service]
    end
    
    subgraph "Data Layer"
        G[Repositories]
        H[(PostgreSQL + pgvector)]
        I[File Storage]
    end
    
    subgraph "AI/ML Layer"
        J[RetinaFace Detector]
        K[ArcFace Embeddings]
    end
    
    A -->|HTTP Request| B
    B --> C
    C --> D
    C --> F
    D --> E
    D --> G
    F --> G
    E --> J
    E --> K
    G --> H
    D --> I
```

---

## Image Processing Pipeline

When an image is uploaded to the system, it goes through several processing stages before a face can be detected and converted to a vector.

### Stage 1: Image Validation

```mermaid
flowchart LR
    A[Raw Image Bytes] --> B{Content Type?}
    B -->|image/jpeg| C{File Size?}
    B -->|image/png| C
    B -->|image/webp| C
    B -->|Other| X[Reject: Invalid Type]
    
    C -->|≤ 10MB| D{Dimensions?}
    C -->|> 10MB| Y[Reject: Too Large]
    
    D -->|100-4096px| E{Can Open?}
    D -->|Outside Range| Z[Reject: Invalid Size]
    
    E -->|Yes| F[Valid Image]
    E -->|No| W[Reject: Corrupted]
```

**Validation Rules:**
| Check | Requirement |
|-------|-------------|
| Content Type | `image/jpeg`, `image/png`, `image/webp` |
| File Size | ≤ 10 MB |
| Min Dimension | 100 × 100 pixels |
| Max Dimension | 4096 × 4096 pixels |
| Format | Must be readable by PIL |

### Stage 2: Image Preprocessing

```mermaid
flowchart TD
    A[Validated Image Bytes] --> B[Load with PIL]
    B --> C{Color Mode?}
    C -->|RGBA| D[Convert to RGB]
    C -->|Grayscale| D
    C -->|RGB| E[NumPy Array]
    D --> E
    E --> F[Shape: Height × Width × 3]
    F --> G[Ready for Face Detection]
```

**Code Reference:**
```python
def bytes_to_numpy(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)  # Shape: (H, W, 3)
```

---

## Face Embedding Generation

### What is a Face Embedding?

A face embedding is a **512-dimensional numerical vector** that represents the unique features of a face. Think of it as a "fingerprint" for the face.

```mermaid
graph LR
    A[Face Image<br/>224×224×3] --> B[Deep Neural Network<br/>ArcFace CNN]
    B --> C[512-D Vector<br/>0.023, -0.156, ..., 0.234]
    
    style A fill:#e1f5fe
    style B fill:#fff3e0
    style C fill:#e8f5e9
```

### Face Detection Process (RetinaFace)

```mermaid
flowchart TD
    A[Input Image] --> B[RetinaFace CNN]
    
    subgraph "Detection Output"
        B --> C[Bounding Boxes]
        B --> D[Facial Landmarks]
        B --> E[Confidence Scores]
    end
    
    C --> F{Multiple Faces?}
    D --> G[5 Key Points:<br/>Left Eye, Right Eye,<br/>Nose, Left Mouth, Right Mouth]
    
    F -->|Yes| H[Select Largest Face]
    F -->|No| I[Use Detected Face]
    
    H --> J[Face Alignment]
    I --> J
    G --> J
    
    J --> K[Aligned Face<br/>112×112 pixels]
```

### Embedding Generation Process (ArcFace)

```mermaid
flowchart TD
    A[Aligned Face<br/>112×112×3] --> B[Normalize Pixels<br/>0-255 → -1 to 1]
    
    B --> C[ArcFace CNN]
    
    subgraph "Neural Network Layers"
        C --> D[Conv Layers<br/>Feature Extraction]
        D --> E[Residual Blocks<br/>Deep Features]
        E --> F[Global Average Pool]
        F --> G[Fully Connected<br/>512 neurons]
    end
    
    G --> H[L2 Normalization]
    H --> I[512-D Unit Vector<br/>||v|| = 1]
    
    style I fill:#c8e6c9
```

### What Each Dimension Represents

The 512 dimensions capture abstract facial features learned by the neural network:

```mermaid
graph TD
    subgraph "Example Dimensions (Conceptual)"
        A[Dim 1-50: Eye Shape & Spacing]
        B[Dim 51-100: Nose Structure]
        C[Dim 101-150: Mouth Shape]
        D[Dim 151-200: Face Contour]
        E[Dim 201-300: Skin Texture Patterns]
        F[Dim 301-400: Facial Proportions]
        G[Dim 401-512: Complex Combinations]
    end
```

> **Note:** These are conceptual groupings. In reality, each dimension captures a complex combination of features learned during training on millions of faces.

---

## Vector Storage in PostgreSQL

### Database Schema

```mermaid
erDiagram
    EMPLOYEES ||--o{ FACE_EMBEDDINGS : has
    EMPLOYEES ||--o{ ENTRY_LOGS : creates
    
    EMPLOYEES {
        int id PK
        string employee_id UK
        string name
        string department
        string email
        string phone
        boolean is_active
        timestamp created_at
        timestamp updated_at
    }
    
    FACE_EMBEDDINGS {
        int id PK
        int employee_id FK
        vector_512 embedding
        string image_path
        boolean is_primary
        timestamp created_at
    }
    
    ENTRY_LOGS {
        int id PK
        int employee_id FK
        string entry_method
        float match_confidence
        string device_id
        timestamp entry_time
    }
```

### How pgvector Stores Vectors

```mermaid
flowchart TD
    A[512-D Float Array] --> B[pgvector VECTOR type]
    
    subgraph "Storage Format"
        B --> C["VECTOR(512)"]
        C --> D[Binary Format:<br/>4 bytes × 512 = 2KB per vector]
    end
    
    subgraph "Index Structure (IVFFlat)"
        E[All Vectors] --> F[Cluster into 100 Lists]
        F --> G[List 1: Similar Vectors]
        F --> H[List 2: Similar Vectors]
        F --> I[...]
        F --> J[List 100: Similar Vectors]
    end
    
    D --> E
```

### SQL Operations

**Storing an Embedding:**
```sql
INSERT INTO face_embeddings (employee_id, embedding, image_path, is_primary)
VALUES (
    1,
    '[0.023, -0.156, 0.089, ..., 0.234]'::vector,
    'face_images/EMP-001/primary_abc123.jpg',
    true
);
```

**Creating the Index:**
```sql
CREATE INDEX face_embedding_idx 
ON face_embeddings 
USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);
```

---

## Face Matching Algorithm

### Cosine Distance Explained

```mermaid
graph TD
    subgraph "Vector Space Visualization"
        A[Origin] --> B[Vector A: Registered Face]
        A --> C[Vector B: Input Face]
        B -.->|θ = angle| C
    end
    
    subgraph "Distance Calculation"
        D["Cosine Similarity = cos(θ) = (A · B) / (||A|| × ||B||)"]
        E["Cosine Distance = 1 - Cosine Similarity"]
    end
    
    subgraph "Interpretation"
        F["Distance = 0.0 → Identical (θ = 0°)"]
        G["Distance = 0.3 → Very Similar (θ ≈ 45°)"]
        H["Distance = 1.0 → Orthogonal (θ = 90°)"]
        I["Distance = 2.0 → Opposite (θ = 180°)"]
    end
```

### Distance to Confidence Conversion

```mermaid
flowchart LR
    A[Cosine Distance] --> B{Distance Value}
    
    B -->|0.0 - 0.2| C[Very High Confidence<br/>80-100%]
    B -->|0.2 - 0.4| D[High Confidence<br/>60-80%]
    B -->|0.4 - 0.55| E[Moderate Confidence<br/>45-60%]
    B -->|> 0.55| F[No Match<br/>Below Threshold]
    
    style C fill:#c8e6c9
    style D fill:#dcedc8
    style E fill:#fff9c4
    style F fill:#ffcdd2
```

**Conversion Formula:**
```
Confidence = 1 - Distance

Examples:
- Distance 0.15 → Confidence 85%
- Distance 0.30 → Confidence 70%
- Distance 0.50 → Confidence 50%
```

### Matching Query Process

```mermaid
sequenceDiagram
    participant App as Application
    participant PG as PostgreSQL
    participant Idx as IVFFlat Index
    
    App->>PG: Search for similar vectors
    Note over App,PG: SELECT ... WHERE embedding <=> input < 0.55
    
    PG->>Idx: Find nearest cluster
    Idx->>Idx: Scan vectors in cluster
    Idx->>PG: Return candidates with distances
    
    PG->>PG: Filter by threshold (< 0.55)
    PG->>PG: Order by distance ASC
    PG->>App: Return best match
```

**SQL Query:**
```sql
SELECT 
    fe.id,
    e.employee_id,
    e.name,
    fe.embedding <=> '[input_vector]'::vector AS distance
FROM face_embeddings fe
JOIN employees e ON fe.employee_id = e.id
WHERE e.is_active = true
  AND fe.embedding <=> '[input_vector]'::vector < 0.55
ORDER BY distance ASC
LIMIT 1;
```

---

## Complete Flow Diagrams

### Employee Registration Flow

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI
    participant VS as Validation Service
    participant FS as Face Service
    participant DB as PostgreSQL
    participant Disk as File System
    
    Client->>API: POST /api/employees/register<br/>(employee_id, name, face_image)
    
    API->>VS: Validate image
    VS->>VS: Check type, size, dimensions
    VS-->>API: Valid image bytes
    
    API->>FS: Detect and embed face
    FS->>FS: RetinaFace: Detect face
    FS->>FS: Align face to 112×112
    FS->>FS: ArcFace: Generate 512-D vector
    FS-->>API: Embedding vector
    
    API->>DB: Check employee_id exists
    DB-->>API: Not exists ✓
    
    API->>DB: INSERT employee record
    DB-->>API: Employee ID: 1
    
    API->>Disk: Save image (async)
    Note over Disk: face_images/EMP-001/primary_xxx.jpg
    
    API->>DB: INSERT face_embedding<br/>(employee_id=1, embedding=[...])
    
    API-->>Client: 201 Created<br/>{employee_id, name, face_count: 1}
```

### Face Recognition Flow

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI
    participant FS as Face Service
    participant Repo as FaceEmbeddingRepo
    participant DB as PostgreSQL
    
    Client->>API: POST /api/recognize<br/>(face_image)
    
    API->>FS: Detect and embed face
    FS->>FS: RetinaFace: Detect face
    
    alt No Face Detected
        FS-->>API: FaceNotDetectedException
        API-->>Client: 400 "No face detected"
    end
    
    FS->>FS: ArcFace: Generate embedding
    FS-->>API: Input embedding [512 floats]
    
    API->>Repo: Find similar faces
    Repo->>DB: Vector similarity search
    
    Note over DB: SELECT ... WHERE<br/>embedding <=> input < 0.55<br/>ORDER BY distance LIMIT 1
    
    DB-->>Repo: Best match (or empty)
    
    alt Match Found (distance < 0.55)
        Repo-->>API: Employee + distance
        API->>API: Calculate confidence<br/>confidence = 1 - distance
        API-->>Client: 200 {success: true,<br/>employee_id, confidence}
    else No Match
        Repo-->>API: None
        API-->>Client: 200 {success: false,<br/>suggestion: "Use manual entry"}
    end
```

### Manual Entry Fallback Flow

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI
    participant ES as Entry Service
    participant DB as PostgreSQL
    
    Client->>API: POST /api/entry/manual<br/>{employee_id: "EMP-001"}
    
    API->>ES: Process manual entry
    ES->>DB: Find employee by ID
    
    alt Employee Not Found
        DB-->>ES: None
        ES-->>API: NotFoundException
        API-->>Client: 404 "Employee not found"
    end
    
    DB-->>ES: Employee record
    
    ES->>DB: INSERT entry_log<br/>(employee_id, method="manual")
    DB-->>ES: Entry logged
    
    ES-->>API: Entry record
    API-->>Client: 200 {success: true,<br/>employee_id, entry_method: "manual"}
```

---

## Performance Characteristics

### Time Complexity

```mermaid
graph TD
    subgraph "Without Index"
        A[Linear Scan: O(n)]
        A --> B[1000 faces ≈ 50ms]
        A --> C[10000 faces ≈ 500ms]
        A --> D[100000 faces ≈ 5s]
    end
    
    subgraph "With IVFFlat Index"
        E[Approximate: O(√n)]
        E --> F[1000 faces ≈ 5ms]
        E --> G[10000 faces ≈ 15ms]
        E --> H[100000 faces ≈ 50ms]
    end
    
    style E fill:#c8e6c9
    style F fill:#c8e6c9
    style G fill:#c8e6c9
    style H fill:#c8e6c9
```

### Memory Usage

| Component | Size |
|-----------|------|
| Single embedding | 512 × 4 bytes = 2 KB |
| 1,000 employees | ~2 MB |
| 10,000 employees | ~20 MB |
| IVFFlat index overhead | ~10-20% |

---

## Summary

```mermaid
graph LR
    A[Image Upload] --> B[Validation]
    B --> C[Face Detection<br/>RetinaFace]
    C --> D[Embedding<br/>ArcFace 512-D]
    D --> E[Vector Storage<br/>pgvector]
    E --> F[Similarity Search<br/>Cosine Distance]
    F --> G[Match Result]
    
    style A fill:#e3f2fd
    style B fill:#e3f2fd
    style C fill:#fff3e0
    style D fill:#fff3e0
    style E fill:#e8f5e9
    style F fill:#e8f5e9
    style G fill:#f3e5f5
```

**Key Takeaways:**

1. **Face Detection**: RetinaFace locates and aligns faces in images
2. **Embedding**: ArcFace converts faces to 512-dimensional vectors
3. **Storage**: pgvector stores vectors with efficient indexing
4. **Matching**: Cosine distance measures similarity (lower = more similar)
5. **Threshold**: 0.55 balances accuracy and false positive rate
