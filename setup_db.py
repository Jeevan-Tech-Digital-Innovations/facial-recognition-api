#!/usr/bin/env python3
"""
Database setup script for Facial Recognition API.

This script:
1. Creates the pgvector extension if not exists
2. Creates all database tables
3. Creates necessary indexes for performance

Run this script before starting the application for the first time.

Usage:
    python setup_db.py
"""

import sys
from urllib.parse import quote_plus

from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError, ProgrammingError

# Load settings
from dotenv import load_dotenv
import os

load_dotenv()

# Build database URL with encoded password
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "facial_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

encoded_password = quote_plus(DB_PASSWORD)
DATABASE_URL = f"postgresql://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def create_extension(engine):
    """Create pgvector extension."""
    print("Creating pgvector extension...")
    try:
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
        print("  ✓ pgvector extension created/verified")
        return True
    except Exception as e:
        error_msg = str(e)
        print(f"  ✗ Failed to create pgvector extension")
        print(f"    Error: {error_msg[:200]}")
        print("\n  ⚠ pgvector must be installed on the PostgreSQL server.")
        print("  Ask your database administrator to install it:")
        print("    - Ubuntu/Debian: sudo apt install postgresql-17-pgvector")
        print("    - Or compile from: https://github.com/pgvector/pgvector")
        print("\n  The API will NOT work for face recognition without pgvector.")
        return False


def create_tables(engine):
    """Create all database tables."""
    print("\nCreating database tables...")

    # Import Base and all models to register them
    from app.core.database import Base
    from app.models import Employee, FaceEmbedding, EntryLog  # noqa: F401

    try:
        Base.metadata.create_all(bind=engine)
        print("  ✓ All tables created successfully")
        return True
    except Exception as e:
        print(f"  ✗ Failed to create tables: {e}")
        return False


def create_indexes(engine):
    """Create additional indexes for performance."""
    print("\nCreating indexes...")

    indexes = [
        # HNSW index for face embedding similarity search.
        # HNSW is preferred over IVFFlat because:
        #   - Works correctly on empty tables (IVFFlat needs data for k-means)
        #   - Better recall at all dataset sizes
        #   - No need to tune 'lists' parameter based on row count
        # m=16: number of bi-directional links per node (default, good for 512-D)
        # ef_construction=64: build-time search width (higher = better recall, slower build)
        """
        CREATE INDEX IF NOT EXISTS idx_face_embeddings_vector 
        ON face_embeddings 
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        """,
        # Index for entry logs by date
        """
        CREATE INDEX IF NOT EXISTS idx_entry_logs_entry_time 
        ON entry_logs (entry_time DESC)
        """,
        # Index for entry logs by employee and date
        """
        CREATE INDEX IF NOT EXISTS idx_entry_logs_employee_time 
        ON entry_logs (employee_id, entry_time DESC)
        """,
        # Index for unsynced entry logs (payroll sync queue)
        """
        CREATE INDEX IF NOT EXISTS idx_entry_logs_payroll_synced 
        ON entry_logs (payroll_synced) WHERE payroll_synced = false
        """,
    ]

    success = True
    for idx_sql in indexes:
        try:
            with engine.connect() as conn:
                conn.execute(text(idx_sql))
                conn.commit()
            # Extract index name for logging
            idx_name = idx_sql.split("IF NOT EXISTS")[1].split("ON")[0].strip()
            print(f"  \u2713 Index {idx_name} created/verified")
        except Exception as e:
            print(f"  \u2717 Failed to create index: {e}")
            success = False

    return success


def migrate_schema(engine):
    """Apply incremental schema migrations for new columns."""
    print("\nApplying schema migrations...")

    migrations = [
        # Add payroll_emp_no to employees table
        ("employees.payroll_emp_no", """
            ALTER TABLE employees 
            ADD COLUMN IF NOT EXISTS payroll_emp_no VARCHAR(50)
        """),
        ("employees.payroll_emp_no_unique", """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_indexes 
                    WHERE indexname = 'ix_employees_payroll_emp_no'
                ) THEN
                    CREATE UNIQUE INDEX ix_employees_payroll_emp_no 
                    ON employees (payroll_emp_no) WHERE payroll_emp_no IS NOT NULL;
                END IF;
            END $$
        """),
        # Add payroll sync columns to entry_logs
        ("entry_logs.payroll_synced", """
            ALTER TABLE entry_logs 
            ADD COLUMN IF NOT EXISTS payroll_synced BOOLEAN DEFAULT FALSE
        """),
        ("entry_logs.payroll_punch_id", """
            ALTER TABLE entry_logs 
            ADD COLUMN IF NOT EXISTS payroll_punch_id BIGINT
        """),
        ("entry_logs.sync_error", """
            ALTER TABLE entry_logs 
            ADD COLUMN IF NOT EXISTS sync_error TEXT
        """),
    ]

    success = True
    for name, sql in migrations:
        try:
            with engine.connect() as conn:
                conn.execute(text(sql))
                conn.commit()
            print(f"  \u2713 Migration {name} applied")
        except Exception as e:
            error_msg = str(e)
            if "already exists" in error_msg.lower() or "duplicate" in error_msg.lower():
                print(f"  \u2713 Migration {name} already applied")
            else:
                print(f"  \u2717 Migration {name} failed: {e}")
                success = False

    return success


def verify_setup(engine):
    """Verify the database setup."""
    print("\nVerifying setup...")

    checks = [
        ("pgvector extension", "SELECT extname FROM pg_extension WHERE extname = 'vector'"),
        ("employees table", "SELECT 1 FROM employees LIMIT 1"),
        ("face_embeddings table", "SELECT 1 FROM face_embeddings LIMIT 1"),
        ("entry_logs table", "SELECT 1 FROM entry_logs LIMIT 1"),
    ]

    all_passed = True
    for name, query in checks:
        try:
            with engine.connect() as conn:
                conn.execute(text(query))
            print(f"  ✓ {name} exists")
        except Exception:
            print(f"  ✗ {name} not found")
            all_passed = False

    return all_passed


def main():
    """Main setup function."""
    print("=" * 60)
    print("Facial Recognition API - Database Setup")
    print("=" * 60)
    print(f"\nConnecting to: {DB_HOST}:{DB_PORT}/{DB_NAME}")
    print(f"User: {DB_USER}")
    print()

    # Create engine
    try:
        engine = create_engine(DATABASE_URL, echo=False)
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✓ Database connection successful\n")
    except OperationalError as e:
        print(f"✗ Failed to connect to database: {e}")
        print("\nPlease check:")
        print("  1. Database server is running")
        print("  2. Database credentials in .env are correct")
        print("  3. Database exists")
        sys.exit(1)

    # Run setup steps
    pgvector_ok = create_extension(engine)
    
    if not pgvector_ok:
        print("\n" + "=" * 60)
        print("⚠ CRITICAL: pgvector extension is NOT installed on the server")
        print("=" * 60)
        print("\nThe face recognition API requires pgvector for vector similarity search.")
        print("Please contact your database administrator to install pgvector on:")
        print(f"  Server: {DB_HOST}")
        print(f"  PostgreSQL version: 17 (based on error message)")
        print("\nOnce pgvector is installed, run this script again.")
        print("\nWould you like to continue creating tables anyway? (for testing structure)")
        
        try:
            response = input("Continue? [y/N]: ").strip().lower()
            if response != 'y':
                print("\nSetup aborted. Install pgvector and try again.")
                return 1
        except (EOFError, KeyboardInterrupt):
            print("\nSetup aborted.")
            return 1
    
    steps = [
        ("database tables", lambda: create_tables(engine)),
        ("schema migrations", lambda: migrate_schema(engine)),
        ("indexes", lambda: create_indexes(engine) if pgvector_ok else True),
        ("verification", lambda: verify_setup(engine)),
    ]

    all_success = pgvector_ok
    for step_name, step_func in steps:
        if not step_func():
            all_success = False
            print(f"\n⚠ Warning: {step_name} setup had issues")

    print("\n" + "=" * 60)
    if all_success:
        print("✓ Database setup completed successfully!")
        print("\nYou can now start the API with:")
        print("  uvicorn app.main:app --reload")
    else:
        print("⚠ Database setup completed with warnings")
        print("Please review the issues above before starting the API")
    print("=" * 60)

    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
