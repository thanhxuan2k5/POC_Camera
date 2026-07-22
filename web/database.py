import os
import logging
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, LargeBinary, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from datetime import datetime

logger = logging.getLogger(__name__)

DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'db'))
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, 'inspection.db')
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Camera(Base):
    __tablename__ = "cameras"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    rtsp_url = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    events = relationship("Event", back_populates="camera", cascade="all, delete-orphan")

class Model(Base):
    __tablename__ = "models"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    type = Column(String) # 'detector' or 'embedder'
    file_path = Column(String)
    format = Column(String) # 'pt', 'onnx', 'engine'
    is_active = Column(Boolean, default=False)
    metadata_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class TrainingSession(Base):
    __tablename__ = "training_sessions"
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("models.id"))
    status = Column(String, default="pending") # pending, running, completed, failed
    num_images = Column(Integer, default=0)
    threshold = Column(Float, default=0.0)
    accuracy = Column(Float, default=0.0)
    log_text = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    model = relationship("Model")
    references = relationship("ReferenceEmbedding", back_populates="session", cascade="all, delete-orphan")

class ReferenceEmbedding(Base):
    __tablename__ = "reference_embeddings"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("training_sessions.id"))
    image_path = Column(String)
    embedding_blob = Column(LargeBinary)
    created_at = Column(DateTime, default=datetime.utcnow)
    session = relationship("TrainingSession", back_populates="references")

class Event(Base):
    __tablename__ = "events"
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"))
    token_track_id = Column(String, index=True)
    result = Column(String) # 'OK' or 'NG'
    confidence = Column(Float)
    similarity_score = Column(Float, nullable=True)
    image_path = Column(String)
    thumbnail_path = Column(String, nullable=True)
    occurred_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    camera = relationship("Camera", back_populates="events")

class Setting(Base):
    __tablename__ = "settings"
    key = Column(String, primary_key=True, index=True)
    value = Column(String)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    logger.info(f"Initializing database at {DB_PATH}")
    Base.metadata.create_all(bind=engine)
