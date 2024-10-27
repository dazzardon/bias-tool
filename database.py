# database.py

from sqlalchemy import create_engine, Column, Integer, String, Float, Text, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
import logging
import os

logger = logging.getLogger(__name__)

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    username = Column(String, primary_key=True, unique=True, nullable=False)
    password = Column(String, nullable=False)
    preferences = Column(Text, default='{}')  # JSON/YAML string
    bias_terms = Column(Text, default='[]')  # JSON/YAML list
    propaganda_terms = Column(Text, default='[]')  # JSON/YAML list
    analysis_history = relationship("AnalysisData", back_populates="user")

class AnalysisData(Base):
    __tablename__ = 'analysis_history'
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, ForeignKey('users.username'), nullable=False)
    title = Column(String, nullable=False)
    date = Column(String, nullable=False)
    sentiment_score = Column(Float, nullable=False)
    sentiment_label = Column(String, nullable=False)
    bias_score = Column(Float, nullable=False)
    propaganda_score = Column(Float, nullable=False)
    entities = Column(Text, nullable=False)  # JSON/YAML string
    entity_sentiments = Column(Text, nullable=False)  # JSON/YAML string
    biased_sentences = Column(Text, nullable=False)  # JSON/YAML string
    propaganda_sentences = Column(Text, nullable=False)  # JSON/YAML string
    user = relationship("User", back_populates="analysis_history")

def get_session():
    """
    Create and return a new SQLAlchemy session connected to SQLite.
    """
    try:
        # Define the path for the SQLite database file
        db_path = os.path.join(os.path.dirname(__file__), 'media_bias_db.sqlite')
        engine = create_engine(f'sqlite:///{db_path}', echo=False, connect_args={"check_same_thread": False})
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        logger.info("SQLite database session created successfully.")
        return session
    except Exception as e:
        logger.error(f"Failed to create SQLite database session: {e}")
        return None
