import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.models import Base, Organism, Species, Resource, SimulationState, Event  # noqa


DATABASE_LOCATION = os.getenv("DATABASE_LOCATION")

engine = create_engine(
    DATABASE_LOCATION,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_LOCATION else {}
)


Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)


def get_session():
    """
    Dependency for providing a database session.
    Use this in API routes or services to interact with the database with fast api.
    """
    db = Session()
    try:
        yield db
    finally:
        db.close()
