from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    Float,
    Index,
    JSON,
    String,
    UniqueConstraint
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped, mapped_column, relationship



class Base(DeclarativeBase):
    def __repr__(self):
        attrs = ", ".join([f"{c.name}={getattr(self, c.name)!r}" for c in self.__table__.columns])
        return f"<{self.__class__.__name__}({attrs})>"


# shared beteen frontend and backend. nn weights can be saved in a filesystem and identified with id.
class Organism(Base):
    __tablename__ = "organisms"

    id: Mapped[int] = mapped_column(primary_key=True)
    # name: Mapped[str] = mapped_column(String)
    horizon: Mapped[int] = mapped_column(Integer)
    energy: Mapped[int] = mapped_column(Integer)
    species: Mapped[int] = mapped_column(Integer, ForeignKey("species.id"))
    # traits: Mapped[dict] = mapped_column(JSON, default={})
    fitness: Mapped[float] = mapped_column(default=0.0)
    alive: Mapped[bool] = mapped_column(Boolean, default=True)
    nn_weights_path: Mapped[str] = mapped_column(String, nullable=True)  # or use a separate table
    created_at: Mapped[str] = mapped_column(DateTime)
    updated_at: Mapped[str] = mapped_column(DateTime)


class Species(Base):
    __tablename__ = "species"

    id: Mapped[int] = mapped_column(primary_key=True)
    parent_id: Mapped[int] = mapped_column(Integer, ForeignKey("species.id"), nullable=True)
    nn_arch_signature: Mapped[str] = mapped_column(String)  # e.g., hash of NN structure
    created_at: Mapped[str] = mapped_column(DateTime)
    color: Mapped[str] = mapped_column(String, nullable=True)  # for visualization

    organisms: Mapped[list["Organism"]] = relationship("Organism", back_populates="species")
    children: Mapped[list["Species"]] = relationship("Species", back_populates="parent", remote_side=[id])
    parent: Mapped["Species"] = relationship("Species", back_populates="children", remote_side=[parent_id])

    def get_lineage(self):
        """Returns the Ancestry of the Species, back to LUCA (Last Universal Common Ancestor)"""
        lineage = []
        node = self
        while node:
            lineage.append(node)
            node = node.parent
        return lineage[::-1]  # from root to self

class Resource(Base):
    __tablename__ = "resources"
    id: Mapped[int] = mapped_column(primary_key=True)
    type: Mapped[str] = mapped_column(String)
    x: Mapped[int] = mapped_column(Integer)
    y: Mapped[int] = mapped_column(Integer)
    quantity: Mapped[int] = mapped_column(Integer)
    refresh_rate: Mapped[float] = mapped_column(Float)


class SimulationState(Base):
    __tablename__ = "simulation_state"
    id: Mapped[int] = mapped_column(primary_key=True)
    current_step: Mapped[int] = mapped_column(Integer)
    config: Mapped[dict] = mapped_column(JSON)


class Event(Base):
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(primary_key=True)
    step: Mapped[int] = mapped_column(Integer)
    event_type: Mapped[str] = mapped_column(String)
    data: Mapped[dict] = mapped_column(JSON)

