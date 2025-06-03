"""Database models for conversation persistence."""

from enum import Enum
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Table,
    Text,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.sql import func

Base = declarative_base()


class MessageRole(str, Enum):
    """Message roles."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ConversationStatus(str, Enum):
    """Conversation status."""

    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


class MessageStatus(str, Enum):
    """Message status."""

    PENDING = "pending"
    SENT = "sent"
    RECEIVED = "received"
    ERROR = "error"
    EDITED = "edited"
    DELETED = "deleted"


# Many-to-many association tables
conversation_tags = Table(
    "conversation_tags",
    Base.metadata,
    Column("conversation_id", String, ForeignKey("conversations.id")),
    Column("tag_id", Integer, ForeignKey("tags.id")),
)

message_reactions = Table(
    "message_reactions",
    Base.metadata,
    Column("message_id", String, ForeignKey("messages.id")),
    Column("reaction_id", Integer, ForeignKey("reactions.id")),
    Column("user_id", String),
    Column("created_at", DateTime, default=func.now()),
)


class Conversation(Base):
    """Conversation model."""

    __tablename__ = "conversations"

    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    status = Column(String, default=ConversationStatus.ACTIVE)

    # Metadata
    model = Column(String)
    provider = Column(String)
    temperature = Column(Float)
    max_tokens = Column(Integer)
    total_tokens = Column(Integer, default=0)
    total_cost = Column(Float, default=0.0)

    # Session info
    session_data = Column(JSON)
    parent_id = Column(String, ForeignKey("conversations.id"))

    # Relationships
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    tags = relationship("Tag", secondary=conversation_tags, back_populates="conversations")
    parent = relationship("Conversation", remote_side=[id], overlaps="children")
    children = relationship("Conversation", overlaps="parent")

    # Indexes
    __table_args__ = (
        Index("idx_conversation_created", "created_at"),
        Index("idx_conversation_status", "status"),
        Index("idx_conversation_updated", "updated_at"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "status": self.status,
            "model": self.model,
            "provider": self.provider,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "message_count": len(self.messages),
            "tags": [tag.name for tag in self.tags],
        }


class Message(Base):
    """Message model."""

    __tablename__ = "messages"

    id = Column(String, primary_key=True)
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    status = Column(String, default=MessageStatus.SENT)

    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    edited_at = Column(DateTime)

    # Metadata
    model = Column(String)
    tokens = Column(Integer)
    cost = Column(Float)
    message_metadata = Column(JSON)

    # Threading
    parent_message_id = Column(String, ForeignKey("messages.id"))
    thread_position = Column(Integer)

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    parent_message = relationship("Message", remote_side=[id], overlaps="replies")
    replies = relationship("Message", overlaps="parent_message")
    reactions = relationship("Reaction", secondary=message_reactions, back_populates="messages")
    annotations = relationship("Annotation", back_populates="message", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index("idx_message_conversation", "conversation_id"),
        Index("idx_message_created", "created_at"),
        Index("idx_message_role", "role"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "role": self.role,
            "content": self.content,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "edited_at": self.edited_at.isoformat() if self.edited_at else None,
            "model": self.model,
            "tokens": self.tokens,
            "cost": self.cost,
            "metadata": self.message_metadata,
            "parent_message_id": self.parent_message_id,
            "reactions": [r.emoji for r in self.reactions],
            "annotation_count": len(self.annotations),
        }


class Tag(Base):
    """Tag model for categorizing conversations."""

    __tablename__ = "tags"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    color = Column(String)
    description = Column(Text)
    created_at = Column(DateTime, default=func.now())

    # Relationships
    conversations = relationship("Conversation", secondary=conversation_tags, back_populates="tags")

    # Indexes
    __table_args__ = (Index("idx_tag_name", "name"),)


class Reaction(Base):
    """Reaction model for messages."""

    __tablename__ = "reactions"

    id = Column(Integer, primary_key=True)
    emoji = Column(String, nullable=False)
    name = Column(String, nullable=False)

    # Relationships
    messages = relationship("Message", secondary=message_reactions, back_populates="reactions")


class Annotation(Base):
    """Annotation model for message notes."""

    __tablename__ = "annotations"

    id = Column(String, primary_key=True)
    message_id = Column(String, ForeignKey("messages.id"), nullable=False)
    content = Column(Text, nullable=False)
    user_id = Column(String)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Position in message
    start_index = Column(Integer)
    end_index = Column(Integer)

    # Relationships
    message = relationship("Message", back_populates="annotations")

    # Indexes
    __table_args__ = (Index("idx_annotation_message", "message_id"),)


class ConversationTemplate(Base):
    """Template model for conversation starters."""

    __tablename__ = "conversation_templates"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    system_prompt = Column(Text)
    initial_messages = Column(JSON)
    default_model = Column(String)
    default_temperature = Column(Float)
    tags = Column(JSON)
    created_at = Column(DateTime, default=func.now())
    is_public = Column(Boolean, default=True)

    # Indexes
    __table_args__ = (Index("idx_template_name", "name"),)


class Analytics(Base):
    """Analytics tracking for conversations."""

    __tablename__ = "analytics"

    id = Column(Integer, primary_key=True)
    conversation_id = Column(String, ForeignKey("conversations.id"))
    event_type = Column(String, nullable=False)
    event_data = Column(JSON)
    timestamp = Column(DateTime, default=func.now())

    # Metrics
    response_time_ms = Column(Integer)
    tokens_used = Column(Integer)
    cost = Column(Float)

    # Indexes
    __table_args__ = (
        Index("idx_analytics_conversation", "conversation_id"),
        Index("idx_analytics_timestamp", "timestamp"),
        Index("idx_analytics_event", "event_type"),
    )


# Database initialization
def init_database(db_path: str = "sqlite:///neuromancer.db"):
    """Initialize the database."""
    engine = create_engine(db_path, echo=False)
    Base.metadata.create_all(engine)

    # Create session factory
    Session = sessionmaker(bind=engine)

    # Add default reactions
    session = Session()
    default_reactions = [
        ("[+]", "thumbs_up"),
        ("[-]", "thumbs_down"),
        ("[HEART]", "heart"),
        ("[CHECK]", "accurate"),
        ("[?]", "thinking"),
        ("[*]", "helpful"),
        ("[!]", "fast"),
        ("[X]", "bug"),
    ]

    for emoji, name in default_reactions:
        if not session.query(Reaction).filter_by(emoji=emoji).first():
            reaction = Reaction(emoji=emoji, name=name)
            session.add(reaction)

    session.commit()
    session.close()

    return engine, Session
