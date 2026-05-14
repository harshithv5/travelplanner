import os
from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, String, Text, DateTime,
    create_engine, desc,
)
from sqlalchemy.orm import DeclarativeBase, Session


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def _engine():
    url = os.getenv("POSTGRES_CONNECTION_STRING", "")
    return create_engine(url, pool_pre_ping=True)


engine = _engine()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class UserPreference(Base):
    __tablename__ = "user_preferences"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    user_id    = Column(String(128), nullable=False, index=True)
    preference = Column(Text, nullable=False)


class ConversationMemory(Base):
    __tablename__ = "conversation_memory"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    user_id         = Column(String(128), nullable=False, index=True)
    session_id      = Column(String(128), nullable=False, index=True)
    user_question   = Column(Text, nullable=False)
    system_response = Column(Text, nullable=False)
    timestamp       = Column(DateTime(timezone=True), nullable=False,
                             default=lambda: datetime.now(timezone.utc))


# Create tables if they don't exist yet
Base.metadata.create_all(engine)


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------

def add_user_preference(user_id: str, preference: str) -> UserPreference:
    """Insert a preference entry for a user."""
    with Session(engine) as session:
        row = UserPreference(user_id=user_id, preference=preference)
        session.add(row)
        session.commit()
        session.refresh(row)
        return row


def save_conversation(
    user_id: str,
    session_id: str,
    user_question: str,
    system_response: str,
) -> ConversationMemory:
    """Append a conversation turn to the history."""
    with Session(engine) as session:
        row = ConversationMemory(
            user_id=user_id,
            session_id=session_id,
            user_question=user_question,
            system_response=system_response,
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return row


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------

def get_user_preferences(user_id: str) -> list[UserPreference]:
    """Return all preference entries for a user."""
    with Session(engine) as session:
        return (
            session.query(UserPreference)
            .filter(UserPreference.user_id == user_id)
            .all()
        )


def get_conversations(
    user_id: str,
    session_id: str,
    limit: int = 20,
) -> list[ConversationMemory]:
    """
    Return the most recent `limit` conversation turns for a user+session,
    ordered oldest-first so they read naturally.
    """
    with Session(engine) as session:
        rows = (
            session.query(ConversationMemory)
            .filter(
                ConversationMemory.user_id   == user_id,
                ConversationMemory.session_id == session_id,
            )
            .order_by(desc(ConversationMemory.timestamp))
            .limit(limit)
            .all()
        )
        return list(reversed(rows))
