"""Enterprise-grade session management for conversations."""

import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import and_, desc, func, or_
from sqlalchemy.orm import Session

from src.core.models import (
    Analytics,
    Conversation,
    ConversationStatus,
    ConversationTemplate,
    Message,
    MessageRole,
    MessageStatus,
    Tag,
    init_database,
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ConversationSession:
    """Manages a single conversation session."""

    def __init__(self, session_id: str, db_session: Session, conversation: Conversation):
        self.session_id = session_id
        self.db_session = db_session
        self.conversation = conversation
        self._lock = asyncio.Lock()
        self._message_buffer = []
        self._analytics_buffer = []

    @property
    def id(self) -> str:
        return self.conversation.id

    @property
    def title(self) -> str:
        return self.conversation.title

    async def add_message(
        self,
        role: MessageRole,
        content: str,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        """Add a message to the conversation."""
        async with self._lock:
            message = Message(
                id=str(uuid.uuid4()),
                conversation_id=self.conversation.id,
                role=role,
                content=content,
                model=model,
                metadata=metadata or {},
            )

            self.db_session.add(message)
            self._message_buffer.append(message)

            # Update conversation
            self.conversation.updated_at = datetime.now()

            # Batch commit for performance
            if len(self._message_buffer) >= 5:
                await self._flush_messages()

            return message

    async def update_message(
        self,
        message_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Update an existing message."""
        async with self._lock:
            message = self.db_session.query(Message).filter_by(id=message_id).first()

            if not message:
                return False

            if content is not None:
                message.content = content
                message.edited_at = datetime.now()
                message.status = MessageStatus.EDITED

            if metadata is not None:
                message.message_metadata = {**(message.message_metadata or {}), **metadata}

            self.db_session.commit()
            return True

    async def get_messages(
        self, limit: int | None = None, offset: int = 0, include_deleted: bool = False
    ) -> list[Message]:
        """Get messages from the conversation."""
        query = self.db_session.query(Message).filter_by(conversation_id=self.conversation.id)

        if not include_deleted:
            query = query.filter(Message.status != MessageStatus.DELETED)

        query = query.order_by(Message.created_at)

        if limit:
            query = query.limit(limit).offset(offset)

        return query.all()

    async def search_messages(self, query: str, role: MessageRole | None = None) -> list[Message]:
        """Search messages in the conversation."""
        search_query = self.db_session.query(Message).filter(
            and_(Message.conversation_id == self.conversation.id, Message.content.contains(query))
        )

        if role:
            search_query = search_query.filter(Message.role == role)

        return search_query.order_by(desc(Message.created_at)).all()

    async def add_reaction(self, message_id: str, reaction_emoji: str, user_id: str) -> bool:
        """Add a reaction to a message."""
        # TODO: Implement reaction logic
        pass

    async def add_annotation(
        self,
        message_id: str,
        content: str,
        start_index: int | None = None,
        end_index: int | None = None,
    ) -> bool:
        """Add an annotation to a message."""
        # TODO: Implement annotation logic
        pass

    async def branch_conversation(
        self, from_message_id: str, title: str | None = None
    ) -> "ConversationSession":
        """Create a branch from a specific message."""
        # TODO: Implement branching logic
        pass

    async def update_metrics(self, tokens_used: int, cost: float, response_time_ms: int):
        """Update conversation metrics."""
        async with self._lock:
            self.conversation.total_tokens += tokens_used
            self.conversation.total_cost += cost

            # Add analytics
            analytics = Analytics(
                conversation_id=self.conversation.id,
                event_type="completion",
                tokens_used=tokens_used,
                cost=cost,
                response_time_ms=response_time_ms,
            )

            self._analytics_buffer.append(analytics)

            # Batch analytics
            if len(self._analytics_buffer) >= 10:
                await self._flush_analytics()

    async def _flush_messages(self):
        """Flush message buffer to database."""
        if self._message_buffer:
            self.db_session.commit()
            self._message_buffer.clear()

    async def _flush_analytics(self):
        """Flush analytics buffer to database."""
        if self._analytics_buffer:
            self.db_session.add_all(self._analytics_buffer)
            self.db_session.commit()
            self._analytics_buffer.clear()

    async def close(self):
        """Close the session and flush buffers."""
        await self._flush_messages()
        await self._flush_analytics()

    def __repr__(self):
        return f"<ConversationSession {self.session_id}: {self.title}>"


class SessionManager:
    """Enterprise session manager for conversations."""

    def __init__(self, db_path: str = "sqlite:///neuromancer.db"):
        """Initialize session manager."""
        self.engine, self.SessionFactory = init_database(db_path)
        self._active_sessions: dict[str, ConversationSession] = {}
        self._session_timeout = timedelta(hours=24)
        self._cleanup_task = None

    async def start(self):
        """Start the session manager."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Session manager started")

    async def stop(self):
        """Stop the session manager."""
        if self._cleanup_task:
            self._cleanup_task.cancel()

        # Close all active sessions
        for session in self._active_sessions.values():
            await session.close()

        self._active_sessions.clear()
        logger.info("Session manager stopped")

    @asynccontextmanager
    async def create_session(
        self, title: str | None = None, template_id: int | None = None
    ) -> ConversationSession:
        """Create a new conversation session."""
        db_session = self.SessionFactory()

        try:
            # Create conversation
            conversation = Conversation(
                id=str(uuid.uuid4()),
                title=title or f"Conversation {datetime.now():%Y-%m-%d %H:%M}",
                created_at=datetime.now(),
            )

            # Apply template if provided
            if template_id:
                template = db_session.query(ConversationTemplate).get(template_id)
                if template:
                    conversation.model = template.default_model
                    conversation.temperature = template.default_temperature

                    # Add system prompt
                    if template.system_prompt:
                        system_msg = Message(
                            id=str(uuid.uuid4()),
                            conversation_id=conversation.id,
                            role=MessageRole.SYSTEM,
                            content=template.system_prompt,
                        )
                        db_session.add(system_msg)

            db_session.add(conversation)
            db_session.commit()

            # Create session
            session_id = str(uuid.uuid4())
            session = ConversationSession(session_id, db_session, conversation)
            self._active_sessions[session_id] = session

            logger.info(f"Created session {session_id} for conversation {conversation.id}")

            yield session

        finally:
            await session.close()
            self._active_sessions.pop(session_id, None)
            db_session.close()

    @asynccontextmanager
    async def load_session(self, conversation_id: str) -> ConversationSession:
        """Load an existing conversation session."""
        db_session = self.SessionFactory()

        try:
            conversation = (
                db_session.query(Conversation)
                .filter_by(id=conversation_id, status=ConversationStatus.ACTIVE)
                .first()
            )

            if not conversation:
                raise ValueError(f"Conversation {conversation_id} not found")

            session_id = str(uuid.uuid4())
            session = ConversationSession(session_id, db_session, conversation)
            self._active_sessions[session_id] = session

            logger.info(f"Loaded session {session_id} for conversation {conversation_id}")

            yield session

        finally:
            await session.close()
            self._active_sessions.pop(session_id, None)
            db_session.close()

    async def list_conversations(
        self,
        status: ConversationStatus | None = None,
        tag: str | None = None,
        search: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List conversations with filtering."""
        db_session = self.SessionFactory()

        try:
            query = db_session.query(Conversation)

            if status:
                query = query.filter(Conversation.status == status)
            else:
                query = query.filter(Conversation.status != ConversationStatus.DELETED)

            if tag:
                query = query.join(Conversation.tags).filter(Tag.name == tag)

            if search:
                query = query.filter(
                    or_(
                        Conversation.title.contains(search),
                        Conversation.description.contains(search),
                    )
                )

            query = query.order_by(desc(Conversation.updated_at))
            query = query.limit(limit).offset(offset)

            conversations = query.all()
            return [conv.to_dict() for conv in conversations]

        finally:
            db_session.close()

    async def get_conversation_stats(self, conversation_id: str) -> dict[str, Any]:
        """Get detailed statistics for a conversation."""
        db_session = self.SessionFactory()

        try:
            # Get conversation
            conversation = db_session.query(Conversation).get(conversation_id)
            if not conversation:
                return {}

            # Get message stats
            message_stats = (
                db_session.query(
                    Message.role,
                    func.count(Message.id).label("count"),
                    func.sum(Message.tokens).label("total_tokens"),
                )
                .filter_by(conversation_id=conversation_id)
                .group_by(Message.role)
                .all()
            )

            # Get analytics
            analytics = (
                db_session.query(
                    func.avg(Analytics.response_time_ms).label("avg_response_time"),
                    func.min(Analytics.response_time_ms).label("min_response_time"),
                    func.max(Analytics.response_time_ms).label("max_response_time"),
                )
                .filter_by(conversation_id=conversation_id)
                .first()
            )

            return {
                "conversation": conversation.to_dict(),
                "message_stats": {
                    stat.role: {"count": stat.count, "tokens": stat.total_tokens or 0}
                    for stat in message_stats
                },
                "performance": {
                    "avg_response_time_ms": analytics.avg_response_time if analytics else 0,
                    "min_response_time_ms": analytics.min_response_time if analytics else 0,
                    "max_response_time_ms": analytics.max_response_time if analytics else 0,
                },
            }

        finally:
            db_session.close()

    async def archive_conversation(self, conversation_id: str) -> bool:
        """Archive a conversation."""
        db_session = self.SessionFactory()

        try:
            conversation = db_session.query(Conversation).get(conversation_id)
            if conversation:
                conversation.status = ConversationStatus.ARCHIVED
                db_session.commit()
                return True
            return False

        finally:
            db_session.close()

    async def delete_conversation(self, conversation_id: str, hard_delete: bool = False) -> bool:
        """Delete a conversation."""
        db_session = self.SessionFactory()

        try:
            conversation = db_session.query(Conversation).get(conversation_id)
            if conversation:
                if hard_delete:
                    db_session.delete(conversation)
                else:
                    conversation.status = ConversationStatus.DELETED

                db_session.commit()
                return True
            return False

        finally:
            db_session.close()

    async def clear_all_conversations(self, hard_delete: bool = False) -> int:
        """Clear all conversations. Returns count of cleared conversations."""
        db_session = self.SessionFactory()

        try:
            conversations = (
                db_session.query(Conversation)
                .filter(Conversation.status != ConversationStatus.DELETED)
                .all()
            )

            count = len(conversations)

            for conversation in conversations:
                if hard_delete:
                    db_session.delete(conversation)
                else:
                    conversation.status = ConversationStatus.DELETED

            db_session.commit()
            return count

        finally:
            db_session.close()

    async def export_conversation(
        self, conversation_id: str, format: str = "json"
    ) -> dict[str, Any]:
        """Export a conversation in various formats."""
        db_session = self.SessionFactory()

        try:
            conversation = db_session.query(Conversation).get(conversation_id)
            if not conversation:
                return {}

            messages = (
                db_session.query(Message)
                .filter_by(conversation_id=conversation_id)
                .order_by(Message.created_at)
                .all()
            )

            export_data = {
                "conversation": conversation.to_dict(),
                "messages": [msg.to_dict() for msg in messages],
                "exported_at": datetime.now().isoformat(),
            }

            if format == "markdown":
                # Convert to markdown
                md_content = f"# {conversation.title}\n\n"
                md_content += f"*Created: {conversation.created_at}*\n\n"

                for msg in messages:
                    md_content += f"## {msg.role.title()} ({msg.created_at:%Y-%m-%d %H:%M})\n\n"
                    md_content += f"{msg.content}\n\n"

                export_data["markdown"] = md_content

            return export_data

        finally:
            db_session.close()

    async def _cleanup_loop(self):
        """Background task to cleanup old sessions."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour

                # Clean up inactive sessions
                now = datetime.now()
                to_remove = []

                for session_id, session in self._active_sessions.items():
                    if session.conversation.updated_at < now - self._session_timeout:
                        to_remove.append(session_id)

                for session_id in to_remove:
                    session = self._active_sessions.pop(session_id)
                    await session.close()

                if to_remove:
                    logger.info(f"Cleaned up {len(to_remove)} inactive sessions")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    async def get_templates(self) -> list[ConversationTemplate]:
        """Get available conversation templates."""
        db_session = self.SessionFactory()

        try:
            return (
                db_session.query(ConversationTemplate)
                .filter_by(is_public=True)
                .order_by(ConversationTemplate.name)
                .all()
            )

        finally:
            db_session.close()

    async def create_template(
        self, name: str, description: str, system_prompt: str | None = None, **kwargs
    ) -> ConversationTemplate:
        """Create a new conversation template."""
        db_session = self.SessionFactory()

        try:
            template = ConversationTemplate(
                name=name, description=description, system_prompt=system_prompt, **kwargs
            )

            db_session.add(template)
            db_session.commit()

            return template

        finally:
            db_session.close()
