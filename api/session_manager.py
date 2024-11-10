# session_manager.py

from datetime import datetime, timedelta
import logging
from typing import Dict, Optional
from pydantic import BaseModel
from fastapi import HTTPException, Header
import uuid
from langchain.memory import ConversationBufferWindowMemory

logger = logging.getLogger("SessionManager")

class Session(BaseModel):
    """Represents a chat session with metadata and memory."""
    session_id: str
    created_at: datetime
    last_accessed: datetime
    memory: ConversationBufferWindowMemory
    user_id: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

class SessionManager:
    """Manages chat sessions and their lifecycle."""
    
    def __init__(self, session_timeout_minutes: int = 30):
        self.sessions: Dict[str, Session] = {}
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        logger.info(f"Initialized SessionManager with {session_timeout_minutes} minute timeout")
        
    def create_session(self, user_id: Optional[str] = None) -> str:
        """Create a new session and return its ID."""
        session_id = str(uuid.uuid4())
        memory = ConversationBufferWindowMemory(
            k=2,
            memory_key="chat_history",
            return_messages=True
        )
        
        self.sessions[session_id] = Session(
            session_id=session_id,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            memory=memory,
            user_id=user_id
        )
        
        logger.info(f"Created new session: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve and update a session if it exists and is valid."""
        session = self.sessions.get(session_id)
        
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return None
            
        if datetime.now() - session.last_accessed > self.session_timeout:
            logger.info(f"Session expired: {session_id}")
            self.end_session(session_id)
            return None
            
        session.last_accessed = datetime.now()
        return session
    
    def end_session(self, session_id: str) -> None:
        """End a session and clean up its resources."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Ended session: {session_id}")
    
    def cleanup_expired_sessions(self) -> None:
        """Remove expired sessions."""
        current_time = datetime.now()
        expired_sessions = [
            session_id for session_id, session in self.sessions.items()
            if current_time - session.last_accessed > self.session_timeout
        ]
        
        for session_id in expired_sessions:
            self.end_session(session_id)
            
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

# Initialize the global session manager
session_manager = SessionManager()