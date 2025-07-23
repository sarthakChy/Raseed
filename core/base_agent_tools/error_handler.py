import logging
import traceback
from typing import Dict, Any, Optional
from datetime import datetime
import json
from enum import Enum

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorType(Enum):
    VALIDATION = "validation"
    PROCESSING = "processing"
    SYSTEM = "system"
    DATABASE = "database"
    API = "api"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    TOOL_EXECUTION = "tool_execution"

class ErrorHandler:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.error_history = []
        self.fallback_responses = {
            ErrorType.VALIDATION: "I need some additional information to help you with that request.",
            ErrorType.PROCESSING: "I'm having trouble processing that request right now. Let me try a different approach.",
            ErrorType.DATABASE: "I'm experiencing some database connectivity issues. Please try again in a moment.",
            ErrorType.API: "There's a temporary service issue. Please try again shortly.",
            ErrorType.SYSTEM: "I'm experiencing technical difficulties. Please try again later.",
            ErrorType.TOOL_EXECUTION: "I encountered an issue with one of my tools. Let me try an alternative approach."
        }
    
    def handle_error(
        self, 
        error: Exception, 
        context: str, 
        user_id: Optional[str] = None,
        error_type: Optional[ErrorType] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM
    ) -> Dict[str, Any]:
        if error_type is None:
            error_type = self._classify_error(error)
        
        # Use 'error_message' instead of 'message' to avoid LogRecord conflict
        error_record = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type.value,
            "severity": severity.value,
            "error_message": str(error),  # Changed from 'message' to 'error_message'
            "context": context,
            "user_id": user_id,
            "traceback": traceback.format_exc(),
            "error_class": error.__class__.__name__
        }
        
        self._log_error(error_record)
        self.error_history.append(error_record)
        if len(self.error_history) > 100:
            self.error_history.pop(0)
        
        fallback_response = self._generate_fallback_response(error_type, context)
        
        return {
            "error_handled": True,
            "error_id": self._generate_error_id(error_record),
            "fallback_response": fallback_response,
            "retry_suggested": self._should_suggest_retry(error_type),
            "user_message": self._generate_user_message(error_type, context)
        }
    
    def log_error(self, error_type: str, message: str, context: Dict[str, Any]):
        error_record = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "severity": "medium",
            "error_message": message,  # Changed from 'message' to 'error_message'
            "context": context
        }
        
        self._log_error(error_record)
        self.error_history.append(error_record)
    
    def _classify_error(self, error: Exception) -> ErrorType:
        error_class = error.__class__.__name__.lower()
        error_message = str(error).lower()
        
        if "validation" in error_message or "invalid" in error_message:
            return ErrorType.VALIDATION
        elif "database" in error_message or "sql" in error_message:
            return ErrorType.DATABASE
        elif "api" in error_message or "request" in error_message:
            return ErrorType.API
        elif "auth" in error_message:
            return ErrorType.AUTHENTICATION
        elif "permission" in error_message or "forbidden" in error_message:
            return ErrorType.AUTHORIZATION
        elif "tool" in error_message or "function" in error_message:
            return ErrorType.TOOL_EXECUTION
        else:
            return ErrorType.SYSTEM
    
    def _log_error(self, error_record: Dict[str, Any]):
        severity = error_record.get("severity", "medium")
        message = f"[{error_record['error_type']}] {error_record['error_message']} | Context: {error_record['context']}"
        
        if severity == "critical":
            self.logger.critical(message, extra=error_record)
        elif severity == "high":
            self.logger.error(message, extra=error_record)
        elif severity == "medium":
            self.logger.warning(message, extra=error_record)
        else:
            self.logger.info(message, extra=error_record)
    
    def _generate_error_id(self, error_record: Dict[str, Any]) -> str:
        timestamp = error_record["timestamp"].replace(":", "").replace("-", "").replace(".", "")
        error_type = error_record["error_type"][:3].upper()
        return f"{error_type}-{timestamp[-8:]}"
    
    def _generate_fallback_response(self, error_type: ErrorType, context: str) -> str:
        base_response = self.fallback_responses.get(error_type, 
            "I encountered an unexpected issue. Please try again.")
        
        if "query" in context.lower():
            base_response += " You might try rephrasing your question."
        elif "data" in context.lower():
            base_response += " The issue might be with the underlying data."
        
        return base_response
    
    def _should_suggest_retry(self, error_type: ErrorType) -> bool:
        transient_errors = {
            ErrorType.DATABASE,
            ErrorType.API,
            ErrorType.SYSTEM
        }
        return error_type in transient_errors
    
    def _generate_user_message(self, error_type: ErrorType, context: str) -> str:
        if error_type == ErrorType.VALIDATION:
            return "Please check your input and try again."
        elif error_type == ErrorType.DATABASE:
            return "We're experiencing temporary data access issues."
        elif error_type == ErrorType.API:
            return "There's a temporary service interruption."
        else:
            return "We're working to resolve a technical issue."
    
    def get_error_stats(self) -> Dict[str, Any]:
        if not self.error_history:
            return {"total_errors": 0, "error_types": {}, "recent_errors": []}
        
        error_types = {}
        for error in self.error_history:
            error_type = error["error_type"]
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        recent_errors = self.error_history[-10:]
        
        return {
            "total_errors": len(self.error_history),
            "error_types": error_types,
            "recent_errors": recent_errors,
            "most_common_error": max(error_types.items(), key=lambda x: x[1])[0] if error_types else None
        }
    
    def clear_error_history(self):
        self.error_history.clear()
        self.logger.info("Error history cleared")
    
    def is_service_degraded(self, error_threshold: int = 5) -> bool:
        if len(self.error_history) < error_threshold:
            return False
        
        recent_cutoff = datetime.now().timestamp() - 600
        recent_errors = [
            error for error in self.error_history[-20:]
            if datetime.fromisoformat(error["timestamp"]).timestamp() > recent_cutoff
        ]
        
        return len(recent_errors) >= error_threshold