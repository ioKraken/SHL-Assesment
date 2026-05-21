"""Pydantic models for API request/response schemas.

The response schema is NON-NEGOTIABLE — the automated evaluator
will reject any deviation.
"""

from pydantic import BaseModel, Field
from typing import Optional


class Message(BaseModel):
    """A single message in the conversation history."""
    role: str = Field(..., description="Either 'user' or 'assistant'")
    content: str = Field(..., description="The message content")


class ChatRequest(BaseModel):
    """POST /chat request body."""
    messages: list[Message] = Field(..., description="Full conversation history")


class Recommendation(BaseModel):
    """A single assessment recommendation."""
    name: str = Field(..., description="Assessment name from catalog")
    url: str = Field(..., description="Full URL to the catalog page")
    test_type: str = Field(..., description="Test type code(s), e.g. 'K' or 'K, P'")


class ChatResponse(BaseModel):
    """POST /chat response body.
    
    Schema is non-negotiable:
    - reply: always present
    - recommendations: empty list when gathering context, 1-10 items when recommending
    - end_of_conversation: true only when agent considers task complete
    """
    reply: str = Field(..., description="Agent's natural language response")
    recommendations: list[Recommendation] = Field(
        default_factory=list,
        description="Empty when gathering context; 1-10 items when recommending"
    )
    end_of_conversation: bool = Field(
        default=False,
        description="True only when the agent considers the task complete"
    )


class HealthResponse(BaseModel):
    """GET /health response body."""
    status: str = "ok"
