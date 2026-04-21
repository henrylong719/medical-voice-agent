"""
Chat API routes for the medical scheduling agent.

Provides two endpoints:
  POST /api/v1/chat          → streaming response (Server-Sent Events)
  POST /api/v1/chat/invoke   → full response (for testing)

The streaming endpoint uses SSE (Server-Sent Events), which is simpler
than WebSocket for one-directional streaming. The browser sends a message,
then receives a stream of text chunks as the agent generates its response.
We'll switch to WebSocket in Phase 6 when we need bidirectional audio streaming.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent.graph import (
    AgentConfigurationError,
    ensure_agent_ready,
    invoke_agent,
    stream_agent_response,
)


router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    """Request body for the chat endpoint."""

    message: str = Field(min_length=1, description="The patient's message")
    thread_id: str = Field(
        description=(
            "Unique conversation ID. Use the same thread_id for all messages "
            "in one conversation so the agent remembers context."
        )
    )


class ChatResponse(BaseModel):
    """Response body for the non-streaming chat endpoint."""

    response: str
    thread_id: str


@router.post("")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """
    Send a message to the agent and receive a streaming response.

    Returns Server-Sent Events (SSE) — the response streams token by token
    as the agent generates it. The frontend can display text progressively
    for a more responsive feel.

    Usage:
        POST /api/v1/chat
        {"message": "I have a headache", "thread_id": "conv-123"}

        Response: text/event-stream with chunks like:
        data: I'd
        data:  be
        data:  happy
        data:  to help...
    """
    try:
        await ensure_agent_ready()
    except AgentConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    async def event_generator():
        async for chunk in stream_agent_response(
            message=request.message,
            thread_id=request.thread_id,
        ):
            # SSE format: "data: <content>\n\n"
            yield f"data: {chunk}\n\n"
        # Signal that the stream is complete
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.post("/invoke", response_model=ChatResponse)
async def chat_invoke(request: ChatRequest) -> ChatResponse:
    """
    Send a message and get the complete response (non-streaming).

    Simpler to use for testing — returns the full response as JSON
    instead of a stream. Same agent logic, just waits for completion.

    Usage:
        POST /api/v1/chat/invoke
        {"message": "I have a headache", "thread_id": "conv-123"}

        Response: {"response": "I'd be happy to help...", "thread_id": "conv-123"}
    """
    try:
        response = await invoke_agent(
            message=request.message,
            thread_id=request.thread_id,
        )
    except AgentConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ChatResponse(response=response, thread_id=request.thread_id)
