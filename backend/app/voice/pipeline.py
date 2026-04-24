"""
Voice pipeline: STT → LangGraph Agent → TTS.

Orchestrates the three stages of the voice sandwich:
  1. STT (AssemblyAI): audio bytes → transcript text
  2. Agent (LangGraph): transcript text → response text tokens
  3. TTS (Cartesia): response text tokens → audio bytes

The pipeline is designed around async generators so each stage
streams naturally into the next without blocking.

Usage:
    pipeline = VoicePipeline(
        stt=STTClient(api_key="..."),
        tts=TTSClient(api_key="..."),
        thread_id="call-123",
    )
    async for event in pipeline.run(audio_source):
        if event.type == "audio":
            send_to_speaker(event.audio)
        elif event.type == "transcript":
            update_live_caption(event.text)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator

from app.agent.graph import stream_agent_response
from app.voice.stt_client import STTClient, TranscriptEvent
from app.voice.tts_client import TTSClient

logger = logging.getLogger(__name__)


# ============================================================
# PIPELINE EVENTS
# ============================================================
# The pipeline yields typed events so the caller (WebSocket
# endpoint or test script) can handle each type appropriately.
# ============================================================


class PipelineEventType(str, Enum):
    """Types of events the voice pipeline can produce."""

    # Partial transcript — patient is still speaking, text is updating.
    # Useful for live captioning in the UI.
    TRANSCRIPT_PARTIAL = "transcript_partial"

    # Final transcript — patient finished a thought. This text
    # has been sent to the agent for processing.
    TRANSCRIPT_FINAL = "transcript_final"

    # Audio chunk — synthesized speech from the agent's response.
    # Raw PCM bytes ready to send to the browser/speaker.
    AUDIO = "audio"

    # Agent text — a chunk of the agent's text response.
    # Useful for displaying the agent's response as text alongside audio.
    AGENT_TEXT = "agent_text"


@dataclass(frozen=True, slots=True)
class PipelineEvent:
    """A single event from the voice pipeline.

    Attributes:
        type: What kind of event this is.
        text: Transcript or agent text (for text events).
        audio: Raw PCM audio bytes (for audio events).
        turn: Which speaking turn this belongs to (0-indexed).
    """

    type: PipelineEventType
    text: str = ""
    audio: bytes = b""
    turn: int = 0


# ============================================================
# VOICE PIPELINE
# ============================================================


class VoicePipeline:
    """Orchestrates STT → Agent → TTS for a single voice session.

    Each call to `run()` processes a complete voice session:
    the patient speaks, gets transcribed, the agent responds,
    and the response is synthesized back to audio.

    The pipeline handles multiple turns within a session:
    patient speaks → agent responds → patient speaks again → ...

    Args:
        stt: Configured STT client for speech recognition.
        tts: Configured TTS client for speech synthesis.
        thread_id: Conversation thread ID for the agent. Same
            thread_id = same conversation history, so the agent
            remembers previous turns.
    """

    def __init__(
        self,
        stt: STTClient,
        tts: TTSClient,
        thread_id: str,
    ) -> None:
        self._stt = stt
        self._tts = tts
        self._thread_id = thread_id

    async def run(
        self,
        audio_source: AsyncIterator[bytes],
    ) -> AsyncIterator[PipelineEvent]:
        """Run the voice pipeline on an audio source.

        Streams audio through STT, sends final transcripts to the
        agent, and yields synthesized audio responses.

        Args:
            audio_source: Async iterator of raw PCM audio chunks
                from the microphone.

        Yields:
            PipelineEvent instances:
              - TRANSCRIPT_PARTIAL: live caption updates
              - TRANSCRIPT_FINAL: completed patient utterance
              - AGENT_TEXT: text chunks from the agent's response
              - AUDIO: synthesized speech audio chunks
        """
        turn = 0

        logger.info("Voice pipeline starting (thread=%s)", self._thread_id)

        # ── Stage 1: STT ──────────────────────────────────────
        # Stream audio to AssemblyAI and process transcript events
        # as they arrive. For each final transcript, run the agent
        # and TTS stages.
        async for transcript_event in self._stt.transcribe(audio_source):
            if transcript_event.is_final:
                # Patient finished a thought — send to agent
                text = transcript_event.text.strip()
                if not text:
                    continue

                logger.info(
                    "Turn %d transcript: %r (confidence=%.2f)",
                    turn,
                    text,
                    transcript_event.confidence,
                )

                yield PipelineEvent(
                    type=PipelineEventType.TRANSCRIPT_FINAL,
                    text=text,
                    turn=turn,
                )

                # ── Stage 2 + 3: Agent → TTS ──────────────────
                # Stream the agent's response through TTS and yield
                # both text and audio events.
                async for event in self._process_agent_response(text, turn):
                    yield event

                turn += 1

            else:
                # Partial transcript — yield for live captioning
                if transcript_event.text:
                    yield PipelineEvent(
                        type=PipelineEventType.TRANSCRIPT_PARTIAL,
                        text=transcript_event.text,
                        turn=turn,
                    )

        logger.info(
            "Voice pipeline ended after %d turn(s) (thread=%s)",
            turn,
            self._thread_id,
        )

    async def _process_agent_response(
        self,
        transcript: str,
        turn: int,
    ) -> AsyncIterator[PipelineEvent]:
        """Send transcript to agent and stream synthesized response.

        This is where Stages 2 and 3 happen:
          - The agent's stream_agent_response yields text tokens
          - Those tokens feed into the TTS client's synthesize method
          - We yield both AGENT_TEXT events (for display) and
            AUDIO events (for playback)

        The text tokens are "teed" — each token is yielded as an
        AGENT_TEXT event AND fed to the TTS client. This lets the
        caller show text and play audio simultaneously.
        """
        logger.info("Processing turn %d through agent", turn)

        # ── Tee the agent output ──────────────────────────────
        # We need to send agent tokens to BOTH the TTS client AND
        # yield them as text events. We use an async generator that
        # captures tokens as they pass through.
        agent_tokens = _AgentTokenTee(
            stream_agent_response(transcript, self._thread_id)
        )

        # ── Stage 3: TTS ─────────────────────────────────────
        # Feed the tee'd agent output to TTS and yield audio chunks.
        async for audio_chunk in self._tts.synthesize(agent_tokens):
            yield PipelineEvent(
                type=PipelineEventType.AUDIO,
                audio=audio_chunk,
                turn=turn,
            )

        # ── Yield captured text after audio ───────────────────
        # The agent text was captured by the tee as it flowed through.
        # We yield it as a single AGENT_TEXT event with the full response.
        full_text = agent_tokens.captured_text
        if full_text:
            yield PipelineEvent(
                type=PipelineEventType.AGENT_TEXT,
                text=full_text,
                turn=turn,
            )

        logger.info(
            "Turn %d complete: agent responded with %d chars",
            turn,
            len(full_text),
        )


class _AgentTokenTee:
    """Async iterator wrapper that captures text tokens as they pass through.

    Used to "tee" the agent's output: tokens flow through to the TTS
    client (which consumes this as its text_source), and are also
    captured so we can yield the full text as a pipeline event.

    This avoids calling stream_agent_response twice or buffering the
    entire response before starting TTS.
    """

    def __init__(self, agent_stream: AsyncIterator[str]) -> None:
        self._stream = agent_stream
        self._captured: list[str] = []

    @property
    def captured_text(self) -> str:
        """The full agent response text captured so far."""
        return "".join(self._captured)

    def __aiter__(self) -> _AgentTokenTee:
        return self

    async def __anext__(self) -> str:
        token = await self._stream.__anext__()
        self._captured.append(token)
        return token
