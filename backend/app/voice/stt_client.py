"""
AssemblyAI Streaming Speech-to-Text client (v3 API).

Connects to AssemblyAI's WebSocket endpoint, sends raw PCM audio chunks,
and yields transcript events as they arrive. Uses the v3 "Turn" message
format with immutable transcriptions and intelligent endpointing.

Design:
  - Async generator pattern: callers iterate with `async for event in client.transcribe(audio_source)`
  - Raw `websockets` library instead of AssemblyAI's SDK — gives us full async control
    and avoids the callback-to-generator impedance mismatch
  - Yields TranscriptEvent dataclass instances, not raw JSON — typed, predictable interface
  - Handles connection lifecycle: connect → stream audio → receive transcripts → terminate

Audio requirements:
  - PCM 16-bit signed little-endian (the default for browser MediaRecorder and most mic capture)
  - Mono (1 channel)
  - 16 kHz sample rate (configured via SAMPLE_RATE constant)

Usage:
    client = STTClient(api_key="...")
    async for event in client.transcribe(audio_source):
        if event.is_final:
            # Full turn complete — send to agent
            send_to_agent(event.text)
        else:
            # Partial update — could display live captions
            update_display(event.text)
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import AsyncIterator
from urllib.parse import urlencode

import websockets
from websockets.asyncio.client import ClientConnection

logger = logging.getLogger(__name__)

# ============================================================
# CONSTANTS
# ============================================================

# Audio format that AssemblyAI expects (and that the browser will send).
# All three values must agree between browser capture, our WebSocket
# relay, and this client — a mismatch causes pitch/speed distortion.
SAMPLE_RATE = 16_000  # 16 kHz — standard for speech

# AssemblyAI v3 WebSocket endpoint
AAI_WSS_BASE = "wss://streaming.assemblyai.com/v3/ws"

# Speech model: Universal-3 Real-Time Pro — lowest latency, highest
# accuracy for voice agent use cases. This is required in v3 (no default).
SPEECH_MODEL = "u3-rt-pro"

# ============================================================
# TRANSCRIPT EVENT
# ============================================================
# A clean, typed wrapper around the raw JSON that AssemblyAI sends.
# The rest of the pipeline never touches raw JSON — it works with these.
# ============================================================


@dataclass(frozen=True, slots=True)
class TranscriptEvent:
    """A single transcript event from AssemblyAI.

    Attributes:
        text: The accumulated transcript text for this turn.
            Grows progressively: "" → "Hi," → "Hi, my name is Sonny."
        is_final: True when AssemblyAI's endpointing model decides
            the speaker has finished their turn. This is the signal
            to send the text to the agent.
        turn_order: Integer that increments with each new speaking turn.
            Useful for tracking conversation flow.
        confidence: End-of-turn confidence score (0.0 to 1.0).
            Crosses ~0.5 when is_final becomes True.
        utterance: Populated when an utterance boundary is detected
            within a turn. Fires faster than end_of_turn — could be
            used for preemptive LLM generation in the future.
    """

    text: str
    is_final: bool
    turn_order: int = 0
    confidence: float = 0.0
    utterance: str = ""


# ============================================================
# STT CLIENT
# ============================================================


class STTClient:
    """Async streaming Speech-to-Text client for AssemblyAI v3.

    The core method is `transcribe()`, an async generator that:
      1. Opens a WebSocket connection to AssemblyAI
      2. Spawns a background task to send audio chunks from the source
      3. Yields TranscriptEvent instances as they arrive
      4. Cleans up on completion or error

    Args:
        api_key: AssemblyAI API key for authentication.
        sample_rate: Audio sample rate in Hz. Must match the audio
            source. Default 16000 (standard for speech).
    """

    def __init__(self, api_key: str, sample_rate: int = SAMPLE_RATE) -> None:
        self._api_key = api_key
        self._sample_rate = sample_rate

    def _build_url(self) -> str:
        """Build the v3 WebSocket URL with query parameters."""
        params = {
            "speech_model": SPEECH_MODEL,
            "sample_rate": self._sample_rate,
        }
        return f"{AAI_WSS_BASE}?{urlencode(params)}"

    async def transcribe(
        self,
        audio_source: AsyncIterator[bytes],
    ) -> AsyncIterator[TranscriptEvent]:
        """Stream audio to AssemblyAI and yield transcript events.

        This is the main entry point. It manages the full lifecycle:
        connect → send/receive concurrently → terminate → disconnect.

        Args:
            audio_source: An async iterator that yields raw PCM audio
                chunks (bytes). Each chunk is typically 50–100ms of audio.
                The iterator signals "no more audio" by raising StopAsyncIteration
                (i.e., the async for loop ends).

        Yields:
            TranscriptEvent instances. Check event.is_final to know
            when a complete turn is ready for the agent.

        The method uses two concurrent tasks:
          - _send_audio: reads from audio_source, sends to AssemblyAI
          - The main loop: reads from AssemblyAI, yields events

        When the audio source is exhausted, we send a Terminate message
        and drain any remaining transcript events before closing.
        """
        url = self._build_url()
        headers = {"Authorization": self._api_key}

        logger.info("Connecting to AssemblyAI STT at %s", AAI_WSS_BASE)

        # ── Open the WebSocket connection ──────────────────────
        # `websockets.connect` returns an async context manager that
        # handles the handshake and cleanup. We pass extra_headers
        # for auth (AssemblyAI uses a plain Authorization header,
        # not Bearer token).
        async with websockets.connect(
            url,
            additional_headers=headers,
            # Ping/pong keepalive — prevents proxies and load balancers
            # from closing idle connections. AssemblyAI sessions can last
            # up to 3 hours, so we need this for long conversations.
            ping_interval=20,
            ping_timeout=10,
        ) as ws:
            # ── Wait for the Begin message ─────────────────────
            # The first message from AssemblyAI is always a Begin
            # event with the session ID. We log it but don't yield it.
            begin_raw = await ws.recv()
            begin_data = json.loads(begin_raw)

            if begin_data.get("type") != "Begin":
                logger.error(
                    "Expected 'Begin' message, got: %s", begin_data.get("type")
                )
                return

            session_id = begin_data.get("id", "unknown")
            logger.info("AssemblyAI session started: %s", session_id)

            # ── Start sending audio in the background ──────────
            # We need to send audio AND receive transcripts at the
            # same time — that's the whole point of WebSocket being
            # bidirectional. asyncio.create_task lets us do both
            # concurrently within a single async function.
            send_task = asyncio.create_task(self._send_audio(ws, audio_source))

            try:
                # ── Receive transcript events ──────────────────
                # Loop until the WebSocket closes or we get a
                # Termination message. Each message is either a
                # Turn (transcript data) or Termination (session end).
                async for message in ws:
                    data = json.loads(message)
                    msg_type = data.get("type")

                    if msg_type == "Turn":
                        event = TranscriptEvent(
                            text=data.get("transcript", ""),
                            is_final=data.get("end_of_turn", False),
                            turn_order=data.get("turn_order", 0),
                            confidence=data.get("end_of_turn_confidence", 0.0),
                            utterance=data.get("utterance", ""),
                        )

                        # Only yield events that have text content.
                        # AssemblyAI sends Turn messages even when
                        # transcript is empty (start of a new turn
                        # before any words are finalized).
                        if event.text or event.is_final:
                            yield event
                    elif msg_type == "Termination":
                        audio_dur = data.get("audio_duration_seconds", 0)
                        session_dur = data.get("session_duration_seconds", 0)
                        logger.info(
                            "AssemblyAI session ended: audio=%.1fs, session=%.1fs",
                            audio_dur,
                            session_dur,
                        )
                        break
                    elif msg_type == "Error":
                        error_msg = data.get("error", "Unknown error")
                        logger.error("AssemblyAI error: %s", error_msg)
                        break
            finally:
                # ── Cleanup ────────────────────────────────────
                # Cancel the send task if it's still running (e.g.,
                # if we broke out of the receive loop due to an error
                # while audio was still streaming).
                if not send_task.done():
                    send_task.cancel()
                    try:
                        await send_task
                    except asyncio.CancelledError:
                        pass

                    logger.info("STT client disconnected")

    async def _send_audio(
        self,
        ws: ClientConnection,
        audio_source: AsyncIterator[bytes],
    ) -> None:
        """Background task: read audio chunks and send to AssemblyAI.

        Runs concurrently with the receive loop in transcribe().
        When the audio source is exhausted (patient hangs up, mic
        disconnected, etc.), sends a Terminate message to tell
        AssemblyAI to flush any remaining transcript and end the session.

        AssemblyAI v3 accepts raw binary audio frames directly —
        no base64 encoding or JSON wrapping needed for audio data.
        Only control messages (like Terminate) are sent as JSON.
        """
        try:
            async for chunk in audio_source:
                # Send raw PCM bytes directly as a binary WebSocket frame.
                # This is more efficient than base64-encoding into JSON
                # (which the v2 API required). Each chunk is typically
                # 50ms of audio = 1,600 bytes at 16kHz/16bit/mono.
                await ws.send(chunk)

            # Audio source exhausted — tell AssemblyAI to wrap up.
            # It will flush any buffered transcript and send a
            # Termination response, which our receive loop catches.
            logger.info("Audio source ended: sending Terminate")
            await ws.send(json.dumps({"type": "Terminate"}))

        except asyncio.CancelledError:
            try:
                await ws.send(json.dumps({"type": "Terminate"}))
            except Exception:
                pass  # Connection might already be closed
            raise

        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket closed while sending audio")

        except Exception:
            logger.exception("Unexpected error in audio send loop")
