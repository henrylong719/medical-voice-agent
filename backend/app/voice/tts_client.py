"""
Cartesia Streaming Text-to-Speech client (Sonic 3).

Connects to Cartesia's WebSocket API, sends text chunks, and yields
synthesized audio bytes in real-time. Uses the official Cartesia SDK's
async WebSocket client with context-based continuations.

Design:
  - Async generator pattern: callers iterate with `async for audio in client.synthesize(text_source)`
  - Uses Cartesia SDK (not raw websockets) — the SDK's context/continuation model
    maps cleanly to our pipeline pattern and handles auth, versioning, and connection management
  - Buffers incoming text tokens into sentence-sized chunks before sending to Cartesia,
    so each chunk has enough context for natural prosody
  - Each agent response is a separate "context" — this enables barge-in cancellation

Audio output:
  - PCM 16-bit signed little-endian (pcm_s16le) — same format as our STT input
  - Mono (1 channel)
  - 16 kHz sample rate — matches the rest of our pipeline

Usage:
    client = TTSClient(api_key="...")
    async for audio_chunk in client.synthesize(text_token_source):
        # audio_chunk is raw PCM bytes, ready to send to the browser
        send_to_browser(audio_chunk)
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator

import websockets

logger = logging.getLogger(__name__)


# ============================================================
# CONSTANTS
# ============================================================

# Audio format — must match what the browser expects to play.
# We use the same format as STT (16kHz, 16-bit, mono) so no
# conversion is needed anywhere in the pipeline.
SAMPLE_RATE = 16_000
ENCODING = "pcm_s16le"  # 16-bit signed little-endian
CONTAINER = "raw"  # No WAV headers — raw PCM bytes

# Cartesia WebSocket endpoint
CARTESIA_WSS_URL = "wss://api.cartesia.ai/tts/websocket"

# Cartesia API version
CARTESIA_VERSION = "2025-11-04"

# TTS model — Sonic 3 is the latest, lowest-latency model
MODEL_ID = "sonic-3"

# Default voice — "Tessa" is a warm, professional American English voice.
# Good for a medical receptionist. Browse voices at play.cartesia.ai
DEFAULT_VOICE_ID = "6ccbfb76-1fc6-48f7-b71d-91ac6298247b"

# Sentence-ending punctuation for text buffering.
# We accumulate tokens until we hit one of these, then send the
# buffered sentence to Cartesia for synthesis.
SENTENCE_ENDINGS = frozenset(".!?")

# When streaming from an LLM, sometimes the final chunk doesn't end
# with punctuation. We flush the buffer after the text source ends.
# This minimum length prevents flushing tiny fragments like "OK"
# that sound unnatural on their own — but we also always flush on
# source exhaustion regardless, so this is just for mid-stream splits.
MIN_FLUSH_LENGTH = 5


# ============================================================
# TTS CLIENT
# ============================================================


class TTSClient:
    """Async streaming Text-to-Speech client for Cartesia Sonic 3.

    The core method is `synthesize()`, an async generator that:
      1. Opens a WebSocket connection to Cartesia
      2. Creates a context for the current agent response
      3. Buffers incoming text tokens into sentences
      4. Sends sentences to Cartesia via context continuation
      5. Yields PCM audio chunks as they arrive
      6. Cleans up on completion or cancellation (barge-in)

    Args:
        api_key: Cartesia API key for authentication.
        voice_id: Cartesia voice ID. Defaults to "Tessa".
        sample_rate: Audio sample rate in Hz. Default 16000.
    """

    def __init__(
        self,
        api_key: str,
        voice_id: str = DEFAULT_VOICE_ID,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self._api_key = api_key
        self._voice_id = voice_id
        self._sample_rate = sample_rate

    def _build_url(self) -> str:
        """Build the Cartesia WebSocket URL with query parameters."""
        return (
            f"{CARTESIA_WSS_URL}"
            f"?cartesia_version={CARTESIA_VERSION}"
            f"&api_key={self._api_key}"
        )

    def _make_generation_request(
        self,
        text: str,
        context_id: str,
        is_last: bool = False,
    ) -> str:
        """Build a JSON generation request for Cartesia.

        Args:
            text: The text to synthesize.
            context_id: Unique ID for this generation context.
            is_last: If True, this is the final text chunk — sets
                "continue" to false so Cartesia closes the context.
                All non-final chunks set "continue" to true, telling
                Cartesia to expect more text and maintain prosody.
        """
        request = {
            "model_id": MODEL_ID,
            "transcript": text,
            "voice": {
                "mode": "id",
                "id": self._voice_id,
            },
            "language": "en",
            "context_id": context_id,
            "output_format": {
                "container": CONTAINER,
                "encoding": ENCODING,
                "sample_rate": self._sample_rate,
            },
            # "continue": true  → more text coming, keep context open
            # "continue": false → final chunk, close context after this
            "continue": not is_last,
        }
        return json.dumps(request)

    def _make_cancel_request(self, context_id: str) -> str:
        """Build a JSON cancel request for a context.

        Used for barge-in: when the patient starts speaking while
        the agent is still talking, we cancel the current TTS context
        to stop generating audio immediately.
        """
        return json.dumps({
            "context_id": context_id,
            "cancel": True,
        })

    async def synthesize(
        self,
        text_source: AsyncIterator[str],
    ) -> AsyncIterator[bytes]:
        """Stream text through Cartesia and yield audio chunks.

        This is the main entry point. It manages the full lifecycle:
        connect → buffer text into sentences → send to Cartesia →
        yield audio → disconnect.

        Args:
            text_source: An async iterator that yields text chunks
                (typically individual tokens from the LLM). The iterator
                signals completion by raising StopAsyncIteration.

        Yields:
            Raw PCM audio bytes (16kHz, 16-bit, mono). Each yield is
            one audio chunk from Cartesia, typically a few hundred ms
            of audio.
        """
        url = self._build_url()
        context_id = uuid.uuid4().hex

        logger.info("Connecting to Cartesia TTS")

        async with websockets.connect(
            url,
            ping_interval=20,
            ping_timeout=10,
        ) as ws:
            logger.info("Cartesia WebSocket connected, context=%s", context_id)

            # ── Coordination between send and receive ──────────
            # Each sentence we send to Cartesia generates its own
            # stream of audio chunks followed by a "done" message.
            # The receive task needs to know how many "done" messages
            # to expect before it can signal completion.
            #
            # send_complete: set when the send task has finished
            #   sending all text (so the receive task knows the
            #   total count is final)
            # send_count: how many generation requests were sent
            audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
            send_complete = asyncio.Event()
            send_count: list[int] = [0]  # Mutable container for sharing

            receive_task = asyncio.create_task(
                self._receive_audio(
                    ws, context_id, audio_queue, send_complete, send_count
                )
            )
            send_task = asyncio.create_task(
                self._send_text(
                    ws, text_source, context_id, send_complete, send_count
                )
            )

            try:
                # ── Yield audio chunks from the queue ──────────
                # The receive task puts audio bytes into the queue.
                # None is the sentinel that means "done, no more audio."
                while True:
                    chunk = await audio_queue.get()
                    if chunk is None:
                        break
                    yield chunk

            finally:
                # ── Cleanup ────────────────────────────────────
                if not send_task.done():
                    send_task.cancel()
                    try:
                        await send_task
                    except asyncio.CancelledError:
                        pass

                if not receive_task.done():
                    receive_task.cancel()
                    try:
                        await receive_task
                    except asyncio.CancelledError:
                        pass

                logger.info("TTS client disconnected")

    async def _send_text(
        self,
        ws: websockets.ClientConnection,
        text_source: AsyncIterator[str],
        context_id: str,
        send_complete: asyncio.Event,
        send_count: list[int],
    ) -> None:
        """Background task: buffer text tokens into sentences and send.

        Accumulates tokens from the LLM until a sentence boundary
        (period, question mark, exclamation mark) is reached, then
        sends the complete sentence to Cartesia. This ensures each
        TTS request has enough context for natural-sounding prosody.

        All mid-stream sentences are sent with continue=true (more
        text is coming). Only the final flush — after the text source
        is exhausted — is sent with continue=false (close the context).

        When the text source is exhausted, flushes any remaining
        buffered text as the final chunk, then signals send_complete
        so the receive task knows how many "done" messages to expect.
        """
        buffer = ""

        try:
            async for token in text_source:
                buffer += token

                # Check if the buffer contains a sentence boundary.
                # We look for punctuation followed by a space or end,
                # which indicates a natural break point.
                send_up_to = self._find_sentence_boundary(buffer)

                if send_up_to is not None:
                    sentence = buffer[:send_up_to].strip()
                    buffer = buffer[send_up_to:].lstrip()

                    if sentence:
                        logger.debug("TTS sending: %r", sentence)
                        # Mid-stream: always is_last=False so Cartesia
                        # keeps the context open for more text.
                        request = self._make_generation_request(
                            text=sentence,
                            context_id=context_id,
                            is_last=False,
                        )
                        await ws.send(request)
                        send_count[0] += 1

            # Flush remaining buffer — this is the final chunk.
            remaining = buffer.strip()
            if remaining:
                logger.debug("TTS flushing final: %r", remaining)
                request = self._make_generation_request(
                    text=remaining,
                    context_id=context_id,
                    is_last=True,
                )
                await ws.send(request)
                send_count[0] += 1
            elif send_count[0] > 0:
                # All text ended on a sentence boundary, so buffer is
                # empty. But we already sent the last sentence with
                # continue=true. We need to close the context.
                # Send an empty-ish final request to signal completion.
                logger.debug("TTS closing context (no remainder)")
                request = self._make_generation_request(
                    text=" ",
                    context_id=context_id,
                    is_last=True,
                )
                await ws.send(request)
                send_count[0] += 1

        except asyncio.CancelledError:
            # Barge-in or shutdown — cancel the TTS context so
            # Cartesia stops generating audio for this response.
            try:
                await ws.send(self._make_cancel_request(context_id))
            except Exception:
                pass
            raise

        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket closed while sending text")

        except Exception:
            logger.exception("Unexpected error in TTS send loop")

        finally:
            # Signal that no more requests will be sent.
            # The receive task uses this + send_count to know
            # when all audio has been received.
            logger.debug("Send complete: %d requests sent", send_count[0])
            send_complete.set()

    async def _receive_audio(
        self,
        ws: websockets.ClientConnection,
        context_id: str,
        audio_queue: asyncio.Queue[bytes | None],
        send_complete: asyncio.Event,
        send_count: list[int],
    ) -> None:
        """Background task: receive audio chunks from Cartesia.

        Listens for WebSocket messages and puts decoded audio bytes
        into the queue. With proper continuation semantics (all chunks
        use continue=true except the final one), Cartesia streams
        audio continuously and sends a single "done" when the context
        completes.

        Cartesia message types we handle:
          - "chunk": contains base64-encoded audio data
          - "done": context complete, all audio has been sent
          - "error": something went wrong
          - "flush_done": acknowledgment (informational only)
        """
        try:
            async for message in ws:
                data = json.loads(message)
                msg_type = data.get("type")

                # Ignore messages for other contexts
                if data.get("context_id") != context_id:
                    continue

                if msg_type == "chunk":
                    audio_b64 = data.get("data")
                    if audio_b64:
                        audio_bytes = base64.b64decode(audio_b64)
                        await audio_queue.put(audio_bytes)

                elif msg_type == "done":
                    logger.info(
                        "TTS context complete: %s", context_id
                    )
                    break

                elif msg_type == "error":
                    error_msg = data.get("message", "Unknown error")
                    logger.error("Cartesia error: %s", error_msg)
                    break

                elif msg_type == "flush_done":
                    logger.debug("TTS flush acknowledged")

        except asyncio.CancelledError:
            raise

        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket closed while receiving audio")

        except Exception:
            logger.exception("Unexpected error in TTS receive loop")

        finally:
            # Signal to the consumer that no more audio is coming
            await audio_queue.put(None)

    @staticmethod
    def _find_sentence_boundary(text: str) -> int | None:
        """Find the position just past the last sentence-ending punctuation.

        Returns the index to split at, or None if no boundary found.

        We look for sentence-ending punctuation (.!?) that is followed
        by a space and more text. We don't split at punctuation that's
        at the very end of the buffer — because more tokens might be
        coming that belong to the same sentence (e.g., the LLM might
        send "Dr." and then "Smith" as separate tokens).

        Examples:
            "Hello world. How are"  → 13  (split after ".")
            "Hello world."          → None (punctuation at end, wait for more)
            "Hello world"           → None (no punctuation)
            "Call 911! Are you"     → 10   (split after "!")
        """
        best = None
        for i, char in enumerate(text):
            if char in SENTENCE_ENDINGS:
                # Check if there's content after this punctuation
                # (at least a space and one more character)
                remaining = text[i + 1:]
                if remaining and remaining.lstrip():
                    # There's more text after the punctuation — this
                    # is a valid split point. Use the position after
                    # the punctuation.
                    best = i + 1

        return best
