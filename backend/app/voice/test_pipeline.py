"""
End-to-end test for the voice pipeline (STT → Agent → TTS).

Reads a WAV file containing a patient utterance, runs it through
the full pipeline (AssemblyAI → LangGraph Agent → Cartesia), and
saves the agent's spoken response as a WAV file.

This proves the pipeline works end-to-end without a browser.

Usage:
    # Create a test audio file (macOS):
    say -o test.wav --data-format=LEI16@16000 "I have a headache and blurry vision"

    # Run the pipeline:
    python -m app.voice.test_pipeline test.wav

    # Listen to the agent's response:
    afplay output_pipeline.wav

Requirements:
    - ASSEMBLYAI_API_KEY set in backend/.env
    - CARTESIA_API_KEY set in backend/.env
    - ANTHROPIC_API_KEY set in backend/.env
    - SUPABASE_DB_URI set in backend/.env (for agent conversation memory)
"""

from __future__ import annotations

import asyncio
import argparse
import sys
import uuid
import wave
from pathlib import Path
from typing import AsyncIterator

from app.config import settings
from app.voice.stt_client import STTClient, SAMPLE_RATE
from app.voice.tts_client import TTSClient
from app.voice.pipeline import VoicePipeline, PipelineEventType


# Audio chunk config (same as test_stt.py)
CHUNK_DURATION_MS = 50
BYTES_PER_SAMPLE = 2
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000) * BYTES_PER_SAMPLE


def read_wav_as_pcm(path: Path) -> bytes:
    """Read a WAV file and return raw PCM bytes."""
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()

        print(f"Input: {path.name}")
        print(f"  {framerate}Hz, {sample_width * 8}-bit, {'stereo' if channels == 2 else 'mono'}")
        print(f"  Duration: {n_frames / framerate:.2f}s")
        print()

        if sample_width != BYTES_PER_SAMPLE:
            raise ValueError(
                f"Expected {BYTES_PER_SAMPLE * 8}-bit audio, got {sample_width * 8}-bit"
            )

        pcm_data = wf.readframes(n_frames)

        # Downmix stereo to mono if needed
        if channels == 2:
            import struct

            samples = struct.unpack(f"<{n_frames * 2}h", pcm_data)
            mono = [(samples[i] + samples[i + 1]) // 2 for i in range(0, len(samples), 2)]
            pcm_data = struct.pack(f"<{len(mono)}h", *mono)

        if framerate != SAMPLE_RATE:
            print(f"  WARNING: File is {framerate}Hz, expected {SAMPLE_RATE}Hz")

        return pcm_data


async def audio_chunk_generator(pcm_data: bytes) -> AsyncIterator[bytes]:
    """Yield PCM audio in small chunks at real-time pace."""
    sleep_time = CHUNK_DURATION_MS / 1000

    for i in range(0, len(pcm_data), CHUNK_SIZE):
        chunk = pcm_data[i : i + CHUNK_SIZE]
        if chunk:
            yield chunk
        await asyncio.sleep(sleep_time * 0.9)


async def main(audio_path: str, output_path: str) -> None:
    """Run the full pipeline test."""
    path = Path(audio_path)
    if not path.exists():
        print(f"Error: File not found: {path}")
        sys.exit(1)

    # Validate all required API keys
    missing = []
    if not settings.assemblyai_api_key.strip():
        missing.append("ASSEMBLYAI_API_KEY")
    if not settings.cartesia_api_key.strip():
        missing.append("CARTESIA_API_KEY")
    if not settings.anthropic_api_key.strip():
        missing.append("ANTHROPIC_API_KEY")
    if missing:
        print(f"Error: Missing environment variables: {', '.join(missing)}")
        sys.exit(1)

    # Load audio
    pcm_data = read_wav_as_pcm(path)

    # Create pipeline components
    stt = STTClient(api_key=settings.assemblyai_api_key)
    tts = TTSClient(api_key=settings.cartesia_api_key)

    # Use a unique thread_id for this test so it starts fresh.
    # In the real app, thread_id would come from the WebSocket session.
    thread_id = f"test-pipeline-{uuid.uuid4().hex[:8]}"

    pipeline = VoicePipeline(stt=stt, tts=tts, thread_id=thread_id)

    print(f"Thread: {thread_id}")
    print("Running pipeline: Audio → STT → Agent → TTS")
    print("=" * 60)

    # Collect audio output
    audio_chunks: list[bytes] = []
    agent_text = ""

    async for event in pipeline.run(audio_chunk_generator(pcm_data)):
        if event.type == PipelineEventType.TRANSCRIPT_PARTIAL:
            display = event.text[:60] + "..." if len(event.text) > 60 else event.text
            print(f"\r  [partial] {display}", end="", flush=True)

        elif event.type == PipelineEventType.TRANSCRIPT_FINAL:
            print(f"\r{'':80}")  # Clear partial line
            print(f"  [PATIENT] {event.text}")
            print()
            print("  Agent thinking...")

        elif event.type == PipelineEventType.AUDIO:
            audio_chunks.append(event.audio)
            total_bytes = sum(len(c) for c in audio_chunks)
            duration = total_bytes / (SAMPLE_RATE * 2)
            print(f"\r  [audio] {duration:.1f}s received", end="", flush=True)

        elif event.type == PipelineEventType.AGENT_TEXT:
            agent_text = event.text

    print()
    print("=" * 60)

    # Show agent's text response
    if agent_text:
        print(f"\n  [AGENT] {agent_text}\n")

    # Save audio output
    if audio_chunks:
        pcm_out = b"".join(audio_chunks)
        total_samples = len(pcm_out) // 2
        duration = total_samples / SAMPLE_RATE

        output = Path(output_path)
        with wave.open(str(output), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm_out)

        print(f"Saved agent response: {output} ({duration:.2f}s)")
        print(f"Play with: afplay {output}")
    else:
        print("No audio received from the pipeline!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the full voice pipeline")
    parser.add_argument("audio_file", help="Path to a WAV file with a patient utterance")
    parser.add_argument(
        "-o", "--output",
        default="output_pipeline.wav",
        help="Output WAV file for the agent's response (default: output_pipeline.wav)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.audio_file, args.output))
