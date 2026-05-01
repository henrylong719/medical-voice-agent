"""
Standalone test for the AssemblyAI STT client.

Streams a WAV file (or raw PCM file) to AssemblyAI and prints transcript
events as they arrive. This verifies the STT client works in isolation
before wiring it into the full voice pipeline.

Usage:
    # With a WAV file (automatically extracts PCM):
    python -m app.voice.test_stt path/to/audio.wav

    # With raw PCM (16kHz, 16-bit, mono):
    python -m app.voice.test_stt path/to/audio.pcm --raw

    # Generate a test WAV file using text-to-speech (macOS):
    say -o test.wav --data-format=LEI16@16000 "I have a headache and my vision is blurry"
    python -m app.voice.test_stt test.wav

The script reads the audio file and feeds it to AssemblyAI in small chunks,
simulating real-time microphone input. It sleeps between chunks to approximate
real-time pacing — sending audio faster than real-time can cause AssemblyAI
to reject the session.
"""

from __future__ import annotations

import asyncio
import argparse
import struct
import sys
import wave
from pathlib import Path
from typing import AsyncIterator

from app.core.config import settings
from app.voice.stt_client import STTClient, SAMPLE_RATE


# How much audio to send per chunk. 50ms is a good balance:
# small enough for low latency, large enough to avoid overhead
# from too many WebSocket frames.
CHUNK_DURATION_MS = 50
BYTES_PER_SAMPLE = 2  # 16-bit = 2 bytes
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000) * BYTES_PER_SAMPLE
# At 16kHz, 16-bit: 50ms = 800 samples = 1600 bytes


def read_wav_as_pcm(path: Path) -> bytes:
    """Read a WAV file and return raw PCM bytes.

    Validates that the WAV file matches our expected format
    (16kHz, 16-bit, mono). Raises ValueError if it doesn't.
    """
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()

        print(f"WAV file: {path.name}")
        print(f"  Channels: {channels}")
        print(f"  Sample width: {sample_width} bytes ({sample_width * 8}-bit)")
        print(f"  Sample rate: {framerate} Hz")
        print(f"  Frames: {n_frames}")
        print(f"  Duration: {n_frames / framerate:.2f}s")
        print()

        if sample_width != BYTES_PER_SAMPLE:
            raise ValueError(
                f"Expected {BYTES_PER_SAMPLE * 8}-bit audio, got {sample_width * 8}-bit"
            )

        pcm_data = wf.readframes(n_frames)

        # If stereo, downmix to mono by averaging channels
        if channels == 2:
            print("  Converting stereo to mono...")
            samples = struct.unpack(f"<{n_frames * 2}h", pcm_data)
            mono_samples = [
                (samples[i] + samples[i + 1]) // 2 for i in range(0, len(samples), 2)
            ]
            pcm_data = struct.pack(f"<{len(mono_samples)}h", *mono_samples)

        # If sample rate doesn't match, warn (we won't resample here —
        # for testing, just use a file recorded at 16kHz)
        if framerate != SAMPLE_RATE:
            print(f"  WARNING: File is {framerate}Hz but STT expects {SAMPLE_RATE}Hz.")
            print(f"  Transcription may be inaccurate. Re-record at {SAMPLE_RATE}Hz.")
            print()

        return pcm_data


async def audio_chunk_generator(
    pcm_data: bytes,
    realtime_pace: bool = True,
) -> AsyncIterator[bytes]:
    """Yield PCM audio in small chunks, optionally paced to real-time.

    This simulates what a real microphone source would do: producing
    small chunks of audio at regular intervals. Without pacing,
    we'd blast the entire file at AssemblyAI in milliseconds, which
    doesn't match how their streaming model expects to receive audio.

    Args:
        pcm_data: Raw PCM bytes (16kHz, 16-bit, mono).
        realtime_pace: If True, sleep between chunks to simulate
            real-time audio. Set to False only for quick testing.
    """
    total_chunks = (len(pcm_data) + CHUNK_SIZE - 1) // CHUNK_SIZE
    sleep_time = CHUNK_DURATION_MS / 1000  # 50ms in seconds
    sent = 0

    for i in range(0, len(pcm_data), CHUNK_SIZE):
        chunk = pcm_data[i : i + CHUNK_SIZE]
        if chunk:  # Don't send empty trailing chunk
            yield chunk
            sent += 1

        if realtime_pace:
            # Sleep to simulate real-time audio pacing.
            # We sleep slightly less than the chunk duration to
            # account for processing overhead and keep up with
            # real-time. Sending faster than 1x real-time is fine
            # for short bursts; AssemblyAI only rejects sustained
            # faster-than-real-time streaming.
            await asyncio.sleep(sleep_time * 0.9)


async def main(audio_path: str, is_raw: bool = False) -> None:
    """Run the STT test: stream a file and print transcript events."""
    path = Path(audio_path)

    if not path.exists():
        print(f"Error: File not found: {path}")
        sys.exit(1)

    # Validate API key
    api_key = settings.ASSEMBLYAI_API_KEY.strip()
    if not api_key:
        print("Error: ASSEMBLYAI_API_KEY not set in backend/.env")
        print("Get a free key at https://www.assemblyai.com/dashboard")
        sys.exit(1)

    # Load audio
    if is_raw:
        print(f"Reading raw PCM file: {path.name}")
        pcm_data = path.read_bytes()
        duration = len(pcm_data) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
        print(f"  Size: {len(pcm_data)} bytes, Duration: {duration:.2f}s")
        print()
    else:
        pcm_data = read_wav_as_pcm(path)

    # Create client and stream
    client = STTClient(api_key=api_key)

    print("Streaming to AssemblyAI...")
    print("=" * 50)

    final_count = 0
    partial_count = 0

    # This is the async generator pattern in action:
    # audio_chunk_generator yields audio bytes →
    # client.transcribe consumes them and yields TranscriptEvents
    async for event in client.transcribe(audio_chunk_generator(pcm_data)):
        if event.is_final:
            final_count += 1
            # Clear the partial line and print the final transcript
            print(f"\r{'':80}")
            print(f"  FINAL (turn {event.turn_order}): {event.text!r}")
            print(f"  Confidence: {event.confidence:.4f}")
            print()
        else:
            partial_count += 1
            # Overwrite the current line with the latest partial
            display = event.text[:70] + "..." if len(event.text) > 70 else event.text
            print(f"\r  partial: {display}", end="", flush=True)

    print("=" * 50)
    print(f"Done! {final_count} final transcript(s), {partial_count} partial update(s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test AssemblyAI STT client")
    parser.add_argument("audio_file", help="Path to a WAV or raw PCM audio file")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Treat the file as raw PCM (16kHz, 16-bit, mono) instead of WAV",
    )
    args = parser.parse_args()
    asyncio.run(main(args.audio_file, is_raw=args.raw))
