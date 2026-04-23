"""
Standalone test for the Cartesia TTS client.

Sends text to Cartesia for synthesis and saves the output as a WAV file.
This verifies the TTS client works in isolation before wiring it into
the full voice pipeline.

Usage:
    # Synthesize the default medical agent response:
    python -m app.voice.test_tts

    # Synthesize custom text:
    python -m app.voice.test_tts --text "Hello, how can I help you today?"

    # Specify output file:
    python -m app.voice.test_tts -o response.wav

    # Play immediately after generating (macOS):
    afplay output_tts.wav
"""

from __future__ import annotations

import asyncio
import argparse
import struct
import sys
import wave
from pathlib import Path
from typing import AsyncIterator

from app.config import settings
from app.voice.tts_client import TTSClient, SAMPLE_RATE


# A realistic multi-sentence agent response to test with.
# This exercises sentence boundary detection, natural prosody
# across sentences, and medical terminology pronunciation.
DEFAULT_TEXT = (
    "Based on your symptoms, I'd recommend seeing a neurologist. "
    "The sharp pains behind your eyes with flashing lights could "
    "indicate migraines or another neurological condition. "
    "I have several openings this week. "
    "Would you prefer a morning or afternoon appointment?"
)


async def text_token_generator(text: str) -> AsyncIterator[str]:
    """Simulate LLM-style token-by-token output.

    Splits the text into word-level chunks with small delays,
    mimicking how your LangGraph agent's stream_agent_response()
    yields tokens. This tests the TTS client's sentence buffering
    logic under realistic conditions.
    """
    words = text.split(" ")
    for i, word in enumerate(words):
        # Add the leading space back (except for the first word)
        token = f" {word}" if i > 0 else word
        yield token
        # Small delay to simulate LLM token generation (~30ms per token)
        await asyncio.sleep(0.03)


async def main(text: str, output_path: str) -> None:
    """Run the TTS test: synthesize text and save as WAV."""

    # Validate API key
    api_key = settings.cartesia_api_key.strip()
    if not api_key:
        print("Error: CARTESIA_API_KEY not set in backend/.env")
        print("Get a key at https://play.cartesia.ai/")
        sys.exit(1)

    print(f"Text to synthesize ({len(text)} chars):")
    print(f"  {text!r}")
    print()

    # Create client
    client = TTSClient(api_key=api_key)

    print("Streaming to Cartesia...")
    print("=" * 50)

    # Collect all audio chunks
    audio_chunks: list[bytes] = []
    chunk_count = 0

    async for audio_chunk in client.synthesize(text_token_generator(text)):
        audio_chunks.append(audio_chunk)
        chunk_count += 1
        # Show progress without flooding the console
        total_bytes = sum(len(c) for c in audio_chunks)
        duration = total_bytes / (SAMPLE_RATE * 2)  # 2 bytes per sample
        print(
            f"\r  Received {chunk_count} chunks, {total_bytes:,} bytes "
            f"({duration:.1f}s of audio)",
            end="",
            flush=True,
        )

    print()
    print("=" * 50)

    if not audio_chunks:
        print("Error: No audio received from Cartesia!")
        sys.exit(1)

    # Combine all chunks into one PCM buffer
    pcm_data = b"".join(audio_chunks)
    total_samples = len(pcm_data) // 2  # 16-bit = 2 bytes per sample
    duration = total_samples / SAMPLE_RATE

    print(f"Total audio: {len(pcm_data):,} bytes, {duration:.2f}s")

    # Save as WAV file
    output = Path(output_path)
    with wave.open(str(output), "wb") as wf:
        wf.setnchannels(1)  # mono
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_data)

    print(f"Saved to: {output}")
    print()
    print(f"Play with:  afplay {output}  (macOS)")
    print(f"        or: ffplay -f s16le -ar {SAMPLE_RATE} -ac 1 {output}  (any OS)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Cartesia TTS client")
    parser.add_argument(
        "--text",
        default=DEFAULT_TEXT,
        help="Text to synthesize (default: medical agent response)",
    )
    parser.add_argument(
        "-o", "--output",
        default="output_tts.wav",
        help="Output WAV file path (default: output_tts.wav)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.text, args.output))
