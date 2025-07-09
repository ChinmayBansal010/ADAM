import asyncio
import edge_tts
from pydub import AudioSegment
from pydub.playback import play
from io import BytesIO

async def test_tts():
    text = "This is a test of the edge TTS engine."
    audio_stream = BytesIO()
    communicate = edge_tts.Communicate("This is a test", "en-US-GuyNeural")

    received_audio = False
    async for chunk in communicate.stream():
        if chunk.get("type") == "audio" and chunk.get("data"):
            audio_stream.write(chunk["data"])
            received_audio = True

    if not received_audio:
        print("❌ No audio received. Check internet or engine.")
        return

    audio_stream.seek(0)
    audio = AudioSegment.from_file(audio_stream, format="mp3")
    play(audio)

asyncio.run(test_tts())