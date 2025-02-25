import os
import asyncio
import base64
import json
import numpy as np
import sounddevice as sd
import websockets
from dotenv import load_dotenv

##############################
# 1) AUDIO PROCESSING (Real-Time)
##############################
class AudioProcessor:
    def __init__(self, sample_rate=24000):
        self.sample_rate = sample_rate
        self.vad_threshold = 0.015
        self.interrupt_threshold = 0.02
        self.speech_frames = 0
        self.silence_frames = 0
        self.min_speech_duration = int(0.3 * sample_rate)
        self.max_silence_duration = int(0.8 * sample_rate)
        self.main_buffer = []
        self.interrupt_buffer = []
        self.is_speaking = False
        self.speech_detected = False
        self.is_interrupting = False

    def process_audio(self, indata):
        """Process incoming audio, detecting speech and interruptions."""
        audio_level = np.abs(indata).mean() / 32768.0
        if self.is_speaking and audio_level > self.interrupt_threshold:
            self.is_interrupting = True
            self.interrupt_buffer.extend(indata.tobytes())
            return
        if self.is_interrupting:
            self.interrupt_buffer.extend(indata.tobytes())
            return
        if not self.is_speaking:
            if audio_level > self.vad_threshold:
                self.speech_detected = True
                self.speech_frames += len(indata)
                self.silence_frames = 0
                self.main_buffer.extend(indata.tobytes())
            elif self.speech_detected:
                self.silence_frames += len(indata)
                if self.silence_frames < self.max_silence_duration:
                    self.main_buffer.extend(indata.tobytes())

    def check_interruption(self):
        return self.is_interrupting

    def get_interrupt_audio(self):
        if not self.interrupt_buffer:
            return None
        audio_data = bytes(self.interrupt_buffer)
        self.interrupt_buffer.clear()
        self.is_interrupting = False
        return audio_data

    def should_process(self):
        """Decide whether enough speech has been captured to process."""
        return (self.speech_detected and 
                self.speech_frames >= self.min_speech_duration and 
                self.silence_frames >= self.max_silence_duration)

    def reset(self):
        """Reset the speech buffer and counters for the next turn."""
        self.speech_frames = 0
        self.silence_frames = 0
        self.speech_detected = False
        audio_data = bytes(self.main_buffer)
        self.main_buffer.clear()
        return audio_data

##############################
# 2) ORCHESTRATOR (Natural Language Understanding)
##############################
# For demonstration, here’s a minimal orchestrator that “understands” conversation.
# In a real system you might integrate with a full multi-agent framework.
class AutoGenOrchestrator:
    async def handle_user_text(self, user_text: str) -> str:
        # Here the agent is expected to understand conversation naturally.
        # For demonstration, we simply echo or branch based on keywords.
        if "weather" in user_text.lower():
            return "The weather in Istanbul is 22°C and sunny."
        elif "code" in user_text.lower():
            return "Here's a Python snippet: def greet(name): return f'Hello, {name}!'"
        else:
            return f"I heard you say: {user_text}. How can I help further?"

##############################
# 3) CONVERSATION SYSTEM (Real-Time API Integration)
##############################
class ConversationSystem:
    def __init__(self, orchestrator: AutoGenOrchestrator):
        load_dotenv()
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("AZURE_OPENAI_API_KEY not found in environment")
        # Hardcoded WebSocket URL (matching your working example)
        self.url = (
            "wss://aoai-ep-swedencentral02.openai.azure.com/openai/realtime?"
            f"api-version=2024-10-01-preview&deployment=gpt-4o-realtime-preview&api-key={self.api_key}"
        )
        print(f"DEBUG: WebSocket URL = {self.url}")
        print(f"DEBUG: API Key Loaded? {'Yes' if self.api_key else 'No'}")
        self.audio_processor = AudioProcessor()
        self.streams = {'input': None, 'output': None}
        self.orchestrator = orchestrator

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Audio input error: {status}")
        self.audio_processor.process_audio(indata)

    async def setup_audio(self):
        """Initialize audio input and output streams."""
        self.streams['output'] = sd.OutputStream(samplerate=24000, channels=1, dtype=np.int16)
        self.streams['input'] = sd.InputStream(
            samplerate=24000, channels=1, dtype=np.int16,
            callback=self.audio_callback, blocksize=4800
        )
        for stream in self.streams.values():
            stream.start()

    async def setup_websocket_session(self, websocket):
        """Set up session parameters for Azure Real-Time API."""
        session_config = {
            "type": "session.update",
            "session": {
                "voice": "alloy",
                "instructions": "You are a helpful AI assistant. Keep responses brief and engaging.",
                "modalities": ["audio", "text"],
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.3,
                    "prefix_padding_ms": 150,
                    "silence_duration_ms": 600
                }
            }
        }
        await websocket.send(json.dumps(session_config))
        while True:
            response = json.loads(await websocket.recv())
            if response.get("type") == "session.created":
                print("Session setup complete")
                break
            elif response.get("type") == "error":
                raise Exception(f"Session setup failed: {response}")

    async def send_audio(self, websocket, audio_data):
        """Send captured audio to Azure for transcription and response."""
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        await websocket.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": audio_base64
        }))
        await websocket.send(json.dumps({"type": "input_audio_buffer.commit"}))
        await websocket.send(json.dumps({
            "type": "response.create",
            "response": {"modalities": ["audio", "text"]}
        }))

    async def handle_response(self, websocket):
        """Handle incoming response from Azure (both audio and text)."""
        self.audio_processor.is_speaking = True
        recognized_text = ""
        try:
            while True:
                response = json.loads(await websocket.recv())
                # Accumulate text if available
                if response.get("type") == "response.text.delta":
                    recognized_text += response.get("delta", "")
                # Process audio and play it
                elif response.get("type") == "response.audio.delta":
                    if "delta" in response:
                        try:
                            audio_data = response["delta"].strip()
                            padding = -len(audio_data) % 4
                            if padding:
                                audio_data += "=" * padding
                            audio_chunk = np.frombuffer(
                                base64.b64decode(audio_data),
                                dtype=np.int16
                            )
                            self.streams['output'].write(audio_chunk)
                        except Exception as e:
                            print(f"Audio processing error: {e}")
                elif response.get("type") == "response.done":
                    break
        finally:
            self.audio_processor.is_speaking = False

        # Process the recognized text naturally
        if recognized_text.strip():
            print(f"\n[DEBUG] Recognized text: {recognized_text}")
            result = await self.orchestrator.handle_user_text(recognized_text)
            print(f"[DEBUG] Orchestrator response: {result}")
            # Optionally, send the result back for TTS (text-to-speech)
            await websocket.send(json.dumps({
                "type": "response.create",
                "response": {"modalities": ["audio", "text"], "text": result}
            }))
            # Play TTS audio from Azure until response.done is received
            while True:
                resp = json.loads(await websocket.recv())
                if resp.get("type") == "response.audio.delta":
                    audio_data = resp["delta"].strip()
                    pad = -len(audio_data) % 4
                    if pad:
                        audio_data += "=" * pad
                    audio_chunk = np.frombuffer(
                        base64.b64decode(audio_data),
                        dtype=np.int16
                    )
                    self.streams['output'].write(audio_chunk)
                elif resp.get("type") == "response.done":
                    break

    async def run(self):
        """Main conversation loop: capture audio, send it, and handle responses."""
        await self.setup_audio()
        print("Audio setup complete. Connecting to Real-Time...")
        async with websockets.connect(self.url) as ws:
            await self.setup_websocket_session(ws)
            print("Ready for conversation.")
            while True:
                if self.audio_processor.should_process():
                    audio_data = self.audio_processor.reset()
                    await self.send_audio(ws, audio_data)
                    await self.handle_response(ws)
                await asyncio.sleep(0.05)

##############################
# 4) Putting It All Together
##############################
async def main():
    load_dotenv()
    # In a real implementation, you would integrate a full AzureOpenAIChatCompletionClient
    # Here we use our simple orchestrator
    orchestrator = AutoGenOrchestrator()
    system = ConversationSystem(orchestrator)
    await system.run()

if __name__ == "__main__":
    asyncio.run(main())
