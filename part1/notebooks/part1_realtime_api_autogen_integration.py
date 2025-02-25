import os
import asyncio
import base64
import json
import numpy as np
import sounddevice as sd
import websockets
from dotenv import load_dotenv

##############################
# 1) AUTO-GEN ORCHESTRATOR
##############################
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

class WeatherAgent(AssistantAgent):
    """Simplified agent that returns weather data."""
    def handle_custom(self, user_text: str) -> str:
        return "It’s 22°C and sunny in Istanbul."

class CodeAgent(AssistantAgent):
    """Simplified agent that returns sample Python code."""
    def handle_custom(self, user_text: str) -> str:
        return (
            "def greet(name):\n"
            "    return f'Hello, {name}!'"
        )

class AutoGenOrchestrator:
    """
    Orchestrator that routes user queries to the appropriate agent.
    """
    def __init__(self, azure_client):
        self.weather_agent = WeatherAgent(
            name="WeatherAgent",
            model_client=azure_client,
            system_message="You are a weather assistant."
        )
        self.code_agent = CodeAgent(
            name="CodeAgent",
            model_client=azure_client,
            system_message="You are a code-generating assistant."
        )

    async def handle_user_text(self, user_text: str) -> str:
        if "weather" in user_text.lower():
            return await self._run_agent(self.weather_agent, user_text)
        elif "code" in user_text.lower():
            return await self._run_agent(self.code_agent, user_text)
        else:
            return "Try asking for weather or code?"

    async def _run_agent(self, agent: AssistantAgent, user_text: str) -> str:
        final_text = agent.handle_custom(user_text)
        return final_text

##############################
# 2) AUDIO PROCESSING (Real-Time)
##############################
class AudioProcessor:
    """Handles audio input, buffering, and detecting interruptions."""
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
        return (self.speech_detected and
                self.speech_frames >= self.min_speech_duration and
                self.silence_frames >= self.max_silence_duration)

    def reset(self):
        self.speech_frames = 0
        self.silence_frames = 0
        self.speech_detected = False
        audio_data = bytes(self.main_buffer)
        self.main_buffer.clear()
        return audio_data

##############################
# 3) CONVERSATION SYSTEM
##############################
class ConversationSystem:
    """
    - Connects to Azure Real-Time for 2-way audio streaming.
    - Sends user audio to Azure and processes the AI's streaming response.
    - Hands off recognized text to AutoGenOrchestrator if needed.
    """
    def __init__(self, orchestrator: AutoGenOrchestrator):
        load_dotenv()
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("AZURE_OPENAI_API_KEY not found in environment")

        # Hardcoded WebSocket URL as per your working example:
        self.url = (
            f"wss://aoai-ep-swedencentral02.openai.azure.com/openai/realtime?"
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
        import sounddevice as sd
        self.streams['output'] = sd.OutputStream(samplerate=24000, channels=1, dtype=np.int16)
        self.streams['input'] = sd.InputStream(samplerate=24000, channels=1, dtype=np.int16,
                                               callback=self.audio_callback, blocksize=4800)
        for stream in self.streams.values():
            stream.start()

    async def send_audio_to_azure(self, websocket, audio_data: bytes):
        audio_b64 = base64.b64encode(audio_data).decode('utf-8')
        await websocket.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": audio_b64
        }))
        await websocket.send(json.dumps({"type": "input_audio_buffer.commit"}))
        await websocket.send(json.dumps({
            "type": "response.create",
            "response": {"modalities": ["audio", "text"]}
        }))

    async def handle_response(self, websocket):
        """Continuously receive and process the AI's audio response."""
        self.audio_processor.is_speaking = True
        try:
            while True:
                response = await websocket.recv()
                data = json.loads(response)
                if data["type"] == "response.audio.delta":
                    if "delta" in data:
                        try:
                            audio_data = data["delta"].replace(" ", "").replace("\n", "")
                            padding = len(audio_data) % 4
                            if padding:
                                audio_data += "=" * padding
                            audio_bytes = base64.b64decode(audio_data)
                            audio_chunk = np.frombuffer(audio_bytes, dtype=np.int16)
                            self.streams['output'].write(audio_chunk)
                            print(".", end="", flush=True)
                        except Exception as e:
                            print(f"Error processing audio: {e}")
                elif data["type"] == "response.done":
                    break
        finally:
            self.audio_processor.is_speaking = False

    async def run(self):
        """Main conversation loop: set up audio, connect to Azure Real-Time, send audio, and process responses."""
        await self.setup_audio()
        print("Audio setup complete. Connecting to Real-Time...")
        async with websockets.connect(self.url) as ws:
            print("Connected to Azure Real-Time API.")
            while True:
                if self.audio_processor.should_process():
                    audio_data = self.audio_processor.reset()
                    await self.send_audio_to_azure(ws, audio_data)
                    # Process and play the AI's response
                    await self.handle_response(ws)
                await asyncio.sleep(0.05)

##############################
# 4) Putting It All Together
##############################
async def main():
    load_dotenv()
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_client = AzureOpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_version="2024-06-01",
        azure_endpoint="https://aoai-ep-swedencentral02.openai.azure.com",
        api_key=azure_api_key
    )
    orchestrator = AutoGenOrchestrator(azure_client)
    system = ConversationSystem(orchestrator)
    await system.run()

if __name__ == "__main__":
    print("Starting real-time conversation system...")
    asyncio.run(main())
