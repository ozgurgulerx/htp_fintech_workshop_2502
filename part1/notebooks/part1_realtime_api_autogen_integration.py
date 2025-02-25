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
"""
We use AutoGen (with AzureOpenAIChatCompletionClient) to manage multiple agents (sub-experts).
For demonstration, let’s define two minimal agents:
  - WeatherAgent  -> handles weather info
  - CodeAgent     -> handles code-related requests

We’ll build an orchestrator that picks the right agent and returns the final text response.
"""

# Basic AutoGen imports
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat

# Azure OpenAI client for AutoGen
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

class WeatherAgent(AssistantAgent):
    """Simplified agent that returns weather data."""
    def handle_custom(self, user_text: str) -> str:
        # Real logic might call an external weather API
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
    A miniature 'manual' orchestrator that decides which agent to consult
    based on the user's text. In a more advanced flow, you'd let AutoGen
    dynamically route between sub-agents with something like RoundRobinGroupChat
    or function calling.
    """
    def __init__(self, azure_client):
        # Create sub-agents with specialized system instructions
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
        # Super simplistic keyword-based routing:
        if "weather" in user_text.lower():
            return await self._run_agent(self.weather_agent, user_text)
        elif "code" in user_text.lower():
            return await self._run_agent(self.code_agent, user_text)
        else:
            return "Try asking for weather or code?"

    async def _run_agent(self, agent: AssistantAgent, user_text: str) -> str:
        """
        We call agent's 'run' method or a custom function to get the response.
        The agent might do:
         - Summarize the user’s text
         - Possibly do function calling or full chain-of-thought
        In this toy example, we call a custom handle_custom function.
        """
        # For demonstration, let's just call a custom method
        # If you rely on open-ended conversation, you'd do something like:
        #   response = await agent.run(task=user_text)
        #   final_text = response['content']
        # But here, we do it manually:
        final_text = agent.handle_custom(user_text)
        return final_text


##############################
# 2) AUDIO PROCESSING (Real-Time)
##############################
"""
This section is similar to your Real-Time code sample, but we'll also demonstrate
how to tie in text from Azure’s transcription back into our AutoGenOrchestrator.
"""

class AudioProcessor:
    """Same logic you already have for capturing user audio and detecting interruptions."""

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

        # Interruption if AI is speaking
        if self.is_speaking and audio_level > self.interrupt_threshold:
            self.is_interrupting = True
            self.interrupt_buffer.extend(indata.tobytes())
            return

        if self.is_interrupting:
            self.interrupt_buffer.extend(indata.tobytes())
            return

        # Normal user speech detection
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
    - On receiving recognized text from the user, we hand it off to AutoGenOrchestrator.
    - Then we take AutoGen's text response and request Azure to produce spoken audio output.
    """

    def __init__(self, orchestrator: AutoGenOrchestrator):
        load_dotenv()
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("AZURE_OPENAI_API_KEY not found")
        
        # Real-Time endpoint
        # (Change region/deployment to your own)
        self.url = (
            "wss://YOUR-ENDPOINT-HERE.openai.azure.com/openai/realtime?"
            "api-version=2024-10-01-preview&deployment=gpt-4o-realtime-preview&"
            f"api-key={self.api_key}"
        )

        self.audio_processor = AudioProcessor()
        self.streams = {'input': None, 'output': None}
        self.orchestrator = orchestrator

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Audio error: {status}")
            return
        self.audio_processor.process_audio(indata)

    async def setup_audio(self):
        # Use sounddevice or PyAudio to open input and output streams
        import sounddevice as sd
        self.streams['output'] = sd.OutputStream(
            samplerate=24000, channels=1, dtype=np.int16
        )
        self.streams['input'] = sd.InputStream(
            samplerate=24000, channels=1, dtype=np.int16,
            callback=self.audio_callback, blocksize=4800
        )
        for stream in self.streams.values():
            stream.start()

    async def setup_websocket_session(self, websocket):
        """Tell Azure Real-Time how we want to handle input/output, and wait for 'session.created'."""
        session_config = {
            "type": "session.update",
            "session": {
                "voice": "alloy",
                "instructions": "You are a helpful real-time AI assistant. Keep responses brief.",
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
            msg = await websocket.recv()
            response = json.loads(msg)
            if response["type"] == "session.created":
                print("Azure Real-Time session created.")
                break
            if response["type"] == "error":
                raise Exception(f"Session setup error: {response}")

    async def send_audio_to_azure(self, websocket, audio_data: bytes):
        """
        We send user audio to Azure's Real-Time API. Azure will do speech-to-text
        and eventually respond with text & audio output.
        """
        audio_b64 = base64.b64encode(audio_data).decode('utf-8')
        await websocket.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": audio_b64
        }))
        await websocket.send(json.dumps({"type": "input_audio_buffer.commit"}))

        # Usually you'd request a new AI "response"
        # But we might NOT do it here if we want to do manual orchestrations.
        # For a direct flow, we'd do:
        await websocket.send(json.dumps({
            "type": "response.create",
            "response": {"modalities": ["audio", "text"]}
        }))

    async def handle_responses(self, websocket):
        """
        Azure streams back partial text & audio. 
        We intercept the text, route it to the orchestrator, then ask Azure for TTS of the orchestrator's result.
        """
        self.audio_processor.is_speaking = True
        try:
            recognized_user_text = ""
            while True:
                if self.audio_processor.check_interruption():
                    # The user started speaking again mid-response
                    interrupt_audio = self.audio_processor.get_interrupt_audio()
                    if interrupt_audio:
                        print("User interrupted the AI mid-response!")
                        # Cancel current response
                        await websocket.send(json.dumps({"type": "response.cancel"}))
                        # Immediately handle the new user audio
                        await self.send_audio_to_azure(websocket, interrupt_audio)
                        return  # go handle that new chunk

                msg = await websocket.recv()
                response = json.loads(msg)

                # "response.text.delta" is partial recognized text from user (STT) or partial text from AI?
                # In real usage, you'd check docs to confirm. 
                # For example, if the user is speaking, you see partial transcripts of user speech,
                # or if the AI is generating text, you see partial text from the AI. 
                # This snippet is conceptual, so we do the simplest check.
                
                if response["type"] == "response.text.delta":
                    # E.g. partial text recognized from the user or partial text from the AI
                    recognized_user_text += response["delta"]
                    
                elif response["type"] == "response.done":
                    break

                elif response["type"] == "response.audio.delta":
                    # This is the AI responding with TTS
                    audio_b64 = response["delta"].strip()
                    padding = -len(audio_b64) % 4
                    if padding:
                        audio_b64 += "=" * padding
                    audio_chunk = np.frombuffer(
                        base64.b64decode(audio_b64),
                        dtype=np.int16
                    )
                    self.streams['output'].write(audio_chunk)

            # At this point we presumably have the user text fully recognized
            print("User said:", recognized_user_text)

            # Send that recognized text to AutoGen
            if recognized_user_text.strip():
                orchestrator_answer = await self.orchestrator.handle_user_text(recognized_user_text)
                print("AutoGen orchestrator says:", orchestrator_answer)

                # Now request TTS from Azure for the orchestrator's final text
                # We do that by sending a new "response.create" with text input?
                # The Real-Time spec might differ. Another approach:
                await websocket.send(json.dumps({
                    "type": "response.create",
                    "response": {
                        "modalities": ["audio", "text"],
                        # Possibly there's a "text" field to request TTS of a custom user-provided text.
                        "text": orchestrator_answer
                    }
                }))

                # Then read the audio deltas from Azure until "response.done" again
                while True:
                    msg2 = await websocket.recv()
                    resp2 = json.loads(msg2)
                    if resp2["type"] == "response.audio.delta":
                        audio_b64 = resp2["delta"].strip()
                        if audio_b64:
                            pad = -len(audio_b64) % 4
                            if pad:
                                audio_b64 += "=" * pad
                            audio_chunk = np.frombuffer(
                                base64.b64decode(audio_b64),
                                dtype=np.int16
                            )
                            self.streams['output'].write(audio_chunk)
                    elif resp2["type"] == "response.done":
                        break

        finally:
            self.audio_processor.is_speaking = False

    async def run(self):
        """Main conversation loop: set up audio, connect to Real-Time, and wait for user speech."""
        await self.setup_audio()
        print("Audio setup complete. Connecting to Real-Time...")

        async with websockets.connect(self.url) as ws:
            await self.setup_websocket_session(ws)
            print("Ready for conversation. Speak away!")

            while True:
                # If the user has finished speaking a chunk
                if self.audio_processor.should_process():
                    audio_data = self.audio_processor.reset()
                    # Send user chunk to Azure for STT
                    await self.send_audio_to_azure(ws, audio_data)
                    # Now handle streaming AI response (including recognized text, TTS, etc.)
                    await self.handle_responses(ws)

                # The loop continues listening
                await asyncio.sleep(0.05)


##############################
# 4) Putting It All Together
##############################

async def main():
    # 1) Load environment variables for AutoGen
    load_dotenv()
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")

    # 2) Create the AzureOpenAIChatCompletionClient for AutoGen
    azure_client = AzureOpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_version="2024-06-01",
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key
    )

    # 3) Create our orchestrator with sub-agents
    orchestrator = AutoGenOrchestrator(azure_client)

    # 4) Create the conversation system that does real-time audio
    system = ConversationSystem(orchestrator)

    # 5) Run it
    await system.run()

if __name__ == "__main__":
    asyncio.run(main())
