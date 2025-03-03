import asyncio
import os
import base64
import json
import numpy as np
import sounddevice as sd
import websockets
from dotenv import load_dotenv

class AudioProcessor:
    def __init__(self, sample_rate=24000):
        # Basic audio parameters
        self.sample_rate = sample_rate
        self.vad_threshold = 0.015
        self.interrupt_threshold = 0.02
        
        # Frame tracking
        self.speech_frames = 0
        self.silence_frames = 0
        self.min_speech_duration = int(0.3 * sample_rate)
        self.max_silence_duration = int(0.8 * sample_rate)
        
        # Audio buffers - now we have two
        self.main_buffer = []
        self.interrupt_buffer = []
        
        # State tracking
        self.is_speaking = False
        self.speech_detected = False
        self.is_interrupting = False

    def process_audio(self, indata):
        """Process incoming audio, handling both normal speech and interruptions"""
        audio_level = np.abs(indata).mean() / 32768.0
        
        # If we're currently speaking and detect a potential interruption
        if self.is_speaking and audio_level > self.interrupt_threshold:
            self.is_interrupting = True
            self.interrupt_buffer.extend(indata.tobytes())
            return
            
        # If we're collecting interrupted speech
        if self.is_interrupting:
            self.interrupt_buffer.extend(indata.tobytes())
            return
            
        # Normal speech processing
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
        """Check if we're currently in an interruption state"""
        return self.is_interrupting

    def get_interrupt_audio(self):
        """Get the interruption audio if available"""
        if not self.interrupt_buffer:
            return None
        audio_data = bytes(self.interrupt_buffer)
        self.interrupt_buffer.clear()
        self.is_interrupting = False
        return audio_data

    def should_process(self):
        """Check if we have enough speech to process"""
        return (self.speech_detected and 
                self.speech_frames >= self.min_speech_duration and 
                self.silence_frames >= self.max_silence_duration)

    def reset(self):
        """Reset the main speech buffer and state"""
        self.speech_frames = 0
        self.silence_frames = 0
        self.speech_detected = False
        audio_data = bytes(self.main_buffer)
        self.main_buffer.clear()
        return audio_data

class ConversationSystem:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("AZURE_OPENAI_API_KEY not found")
            
        self.url = (
            "wss://aoai-ep-swedencentral02.openai.azure.com/openai/realtime?"
            f"api-version=2024-10-01-preview&deployment=gpt-4o-realtime-preview&"
            f"api-key={self.api_key}"
        )
        
        self.audio_processor = AudioProcessor()
        self.streams = {'input': None, 'output': None}

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Audio error: {status}")
            return
        self.audio_processor.process_audio(indata)

    async def setup_audio(self):
        """Initialize audio streams"""
        self.streams['output'] = sd.OutputStream(
            samplerate=24000, channels=1, dtype=np.int16)
        self.streams['input'] = sd.InputStream(
            samplerate=24000, channels=1, dtype=np.int16,
            callback=self.audio_callback, blocksize=4800)
            
        for stream in self.streams.values():
            stream.start()

    async def setup_websocket_session(self, websocket):
        """Initialize the conversation session"""
        session_config = {
            "type": "session.update",
            "session": {
                "voice": "alloy",
                "instructions": "You are a helpful AI assistant. Keep responses brief.",
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
            if response["type"] == "session.created":
                break
            if response["type"] == "error":
                raise Exception(f"Session setup failed: {response}")

    async def send_audio(self, websocket, audio_data):
        """Send audio data to the API"""
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        # Send the audio data
        await websocket.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": audio_base64
        }))
        await websocket.send(json.dumps({"type": "input_audio_buffer.commit"}))
        
        # Request a response
        await websocket.send(json.dumps({
            "type": "response.create",
            "response": {"modalities": ["audio", "text"]}
        }))

    async def handle_response(self, websocket):
        """Handle AI response with interruption support"""
        self.audio_processor.is_speaking = True
        try:
            while True:
                if self.audio_processor.check_interruption():
                    # Get the interruption audio
                    interrupt_audio = self.audio_processor.get_interrupt_audio()
                    if interrupt_audio:
                        print("Interrupted!")
                        # Cancel current response
                        await websocket.send(json.dumps({"type": "response.cancel"}))
                        # Send the interruption audio immediately
                        await self.send_audio(websocket, interrupt_audio)
                        break
                
                response = json.loads(await websocket.recv())
                
                if response["type"] == "response.audio.delta":
                    if "delta" in response:
                        try:
                            audio_data = response["delta"].strip()
                            padding = -len(audio_data) % 4
                            if padding:
                                audio_data += "=" * padding
                            
                            audio = np.frombuffer(
                                base64.b64decode(audio_data), 
                                dtype=np.int16
                            )
                            self.streams['output'].write(audio)
                            
                        except Exception as e:
                            print(f"Audio processing error: {e}")
                            
                elif response["type"] == "response.done":
                    break
                    
        finally:
            self.audio_processor.is_speaking = False

    async def run(self):
        """Main conversation loop"""
        await self.setup_audio()
        print("Audio setup complete")
        
        async with websockets.connect(self.url) as ws:
            await self.setup_websocket_session(ws)
            print("Ready for conversation")
            
            while True:
                if self.audio_processor.should_process():
                    audio_data = self.audio_processor.reset()
                    await self.send_audio(ws, audio_data)
                    await self.handle_response(ws)
                await asyncio.sleep(0.05)

if __name__ == "__main__":
    system = ConversationSystem()
    asyncio.run(system.run())