import asyncio
import os
import base64
import json
from dotenv import load_dotenv
import websockets
import numpy as np
import sounddevice as sd
import socket  # for DNS test

class ConversationSystem:
    def __init__(self):
        load_dotenv()

        self.input_stream = None
        self.output_stream = None
        self.is_speaking = False
        self.input_buffer = []

        # Grab environment variables
        self.openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")  # e.g. "my-res.openai.azure.com"
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-realtime-preview")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-01-preview")

        # Print out for troubleshooting
        print(f"[DEBUG] AZURE_OPENAI_ENDPOINT: {self.openai_endpoint}")
        print(f"[DEBUG] AZURE_OPENAI_API_KEY: {'<redacted>' if self.api_key else 'None found'}")
        print(f"[DEBUG] AZURE_OPENAI_DEPLOYMENT: {self.deployment_name}")
        print(f"[DEBUG] AZURE_OPENAI_API_VERSION: {self.api_version}")

        if not self.openai_endpoint:
            raise ValueError("Missing environment variable: AZURE_OPENAI_ENDPOINT")
        if not self.api_key:
            raise ValueError("Missing environment variable: AZURE_OPENAI_API_KEY")

        # Build WebSocket URL
        self.url = (
            f"wss://{self.openai_endpoint}/openai/realtime?"
            f"api-version={self.api_version}&deployment={self.deployment_name}&api-key={self.api_key}"
        )
        # Print the final WSS URL for debugging
        print(f"[DEBUG] Real-Time WebSocket URL: {self.url}")

        # Attempt DNS resolution for the endpoint to detect "nodename nor servname" issues
        try:
            socket.gethostbyname(self.openai_endpoint.split("/")[0])
            print(f"[DEBUG] Successfully resolved {self.openai_endpoint} via DNS.")
        except socket.gaierror as e:
            print(f"[DEBUG] DNS resolution error for {self.openai_endpoint}: {e}")

    async def setup_audio(self):
        print("Setting up audio streams...")
        self.output_stream = sd.OutputStream(samplerate=24000, channels=1, dtype=np.int16)
        self.input_stream = sd.InputStream(samplerate=24000, channels=1, dtype=np.int16,
                                           callback=self.audio_callback)
        self.output_stream.start()
        self.input_stream.start()
        print("Audio streams started")
        
    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Input stream error: {status}")
        if not self.is_speaking:
            self.input_buffer.extend(indata.tobytes())

    async def start_conversation(self):
        try:
            async with websockets.connect(self.url) as ws:
                print("Connected to WebSocket")
                await self.setup_session(ws)
                
                # Initial greeting
                await self.send_message(ws, "Hello")
                await self.handle_response(ws)

                while True:
                    print("\nListening... (speak for ~2 seconds)")
                    self.input_buffer.clear()
                    await asyncio.sleep(2)  # Wait for 2 seconds of audio

                    if len(self.input_buffer) > 0:
                        print("Processing your input...")
                        audio_data = bytes(self.input_buffer)
                        self.input_buffer.clear()
                        
                        # Send audio
                        base64_audio = base64.b64encode(audio_data).decode('utf-8')
                        await ws.send(json.dumps({
                            "type": "input_audio_buffer.append",
                            "audio": base64_audio
                        }))
                        await ws.send(json.dumps({
                            "type": "input_audio_buffer.commit"
                        }))
                        
                        # Request response
                        await ws.send(json.dumps({
                            "type": "response.create",
                            "response": {"modalities": ["audio", "text"]}
                        }))
                        
                        await self.handle_response(ws)

        except Exception as e:
            print(f"Error in conversation: {e}")

    async def setup_session(self, ws):
        session_payload = {
            "type": "session.update",
            "session": {
                "voice": "alloy",
                "instructions": "You are a helpful AI assistant. Keep responses brief and engaging.",
                "modalities": ["audio", "text"],
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 200
                }
            }
        }
        await ws.send(json.dumps(session_payload))
        
        while True:
            response = await ws.recv()
            data = json.loads(response)
            if data.get("type") == "session.created":
                print("Session setup complete")
                break
            elif data.get("type") == "error":
                raise Exception(f"Error creating session: {data}")

    async def send_message(self, ws, text):
        message_payload = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}]
            }
        }
        await ws.send(json.dumps(message_payload))
        
        await ws.send(json.dumps({
            "type": "response.create",
            "response": {"modalities": ["audio", "text"]}
        }))

    async def handle_response(self, ws):
        self.is_speaking = True
        try:
            while True:
                response = await ws.recv()
                data = json.loads(response)
                
                if data["type"] == "response.audio.delta":
                    if "delta" in data:
                        try:
                            audio_data = data["delta"].replace(" ", "").replace("\n", "")
                            padding = len(audio_data) % 4
                            if padding:
                                audio_data += "=" * padding
                            
                            audio_bytes = base64.b64decode(audio_data)
                            audio = np.frombuffer(audio_bytes, dtype=np.int16)
                            self.output_stream.write(audio)
                            print(".", end="", flush=True)
                            
                        except Exception as e:
                            print(f"Error processing audio: {e}")
                            
                elif data["type"] == "response.done":
                    break
                    
        finally:
            self.is_speaking = False

async def main():
    system = ConversationSystem()
    await system.setup_audio()
    await system.start_conversation()

if __name__ == "__main__":
    print("Starting real-time conversation system...")
    asyncio.run(main())
