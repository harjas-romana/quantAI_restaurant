"""
QuantAI Restaurant Voice Agent
This module provides voice-based interaction capabilities for the QuantAI Restaurant AI Assistant,
integrating speech recognition and text-to-speech synthesis for a natural conversation experience.
"""

import os
import time
import wave
import json
import numpy as np
import pyaudio
import speech_recognition as sr
from dotenv import load_dotenv
from colorama import init, Fore, Style, Back
import logging
import signal
from tqdm import tqdm
from typing import Optional, Tuple, Dict, List
import tempfile
from pathlib import Path
import io
from pydub import AudioSegment
from pydub.playback import play as pydub_play
from collections import deque
import audioop
import threading
import contextlib
import queue
import requests

# Import the base agent functionality
from agent import QuantAIAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('voice_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize colorama
init()

class GracefulExitException(Exception):
    """Exception for handling graceful exits."""
    pass

class AudioProcessor:
    """Handles sophisticated audio processing and analysis."""
    
    def __init__(self, rate: int = 16000, chunk_size: int = 1024):
        self.rate = rate
        self.chunk_size = chunk_size
        self.base_energy_threshold = 150  # Lower initial threshold for better sensitivity
        self.dynamic_threshold = self.base_energy_threshold
        self.energy_window = deque(maxlen=30)  # Sliding window for energy levels
        self.speech_started = False
        self.silence_frames = 0
        self.speech_frames = 0
        self.min_speech_frames = 5  # Minimum frames to confirm speech
        
    def calculate_energy(self, audio_chunk: bytes) -> float:
        """Calculate audio chunk energy using RMS."""
        try:
            return audioop.rms(audio_chunk, 2)  # 2 bytes per sample for int16
        except Exception as e:
            logger.error(f"Error calculating energy: {e}")
            return 0
        
    def update_energy_threshold(self, energy: float):
        """Dynamically adjust energy threshold based on audio levels."""
        self.energy_window.append(energy)
        if len(self.energy_window) >= 10:
            avg_energy = sum(self.energy_window) / len(self.energy_window)
            # More sensitive threshold adjustment
            self.dynamic_threshold = max(self.base_energy_threshold, avg_energy * 0.3)
            
    def is_speech(self, audio_chunk: bytes) -> bool:
        """Determine if audio chunk contains speech using improved detection."""
        try:
            energy = self.calculate_energy(audio_chunk)
            self.update_energy_threshold(energy)
            
            is_current_frame_speech = energy > self.dynamic_threshold
            
            # State machine for speech detection
            if is_current_frame_speech:
                self.speech_frames += 1
                self.silence_frames = 0
                if self.speech_frames >= self.min_speech_frames:
                    self.speech_started = True
            else:
                self.silence_frames += 1
                self.speech_frames = max(0, self.speech_frames - 1)
            
            # Return True if we've detected enough speech frames
            return self.speech_started and self.speech_frames > 0
            
        except Exception as e:
            logger.error(f"Error in speech detection: {e}")
            return False

class AudioState:
    """Manages audio recording state with thread-safe operations."""
    
    def __init__(self):
        self.is_recording = False
        self.has_speech = False
        self.speech_start_time = None
        self.last_speech_time = None
        self.silence_start_time = None
        self.continuous_silence_duration = 1.5  # Seconds of silence before stopping
        self._lock = threading.Lock()
        
    def start_recording(self):
        with self._lock:
            self.is_recording = True
            self.has_speech = False
            self.speech_start_time = None
            self.last_speech_time = None
            self.silence_start_time = None
            
    def stop_recording(self):
        with self._lock:
            self.is_recording = False
            
    def update_speech_state(self, is_speech: bool, current_time: float):
        """Update speech detection state in a thread-safe manner."""
        with self._lock:
            if is_speech:
                if not self.has_speech:
                    self.has_speech = True
                    self.speech_start_time = current_time
                self.last_speech_time = current_time
                self.silence_start_time = None
            elif self.has_speech:
                if self.silence_start_time is None:
                    self.silence_start_time = current_time
                    
    def should_stop(self, current_time: float) -> bool:
        """Determine if recording should stop based on silence duration."""
        with self._lock:
            if not self.has_speech:
                return False
            if self.silence_start_time is None:
                return False
            return (current_time - self.silence_start_time) >= self.continuous_silence_duration

class VoiceRecorder:
    """Handles voice recording with improved silence detection."""
    
    def __init__(self):
        self.chunk = 1024
        self.format = pyaudio.paInt16  # 16-bit audio
        self.channels = 1
        self.rate = 16000  # Standard rate for speech recognition
        self.max_duration = 30.0  # Maximum recording duration
        
        self.p = None
        self.stream = None
        self.audio_processor = AudioProcessor(self.rate, self.chunk)
        self.state = AudioState()
        self.audio_queue = queue.Queue()
        self.start_time = None  # Track recording start time
        
    def _initialize_pyaudio(self):
        """Initialize PyAudio with error handling."""
        if self.p is None:
            try:
                self.p = pyaudio.PyAudio()
            except Exception as e:
                logger.error(f"Failed to initialize PyAudio: {e}")
                raise RuntimeError("Could not initialize audio system")
                
    def _get_best_microphone(self) -> int:
        """Get the best available microphone with testing."""
        best_device = None
        best_channels = 0
        
        # First try to find a device with "mic" or "microphone" in the name
        for i in range(self.p.get_device_count()):
            try:
                device_info = self.p.get_device_info_by_index(i)
                device_name = device_info.get('name', '').lower()
                if ('mic' in device_name or 'microphone' in device_name) and \
                   device_info['maxInputChannels'] > 0:
                    return i
            except:
                continue
        
        # If no mic found, try all input devices
        for i in range(self.p.get_device_count()):
            try:
                device_info = self.p.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > best_channels:
                    # Test if we can actually open this device
                    try:
                        test_stream = self.p.open(
                            format=self.format,
                            channels=self.channels,
                            rate=self.rate,
                            input=True,
                            input_device_index=i,
                            frames_per_buffer=self.chunk,
                            start=False
                        )
                        test_stream.close()
                        best_channels = device_info['maxInputChannels']
                        best_device = i
                    except:
                        continue
            except:
                continue
                
        if best_device is not None:
            return best_device
            
        # Try default device as last resort
        try:
            default_device = self.p.get_default_input_device_info()
            return default_device['index']
        except:
            raise RuntimeError("No working microphone found")
            
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream processing."""
        if status:
            logger.warning(f"Audio stream status: {status}")
        if self.state.is_recording:
            self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)
        
    def stop_recording(self):
        """Safely stop the recording process."""
        self.state.stop_recording()
        
    @contextlib.contextmanager
    def _audio_stream(self, device_index: int):
        """Context manager for audio stream handling."""
        stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self.chunk,
            stream_callback=self._audio_callback,
            start=True
        )
        try:
            yield stream
        finally:
            stream.stop_stream()
            stream.close()
            
    def record_voice(self) -> Optional[bytes]:
        """Record voice with improved speech detection and error handling."""
        try:
            self._initialize_pyaudio()
            device_index = self._get_best_microphone()
            
            # Print selected device info
            device_info = self.p.get_device_info_by_index(device_index)
            logger.info(f"Using audio device: {device_info.get('name', 'Unknown Device')}")
            
            print(f"\n{Fore.CYAN}Listening... (Speak your question or press Ctrl+C to stop){Style.RESET_ALL}")
            
            # Reset state and clear queue
            self.audio_queue = queue.Queue()
            self.state.start_recording()
            recorded_chunks = []
            self.start_time = time.time()  # Set start time before recording
            
            # Use context manager for stream handling
            with self._audio_stream(device_index) as stream:
                # Progress bar for visual feedback
                with tqdm(total=100, desc="Recording", ncols=75, leave=False) as pbar:
                    while self.state.is_recording:
                        try:
                            # Get audio chunk with timeout
                            try:
                                audio_chunk = self.audio_queue.get(timeout=0.1)
                            except queue.Empty:
                                continue
                                
                            # Calculate duration using instance variable
                            current_time = time.time()
                            duration = current_time - self.start_time
                            
                            # Process audio for speech detection
                            is_speech = self.audio_processor.is_speech(audio_chunk)
                            self.state.update_speech_state(is_speech, current_time)
                            
                            # Store audio
                            recorded_chunks.append(audio_chunk)
                            
                            # Update progress bar
                            if self.state.has_speech:
                                progress = min(100, (duration / self.max_duration) * 100)
                                pbar.n = int(progress)
                                pbar.refresh()
                            
                            # Check stop conditions
                            if (self.state.should_stop(current_time) or 
                                duration >= self.max_duration):
                                break
                                
                        except Exception as e:
                            logger.warning(f"Error processing audio chunk: {e}")
                            continue
                            
            # Check if we got any speech
            if not self.state.has_speech:
                print(f"{Fore.YELLOW}No speech detected.{Style.RESET_ALL}")
                return None
                
            # Combine all chunks
            return b''.join(recorded_chunks)
            
        except Exception as e:
            logger.error(f"Error recording audio: {e}")
            print(f"{Fore.RED}Error recording audio. Please check your microphone.{Style.RESET_ALL}")
            return None
            
    def cleanup(self):
        """Clean up audio resources."""
        if self.p:
            try:
                self.p.terminate()
            except:
                pass
            self.p = None

class TextToSpeech:
    """Handles text-to-speech conversion using direct API calls to ElevenLabs."""
    
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('ELEVENLABS_API_KEY')
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY not found in .env file")
            
        # API endpoint for ElevenLabs
        self.api_url = "https://api.elevenlabs.io/v1/text-to-speech"
        self.voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel voice ID
        self.model_id = "eleven_multilingual_v2"
        
    def convert_to_speech(self, text: str) -> Optional[bytes]:
        """Convert text to speech using direct API calls with requests."""
        try:
            # Prepare headers with API key
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.api_key
            }
            
            # Prepare request body
            data = {
                "text": text,
                "model_id": self.model_id,
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5
                }
            }
            
            # Make the API request
            url = f"{self.api_url}/{self.voice_id}"
            response = requests.post(url, json=data, headers=headers)
            
            # Check for errors
            if response.status_code != 200:
                error_msg = f"API Error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                
                if response.status_code == 401:
                    print(f"\n{Fore.RED}Invalid or expired ElevenLabs API key. Please check your .env file "
                          f"and ensure you have a valid API key from https://elevenlabs.io/account{Style.RESET_ALL}")
                elif "quota_exceeded" in response.text:
                    print(f"\n{Fore.YELLOW}ElevenLabs API quota exceeded. The response will be displayed as text only. "
                          f"To continue using voice features, please:\n"
                          f"1. Check your quota at https://elevenlabs.io/account\n"
                          f"2. Upgrade your plan if needed\n"
                          f"3. Wait for your quota to reset{Style.RESET_ALL}")
                else:
                    print(f"\n{Fore.RED}Error converting text to speech. The response will be displayed as text.{Style.RESET_ALL}")
                
                return None
                
            # Return the audio data
            return response.content
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in text-to-speech conversion: {e}")
            print(f"\n{Fore.RED}Error converting text to speech. The response will be displayed as text.{Style.RESET_ALL}")
            return None
            
    def play_audio(self, audio_data: bytes):
        """Play audio with error handling."""
        if not audio_data:
            return
            
        try:
            # Save to temporary file first
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file.flush()
                
                # Load and play from file
                audio_segment = AudioSegment.from_mp3(temp_file.name)
                pydub_play(audio_segment)
                
            # Clean up temp file
            try:
                os.unlink(temp_file.name)
            except:
                pass
                
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
            print(f"{Fore.RED}Error playing audio response.{Style.RESET_ALL}")

class SpeechToText:
    """Handles speech-to-text conversion with improved accuracy."""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300  # Adjusted for better sensitivity
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.dynamic_energy_adjustment_damping = 0.15
        self.recognizer.dynamic_energy_ratio = 1.5
        
    def convert_to_text(self, audio_data: bytes) -> Optional[str]:
        """Convert speech to text with improved error handling."""
        if not audio_data:
            return None
            
        try:
            # Convert raw audio data to AudioData
            audio = sr.AudioData(
                audio_data,
                sample_rate=16000,
                sample_width=2  # 16-bit audio
            )
            
            # Attempt recognition
            text = self.recognizer.recognize_google(audio)
            return text if text else None
            
        except sr.UnknownValueError:
            print(f"{Fore.YELLOW}Could not understand audio{Style.RESET_ALL}")
            return None
        except sr.RequestError as e:
            logger.error(f"Speech recognition error: {e}")
            print(f"{Fore.RED}Error with speech recognition service.{Style.RESET_ALL}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in speech recognition: {e}")
            return None

class VoiceAgent:
    """Main voice interface for the QuantAI Restaurant AI Assistant."""
    
    def __init__(self):
        self.recorder = VoiceRecorder()
        self.speech_to_text = SpeechToText()
        self.text_to_speech = TextToSpeech()
        self.agent = QuantAIAgent()
        signal.signal(signal.SIGINT, self._handle_interrupt)
        
    def _handle_interrupt(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        print("\nStopping recording...")
        self.recorder.stop_recording()
        
    def process_voice_input(self) -> Optional[str]:
        """Process voice input with improved error handling."""
        try:
            # Record audio
            audio_data = self.recorder.record_voice()
            if not audio_data:
                return None
                
            print("\nProcessing your speech...")
            
            # Convert to text
            text = self.speech_to_text.convert_to_text(audio_data)
            if text:
                print(f"\nYou said: {text}")
            return text
            
        except Exception as e:
            logger.error(f"Error processing voice input: {e}")
            return None
            
    def run(self):
        """Run the voice interface with improved user experience."""
        print(f"\n{Back.BLUE}{Fore.WHITE} Welcome to QuantAI Restaurant Voice Assistant! {Style.RESET_ALL}")
        print("\nYou can:")
        print("1. Speak your questions about QuantAI Restaurant")
        print("2. Say 'quit' or 'exit' to end the conversation")
        print("3. Press Ctrl+C to stop recording at any time")
        
        while True:
            try:
                # Get voice input
                text = self.process_voice_input()
                if not text:
                    continue
                    
                # Check for exit commands
                if text.lower() in ['quit', 'exit', 'goodbye', 'bye']:
                    print(f"\n{Fore.CYAN}Thank you for using QuantAI Restaurant Voice Assistant. Goodbye!{Style.RESET_ALL}")
                    break
                    
                # Generate response
                print("\nGenerating response...")
                with tqdm(total=100, desc="Processing", ncols=75) as pbar:
                    response = self.agent.generate_response(text)
                    pbar.update(100)
                    
                print(f"\nResponse:\n{response}")
                
                # Convert to speech
                print("\nConverting response to speech...")
                audio = self.text_to_speech.convert_to_speech(response)
                
                if audio:
                    self.text_to_speech.play_audio(audio)
                else:
                    print(f"{Fore.YELLOW}Error converting text to speech. Displaying text response instead.{Style.RESET_ALL}")
                    
            except GracefulExitException:
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                print(f"{Fore.RED}An error occurred. Please try again.{Style.RESET_ALL}")
                
        # Cleanup
        self.recorder.cleanup()

def main():
    """Main entry point with improved error handling."""
    try:
        print(f"\n{Back.BLUE}{Fore.WHITE} Initializing QuantAI Restaurant Voice Assistant {Style.RESET_ALL}")
        
        # Initialize voice agent
        agent = VoiceAgent()
        print(f"{Fore.GREEN}✓ Voice assistant initialized successfully{Style.RESET_ALL}")
        
        # Run the voice interface
        agent.run()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\n{Back.RED}{Fore.WHITE} Error: {str(e)} {Style.RESET_ALL}")
        print("\nPlease ensure:")
        print("1. Your microphone is properly connected and working")
        print("2. You have set up the required API keys in .env file")
        print("3. All required dependencies are installed")

if __name__ == "__main__":
    main() 