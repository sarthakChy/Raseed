import logging
import asyncio
import io
import json
import base64
import tempfile
import wave
from typing import Optional, Dict, Any, List, Union, BinaryIO
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
import threading
import queue

# Google Cloud Speech and Text-to-Speech
from google.cloud import speech
from google.cloud import texttospeech
from google.api_core import exceptions as google_exceptions

# Audio processing
import pyaudio
import webrtcvad
import collections
import numpy as np

# Optional: For enhanced audio processing
try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


@dataclass
class VoiceConfig:
    """Configuration for voice services."""
    # Speech-to-Text settings
    stt_language_code: str = "en-US"
    stt_model: str = "latest_long"
    stt_use_enhanced: bool = True
    stt_enable_automatic_punctuation: bool = True
    stt_enable_word_time_offsets: bool = True
    
    # Text-to-Speech settings
    tts_language_code: str = "en-US"
    tts_voice_name: str = "en-US-Neural2-D"  # Neural voice for better quality
    tts_gender: texttospeech.SsmlVoiceGender = texttospeech.SsmlVoiceGender.NEUTRAL
    tts_audio_encoding: texttospeech.AudioEncoding = texttospeech.AudioEncoding.MP3
    tts_speaking_rate: float = 1.0
    tts_pitch: float = 0.0
    
    # Audio capture settings
    sample_rate: int = 16000
    chunk_duration_ms: int = 30  # 30ms chunks
    padding_duration_ms: int = 300
    vad_aggressiveness: int = 2  # 0-3, higher = more aggressive
    
    # Voice Activity Detection
    min_speech_duration_ms: int = 500
    max_silence_duration_ms: int = 2000


@dataclass
class SpeechResult:
    """Result from speech-to-text conversion."""
    success: bool
    transcript: str = ""
    confidence: float = 0.0
    language_detected: str = ""
    word_timestamps: List[Dict] = None
    processing_time: float = 0.0
    error: str = ""
    
    def __post_init__(self):
        if self.word_timestamps is None:
            self.word_timestamps = []


@dataclass
class SynthesisResult:
    """Result from text-to-speech synthesis."""
    success: bool
    audio_data: bytes = b""
    audio_format: str = "mp3"
    duration_seconds: float = 0.0
    processing_time: float = 0.0
    error: str = ""


class VoiceActivityDetector:
    """Enhanced Voice Activity Detection using WebRTC VAD."""
    
    def __init__(self, config: VoiceConfig):
        self.config = config
        self.vad = webrtcvad.Vad(config.vad_aggressiveness)
        self.ring_buffer = collections.deque(maxlen=int(config.padding_duration_ms / config.chunk_duration_ms))
        self.triggered = False
        self.voiced_frames = []
        self.num_padding_frames = int(config.padding_duration_ms / config.chunk_duration_ms)
        self.num_window_frames = int(300 / config.chunk_duration_ms)  # 300ms window
        
    def is_speech(self, frame: bytes) -> bool:
        """Check if frame contains speech."""
        return self.vad.is_speech(frame, self.config.sample_rate)
        
    def process_frame(self, frame: bytes) -> Optional[bytes]:
        """
        Process audio frame and return complete speech segment when detected.
        
        Returns:
            Complete speech audio data when speech segment ends, None otherwise
        """
        is_speech = self.is_speech(frame)
        
        if not self.triggered:
            self.ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in self.ring_buffer if speech])
            
            if num_voiced > 0.9 * self.ring_buffer.maxlen:
                self.triggered = True
                self.voiced_frames.extend([f for f, _ in self.ring_buffer])
                self.ring_buffer.clear()
                
        else:
            self.voiced_frames.append(frame)
            self.ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in self.ring_buffer if not speech])
            
            if num_unvoiced > 0.9 * self.ring_buffer.maxlen:
                self.triggered = False
                # Return complete speech segment
                speech_audio = b''.join(self.voiced_frames)
                self.voiced_frames.clear()
                self.ring_buffer.clear()
                return speech_audio
                
        return None


class AudioCapture:
    """Real-time audio capture with voice activity detection."""
    
    def __init__(self, config: VoiceConfig):
        self.config = config
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.vad = VoiceActivityDetector(config)
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.logger = logging.getLogger("voice_service.audio_capture")
        
        # Calculate frame size
        self.frame_duration = config.chunk_duration_ms / 1000.0
        self.frame_size = int(config.sample_rate * self.frame_duration)
        
    def start_capture(self):
        """Start audio capture."""
        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.config.sample_rate,
                input=True,
                frames_per_buffer=self.frame_size,
                stream_callback=self._audio_callback
            )
            self.is_recording = True
            self.stream.start_stream()
            self.logger.info("Audio capture started")
            
        except Exception as e:
            self.logger.error(f"Failed to start audio capture: {e}")
            raise
            
    def stop_capture(self):
        """Stop audio capture."""
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.logger.info("Audio capture stopped")
        
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream."""
        if self.is_recording:
            # Process frame through VAD
            speech_segment = self.vad.process_frame(in_data)
            if speech_segment:
                self.audio_queue.put(speech_segment)
                
        return (None, pyaudio.paContinue)
        
    def get_speech_segment(self, timeout: float = 1.0) -> Optional[bytes]:
        """Get the next detected speech segment."""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def __del__(self):
        """Cleanup audio resources."""
        if hasattr(self, 'stream') and self.stream:
            self.stream.close()
        if hasattr(self, 'audio'):
            self.audio.terminate()


class VoiceService:
    """
    Comprehensive voice service for speech-to-text and text-to-speech.
    Integrates with the multilingual financial agent system.
    """
    
    def __init__(self, project_id: Optional[str] = None, config: Optional[VoiceConfig] = None):
        """
        Initialize voice service.
        
        Args:
            project_id: Google Cloud project ID
            config: Voice configuration settings
        """
        self.project_id = project_id
        self.config = config or VoiceConfig()
        self.logger = logging.getLogger("financial_agent.voice")
        
        try:
            # Initialize Google Cloud clients
            if project_id:
                self.speech_client = speech.SpeechClient()
                self.tts_client = texttospeech.TextToSpeechClient()
            else:
                self.speech_client = speech.SpeechClient()
                self.tts_client = texttospeech.TextToSpeechClient()
                
            # Initialize audio capture
            self.audio_capture = AudioCapture(self.config)
            
            self.logger.info("Voice service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize voice service: {e}")
            raise
    
    def set_language(self, language_code: str, voice_name: Optional[str] = None):
        """
        Set language for both STT and TTS.
        
        Args:
            language_code: Language code (e.g., 'en-US', 'hi-IN', 'es-ES')
            voice_name: Specific voice name (optional)
        """
        self.config.stt_language_code = language_code
        self.config.tts_language_code = language_code
        
        if voice_name:
            self.config.tts_voice_name = voice_name
        else:
            # Auto-select appropriate voice for language
            self.config.tts_voice_name = self._get_default_voice_for_language(language_code)
    
    def _get_default_voice_for_language(self, language_code: str) -> str:
        """Get default voice for a language."""
        voice_mapping = {
            "en-US": "en-US-Neural2-D",
            "en-GB": "en-GB-Neural2-A", 
            "hi-IN": "hi-IN-Neural2-A",
            "es-ES": "es-ES-Neural2-A",
            "fr-FR": "fr-FR-Neural2-A",
            "de-DE": "de-DE-Neural2-A",
            "ja-JP": "ja-JP-Neural2-B",
            "ko-KR": "ko-KR-Neural2-A",
            "pt-BR": "pt-BR-Neural2-A",
            "zh-CN": "cmn-CN-Standard-A",
            "ta-IN": "ta-IN-Standard-A",
            "te-IN": "te-IN-Standard-A",
            "bn-IN": "bn-IN-Standard-A"
        }
        return voice_mapping.get(language_code, "en-US-Neural2-D")
    
    async def speech_to_text(
        self, 
        audio_data: Union[bytes, str, Path], 
        language_hint: Optional[str] = None
    ) -> SpeechResult:
        """
        Convert speech to text.
        
        Args:
            audio_data: Audio data as bytes, file path, or Path object
            language_hint: Language hint for better recognition
            
        Returns:
            Speech recognition result
        """
        start_time = datetime.now()
        
        try:
            # Handle different input types
            if isinstance(audio_data, (str, Path)):
                with open(audio_data, 'rb') as audio_file:
                    audio_content = audio_file.read()
            else:
                audio_content = audio_data
            
            # Configure recognition
            audio = speech.RecognitionAudio(content=audio_content)
            
            # Use language hint if provided, otherwise use config
            language_code = language_hint or self.config.stt_language_code
            
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.config.sample_rate,
                language_code=language_code,
                model=self.config.stt_model,
                use_enhanced=self.config.stt_use_enhanced,
                enable_automatic_punctuation=self.config.stt_enable_automatic_punctuation,
                enable_word_time_offsets=self.config.stt_enable_word_time_offsets,
                # Alternative language codes for better recognition
                alternative_language_codes=self._get_alternative_languages(language_code)
            )
            
            # Perform recognition
            response = self.speech_client.recognize(config=config, audio=audio)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            if response.results:
                # Get best result
                result = response.results[0]
                alternative = result.alternatives[0]
                
                # Extract word timestamps
                word_timestamps = []
                if hasattr(alternative, 'words'):
                    for word in alternative.words:
                        word_timestamps.append({
                            'word': word.word,
                            'start_time': word.start_time.total_seconds(),
                            'end_time': word.end_time.total_seconds()
                        })
                
                return SpeechResult(
                    success=True,
                    transcript=alternative.transcript.strip(),
                    confidence=alternative.confidence,
                    language_detected=language_code,
                    word_timestamps=word_timestamps,
                    processing_time=processing_time
                )
            else:
                return SpeechResult(
                    success=False,
                    error="No speech detected in audio",
                    processing_time=processing_time
                )
                
        except google_exceptions.GoogleAPICallError as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Google Speech API error: {e}")
            return SpeechResult(
                success=False,
                error=f"Speech recognition API error: {str(e)}",
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Speech to text failed: {e}")
            return SpeechResult(
                success=False,
                error=str(e),
                processing_time=processing_time
            )
    
    async def text_to_speech(
        self, 
        text: str, 
        output_format: str = "mp3",
        voice_settings: Optional[Dict[str, Any]] = None
    ) -> SynthesisResult:
        """
        Convert text to speech.
        
        Args:
            text: Text to synthesize
            output_format: Audio format ('mp3', 'wav', 'ogg')
            voice_settings: Custom voice settings
            
        Returns:
            Speech synthesis result
        """
        start_time = datetime.now()
        
        try:
            # Prepare synthesis input
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            # Configure voice
            voice_config = voice_settings or {}
            voice = texttospeech.VoiceSelectionParams(
                language_code=voice_config.get('language_code', self.config.tts_language_code),
                name=voice_config.get('voice_name', self.config.tts_voice_name),
                ssml_gender=voice_config.get('gender', self.config.tts_gender)
            )
            
            # Configure audio
            audio_encoding = {
                'mp3': texttospeech.AudioEncoding.MP3,
                'wav': texttospeech.AudioEncoding.LINEAR16,
                'ogg': texttospeech.AudioEncoding.OGG_OPUS
            }.get(output_format.lower(), texttospeech.AudioEncoding.MP3)
            
            audio_config = texttospeech.AudioConfig(
                audio_encoding=audio_encoding,
                speaking_rate=voice_config.get('speaking_rate', self.config.tts_speaking_rate),
                pitch=voice_config.get('pitch', self.config.tts_pitch)
            )
            
            # Perform synthesis
            response = self.tts_client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Estimate duration (rough calculation)
            estimated_duration = len(text.split()) * 0.5  # ~0.5 seconds per word
            
            return SynthesisResult(
                success=True,
                audio_data=response.audio_content,
                audio_format=output_format,
                duration_seconds=estimated_duration,
                processing_time=processing_time
            )
            
        except google_exceptions.GoogleAPICallError as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Google TTS API error: {e}")
            return SynthesisResult(
                success=False,
                error=f"Text-to-speech API error: {str(e)}",
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Text to speech failed: {e}")
            return SynthesisResult(
                success=False,
                error=str(e),
                processing_time=processing_time
            )
    
    async def start_listening(self) -> None:
        """Start real-time voice capture."""
        try:
            self.audio_capture.start_capture()
            self.logger.info("Started listening for voice input")
        except Exception as e:
            self.logger.error(f"Failed to start listening: {e}")
            raise
    
    async def stop_listening(self) -> None:
        """Stop real-time voice capture."""
        try:
            self.audio_capture.stop_capture()
            self.logger.info("Stopped listening for voice input")
        except Exception as e:
            self.logger.error(f"Failed to stop listening: {e}")
    
    async def get_next_speech(self, timeout: float = 5.0) -> Optional[SpeechResult]:
        """
        Get the next detected speech segment and convert to text.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Speech recognition result or None if timeout
        """
        try:
            # Get speech segment from audio capture
            speech_data = self.audio_capture.get_speech_segment(timeout)
            if speech_data:
                # Convert to text
                return await self.speech_to_text(speech_data)
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get next speech: {e}")
            return SpeechResult(success=False, error=str(e))
    
    def save_audio(self, audio_data: bytes, file_path: Union[str, Path], format: str = "wav"):
        """
        Save audio data to file.
        
        Args:
            audio_data: Audio data to save
            file_path: Output file path
            format: Audio format
        """
        try:
            if format.lower() == "wav":
                with wave.open(str(file_path), 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(self.config.sample_rate)
                    wav_file.writeframes(audio_data)
            else:
                # For other formats, save as binary
                with open(file_path, 'wb') as f:
                    f.write(audio_data)
                    
            self.logger.info(f"Audio saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save audio: {e}")
            raise
    
    def get_available_voices(self, language_code: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get available voices for text-to-speech.
        
        Args:
            language_code: Filter by language code
            
        Returns:
            List of available voices
        """
        try:
            voices = self.tts_client.list_voices()
            
            available_voices = []
            for voice in voices.voices:
                for lang_code in voice.language_codes:
                    if not language_code or lang_code.startswith(language_code.split('-')[0]):
                        available_voices.append({
                            'name': voice.name,
                            'language_code': lang_code,
                            'gender': voice.ssml_gender.name,
                            'natural_sample_rate': voice.natural_sample_rate_hertz
                        })
            
            return available_voices
            
        except Exception as e:
            self.logger.error(f"Failed to get available voices: {e}")
            return []
    
    def _get_alternative_languages(self, primary_language: str) -> List[str]:
        """Get alternative language codes for better recognition."""
        alternatives = {
            'en-US': ['en-GB', 'en-AU'],
            'hi-IN': ['en-IN'],
            'es-ES': ['es-MX', 'es-AR'],
            'fr-FR': ['fr-CA'],
            'pt-BR': ['pt-PT'],
            'zh-CN': ['zh-TW']
        }
        return alternatives.get(primary_language, [])


# Utility functions for easy integration
def create_voice_service(project_id: Optional[str] = None) -> VoiceService:
    """Factory function to create voice service instance."""
    return VoiceService(project_id)


async def process_audio_file(
    file_path: Union[str, Path], 
    project_id: Optional[str] = None,
    language_hint: Optional[str] = None
) -> SpeechResult:
    """
    Convenience function to process an audio file.
    
    Args:
        file_path: Path to audio file
        project_id: Google Cloud project ID
        language_hint: Language hint for recognition
        
    Returns:
        Speech recognition result
    """
    service = VoiceService(project_id)
    return await service.speech_to_text(file_path, language_hint)


async def synthesize_to_file(
    text: str,
    output_path: Union[str, Path],
    project_id: Optional[str] = None,
    language_code: str = "en-US",
    voice_name: Optional[str] = None
) -> bool:
    """
    Convenience function to synthesize text to audio file.
    
    Args:
        text: Text to synthesize
        output_path: Output file path
        project_id: Google Cloud project ID
        language_code: Language for synthesis
        voice_name: Specific voice name
        
    Returns:
        Success status
    """
    service = VoiceService(project_id)
    service.set_language(language_code, voice_name)
    
    result = await service.text_to_speech(text)
    if result.success:
        with open(output_path, 'wb') as f:
            f.write(result.audio_data)
        return True
    return False