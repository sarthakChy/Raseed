import logging
from typing import Optional, Dict, Any, List
from google.cloud import translate_v3 as translate
from google.api_core import exceptions as google_exceptions
import langdetect
from langdetect.lang_detect_exception import LangDetectException


class TranslationService:
    def __init__(self, project_id: Optional[str] = None):
        self.logger = logging.getLogger("financial_agent.translation")
        self.project_id = project_id

        try:
            # Initialize Google Cloud Translate v3 client
            self.translate_client = translate.TranslationServiceClient()

            if not project_id:
                raise ValueError("project_id must be provided for v3 Translate API")

            self.parent = f"projects/{project_id}/locations/global"
            self.logger.info("Translation service initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize translation service: {e}")
            raise

    def detect_language(self, text: str) -> Dict[str, Any]:
        try:
            response = self.translate_client.detect_language(
                content=text,
                mime_type="text/plain",
                parent=self.parent
            )
            language = response.languages[0]

            return {
                'language': language.language_code,
                'confidence': language.confidence,
                'is_reliable': True,
                'input': text
            }

        except google_exceptions.GoogleAPICallError as e:
            self.logger.warning(f"Google Translate detection failed, falling back to langdetect: {e}")
            try:
                detected_lang = langdetect.detect(text)
                return {
                    'language': detected_lang,
                    'confidence': 0.8,
                    'is_reliable': len(text.strip()) > 10,
                    'input': text
                }
            except LangDetectException:
                return {
                    'language': 'en',
                    'confidence': 0.1,
                    'is_reliable': False,
                    'input': text
                }

        except Exception as e:
            self.logger.error(f"Language detection failed: {e}")
            return {
                'language': 'en',
                'confidence': 0.0,
                'is_reliable': False,
                'input': text,
                'error': str(e)
            }

    def translate_to_english(self, text: str, source_language: Optional[str] = None) -> Dict[str, Any]:
        try:
            if not source_language:
                detection = self.detect_language(text)
                source_language = detection['language']

                if source_language == 'en':
                    return {
                        'success': True,
                        'translated_text': text,
                        'original_text': text,
                        'source_language': 'en',
                        'target_language': 'en',
                        'was_translated': False,
                        'detection_confidence': detection['confidence']
                    }

            response = self.translate_client.translate_text(
                parent=self.parent,
                contents=[text],
                mime_type='text/plain',
                source_language_code=source_language,
                target_language_code='en'
            )

            translated_text = response.translations[0].translated_text

            return {
                'success': True,
                'translated_text': translated_text,
                'original_text': text,
                'source_language': source_language,
                'target_language': 'en',
                'was_translated': True,
                'detection_confidence': 1.0 if source_language else None
            }

        except google_exceptions.GoogleAPICallError as e:
            self.logger.error(f"Google Translate API error: {e}")
            return {
                'success': False,
                'translated_text': text,
                'original_text': text,
                'source_language': source_language or 'unknown',
                'target_language': 'en',
                'was_translated': False,
                'error': f"Translation API error: {str(e)}"
            }

        except Exception as e:
            self.logger.error(f"Translation to English failed: {e}")
            return {
                'success': False,
                'translated_text': text,
                'original_text': text,
                'source_language': source_language or 'unknown',
                'target_language': 'en',
                'was_translated': False,
                'error': str(e)
            }

    def translate_from_english(self, text: str, target_language: str) -> Dict[str, Any]:
        try:
            if target_language == 'en':
                return {
                    'success': True,
                    'translated_text': text,
                    'original_text': text,
                    'source_language': 'en',
                    'target_language': 'en',
                    'was_translated': False
                }

            response = self.translate_client.translate_text(
                parent=self.parent,
                contents=[text],
                mime_type='text/plain',
                source_language_code='en',
                target_language_code=target_language
            )

            translated_text = response.translations[0].translated_text

            return {
                'success': True,
                'translated_text': translated_text,
                'original_text': text,
                'source_language': 'en',
                'target_language': target_language,
                'was_translated': True
            }

        except google_exceptions.GoogleAPICallError as e:
            self.logger.error(f"Google Translate API error: {e}")
            return {
                'success': False,
                'translated_text': text,
                'original_text': text,
                'source_language': 'en',
                'target_language': target_language,
                'was_translated': False,
                'error': f"Translation API error: {str(e)}"
            }

        except Exception as e:
            self.logger.error(f"Translation from English failed: {e}")
            return {
                'success': False,
                'translated_text': text,
                'original_text': text,
                'source_language': 'en',
                'target_language': target_language,
                'was_translated': False,
                'error': str(e)
            }

    def get_supported_languages(self) -> List[Dict[str, str]]:
        try:
            response = self.translate_client.get_supported_languages(parent=self.parent, display_language_code='en')
            return [{'code': lang.language_code, 'name': lang.display_name} for lang in response.languages]
        except Exception as e:
            self.logger.error(f"Failed to get supported languages: {e}")
            return []

    def batch_translate_to_english(self, texts: List[str], source_language: Optional[str] = None) -> List[Dict[str, Any]]:
        return [self.translate_to_english(text, source_language) for text in texts]

    def is_english(self, text: str, confidence_threshold: float = 0.8) -> bool:
        detection = self.detect_language(text)
        return detection['language'] == 'en' and detection['confidence'] >= confidence_threshold
