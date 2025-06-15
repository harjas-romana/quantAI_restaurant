"""
QuantAI Restaurant AI Assistant
Advanced implementation integrating OPENROUTER's QWEN AI for context-aware, multilingual restaurant communication.
This module provides sophisticated response generation and language handling specifically tailored for
QuantAI Restaurant's dining context, ensuring professional, accurate, and empathetic communication.
"""

import os
import json
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import requests
from deep_translator import GoogleTranslator
from colorama import init, Fore, Style, Back
from tqdm import tqdm
import time
from cachetools import TTLCache, cached
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from typing import Dict, List, Any, Optional, Tuple
import re
import pickle
from pathlib import Path
import logging
from datetime import datetime
from fuzzywuzzy import fuzz, process
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quantai_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize colorama for cross-platform colored output
init()

class LanguageManager:
    """Manages language selection and translation with advanced features and persistence."""
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.language_cache_file = self.cache_dir / "language_preferences.pkl"
        self.language_preferences = self._load_language_preferences()
        
        # Initialize supported languages
        try:
            # Get languages from Google Translator
            translator = GoogleTranslator(source='auto', target='en')
            self.supported_languages = set(translator.get_supported_languages())
        except Exception as e:
            logger.warning(f"Error getting supported languages: {e}")
            # Fallback to common languages if API fails
            self.supported_languages = {
                'english', 'spanish', 'french', 'german', 'italian', 'portuguese',
                'chinese', 'japanese', 'korean', 'arabic', 'russian', 'hindi'
            }
        
        self.language_aliases = self._initialize_language_aliases()
        
    def _initialize_language_aliases(self) -> Dict[str, str]:
        """Initialize common language aliases and variations."""
        return {
            "chinese": "chinese",
            "cn": "chinese",
            "zh": "chinese",
            "español": "spanish",
            "esp": "spanish",
            "français": "french",
            "fr": "french",
            "deutsch": "german",
            "de": "german",
            "italiano": "italian",
            "it": "italian",
            "português": "portuguese",
            "pt": "portuguese",
            "русский": "russian",
            "ru": "russian",
            "हिंदी": "hindi",
            "hi": "hindi",
            "日本語": "japanese",
            "ja": "japanese",
            "한국어": "korean",
            "ko": "korean",
            "العربية": "arabic",
            "ar": "arabic"
        }

    def _load_language_preferences(self) -> Dict[str, str]:
        """Load saved language preferences with error handling."""
        try:
            if self.language_cache_file.exists():
                with open(self.language_cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Could not load language preferences: {e}")
        return {}

    def save_language_preference(self, user_id: str, language: str):
        """Save user's language preference persistently."""
        self.language_preferences[user_id] = language
        try:
            with open(self.language_cache_file, 'wb') as f:
                pickle.dump(self.language_preferences, f)
        except Exception as e:
            logger.warning(f"Could not save language preference: {e}")

    def get_language_preference(self, user_id: str) -> Optional[str]:
        """Retrieve user's saved language preference."""
        return self.language_preferences.get(user_id)

    def validate_language(self, language: str) -> Tuple[bool, str]:
        """Validate and normalize language input."""
        language = language.lower().strip()
        
        # Direct match
        if language in self.supported_languages:
            return True, language
            
        # Check aliases
        if language in self.language_aliases:
            normalized = self.language_aliases[language]
            if normalized in self.supported_languages:
                return True, normalized
            
        # Fuzzy matching for close matches
        matches = process.extractBests(
            language,
            self.supported_languages,
            score_cutoff=80,
            limit=3
        )
        
        if matches:
            return True, matches[0][0]
            
        return False, ""

    def detect_language(self, text: str) -> Tuple[str, float]:
        """Detect the language of a given text using GoogleTranslator.
        
        Args:
            text: The text to detect language for
            
        Returns:
            Tuple containing (detected_language, confidence_score)
        """
        try:
            # Use GoogleTranslator to detect language
            translator = GoogleTranslator(source='auto', target='en')
            
            # The translator doesn't have a direct language detection method,
            # but we can use the source language from a translation
            translator.source = 'auto'
            translator.translate(text[:100])  # Use just a sample of text
            detected_code = translator._source
            
            # Map language code to language name
            language_map = {
                'en': 'english',
                'es': 'spanish',
                'fr': 'french',
                'de': 'german',
                'it': 'italian',
                'pt': 'portuguese',
                'zh-cn': 'chinese',
                'zh-tw': 'chinese',
                'zh': 'chinese',
                'ja': 'japanese',
                'ko': 'korean',
                'ar': 'arabic',
                'ru': 'russian',
                'hi': 'hindi'
            }
            
            detected_language = language_map.get(detected_code, detected_code)
            
            # Validate the detected language is in our supported languages
            is_valid, normalized_language = self.validate_language(detected_language)
            if is_valid:
                return normalized_language, 0.9  # Confidence score placeholder
            else:
                return detected_code, 0.7  # Return original code with lower confidence
                
        except Exception as e:
            logger.error(f"Language detection error: {e}")
            return "english", 0.0  # Default to English with zero confidence

    def display_languages(self):
        """Display available languages in an organized, searchable format."""
        print(f"\n{Back.BLUE}{Fore.WHITE} Available Languages {Style.RESET_ALL}")
        print("\nType part of a language name to search, or press Enter to see all languages.")
        
        while True:
            search = input(f"\n{Fore.YELLOW}Search languages (or press Enter): {Style.RESET_ALL}").lower()
            
            # Filter languages based on search term
            if search:
                matching_languages = [
                    lang for lang in sorted(self.supported_languages)
                    if search in lang.lower()
                ]
            else:
                matching_languages = sorted(self.supported_languages)
            
            if not matching_languages:
                print(f"{Fore.RED}No languages found matching '{search}'{Style.RESET_ALL}")
                continue
            
            # Display matching languages in columns
            col_width = 25
            num_cols = 3
            
            print(f"\n{Fore.CYAN}Matching languages:{Style.RESET_ALL}")
            for i in range(0, len(matching_languages), num_cols):
                row = matching_languages[i:i + num_cols]
                print("".join(f"{lang:<{col_width}}" for lang in row))
            
            selection = input(f"\n{Fore.YELLOW}Select a language (or type 'search' to search again): {Style.RESET_ALL}").lower()
            
            if selection == 'search':
                continue
                
            valid, normalized_language = self.validate_language(selection)
            if valid:
                return normalized_language
            else:
                print(f"{Fore.RED}Invalid language selection. Please try again.{Style.RESET_ALL}")

class QuantAIAgent:
    """Advanced AI agent for QuantAI Restaurant with enhanced context awareness and response generation."""
    
    def __init__(self):
        """Initialize the QuantAI Agent with advanced configurations and security measures."""
        # Load environment variables securely
        load_dotenv()
        self._validate_environment()
        
        # Initialize language management
        self.language_manager = LanguageManager()
        self.user_language = None
        
        # Initialize advanced caching system
        self._initialize_caches()
        
        # Initialize NLP components with error handling
        self._initialize_nlp_components()
        
        # Load and prepare restaurant data
        self.load_restaurant_data()
        
        # Initialize API configuration
        self._initialize_api_config()
        
        # Prepare vectorized data for similarity matching
        self._prepare_vectorized_data()
        
        logger.info("QuantAI Agent initialized successfully")

    def _validate_environment(self):
        """Validate all required environment variables and API keys."""
        required_vars = ['OPENROUTER_API_KEY']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
            
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')

    def _initialize_api_config(self):
        """Initialize API configuration with enhanced security and monitoring."""
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://quantai-restaurant.com",
            "X-Title": "QuantAI Restaurant Assistant",
        }

    def _initialize_caches(self):
        """Initialize sophisticated caching system for improved performance."""
        self.response_cache = TTLCache(maxsize=100, ttl=3600)
        self.context_cache = TTLCache(maxsize=50, ttl=1800)
        self.translation_cache = TTLCache(maxsize=200, ttl=7200)
        self.similarity_cache = TTLCache(maxsize=50, ttl=3600)

    def _initialize_nlp_components(self):
        """Initialize NLP components with fallback options."""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('words', quiet=True)
            
            self.stop_words = set(stopwords.words('english'))
            self.vectorizer = TfidfVectorizer(
                stop_words=list(self.stop_words),
                ngram_range=(1, 2),
                max_features=10000
            )
        except Exception as e:
            logger.warning(f"Error initializing NLP components: {e}")
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to'])
            self.vectorizer = TfidfVectorizer(stop_words=list(self.stop_words))

    def _prepare_vectorized_data(self):
        """Prepare vectorized representations of the dataset contents for similarity matching."""
        self.dataset_vectors = {}
        for name, df in self.restaurant_data.items():
            # Convert DataFrame to text for vectorization
            text_data = df.astype(str).agg(' '.join, axis=1).tolist()
            if text_data:
                self.dataset_vectors[name] = self.vectorizer.fit_transform(text_data)

    def load_restaurant_data(self):
        """Load and prepare restaurant data from CSV files with advanced preprocessing."""
        try:
            data_dir = os.path.join(os.path.dirname(__file__), 'data')
            
            self.restaurant_data = {}
            self.data_metadata = {}
            
            for file in os.listdir(data_dir):
                if file.endswith('.csv'):
                    file_path = os.path.join(data_dir, file)
                    df = pd.read_csv(file_path)
                    
                    # Store the original data
                    dataset_name = file.replace('.csv', '')
                    self.restaurant_data[dataset_name] = df
                    
                    # Generate and store metadata
                    self.data_metadata[dataset_name] = {
                        'columns': list(df.columns),
                        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
                        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
                        'row_count': len(df),
                        'summary_stats': df.describe().to_dict() if not df.empty else {},
                        'common_values': {col: df[col].value_counts().head(5).to_dict() 
                                        for col in df.select_dtypes(include=['object']).columns}
                    }
            
            print(f"{Fore.GREEN}✓ Successfully loaded restaurant data{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error loading restaurant data: {e}{Style.RESET_ALL}")
            self.restaurant_data = {}
            self.data_metadata = {}

    @cached(cache=TTLCache(maxsize=100, ttl=3600))
    def generate_response(self, user_query: str) -> str:
        """Generate enhanced, context-aware responses using QWEN AI."""
        try:
            # Prepare rich context from restaurant data
            context = self._prepare_context(user_query)
            
            # Construct sophisticated prompt for dining context
            messages = [
                {
                    "role": "system",
                    "content": """You are the AI assistant for QuantAI Restaurant, a leading dining establishment. Your responses must be:
                    1. Professional and gastronomically accurate
                    2. Empathetic and customer-centered
                    3. Clear and authoritative
                    4. Strictly based on QuantAI Restaurant's actual services and data
                    5. Compliant with dining communication standards
                    
                    Only provide information about restaurant services and general dining guidance.
                    Always maintain customer confidentiality and privacy standards."""
                },
                {
                    "role": "system",
                    "content": f"Context for this interaction: {context}"
                },
                {
                    "role": "user",
                    "content": user_query
                }
            ]
            
            # Prepare optimized API request
            payload = {
                "model": "qwen/qwen-2.5-7b-instruct:free",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1000,
                "top_p": 0.9,
                "frequency_penalty": 0.3,
                "presence_penalty": 0.3,
                "stream": False
            }
            
            # Make API request with comprehensive error handling
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code != 200:
                    logger.error(f"API Error: {response.status_code} - {response.text}")
                    return self.translate_text(
                        "I apologize, but I'm currently unable to access the restaurant's information system. "
                        "Please try again later or contact our help desk for immediate assistance."
                    )
                
                ai_response = response.json()['choices'][0]['message']['content']
                
                # Post-process response for quality assurance
                processed_response = self._post_process_response(ai_response)
                return self.translate_text(processed_response)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"API Request Error: {str(e)}")
                return self.translate_text(
                    "I apologize, but I'm experiencing technical difficulties. "
                    "Please try again or contact our support team for assistance."
                )
                
        except Exception as e:
            logger.error(f"Unexpected Error: {str(e)}")
            return self.translate_text(
                "I apologize for the inconvenience, but I'm unable to process your request at this moment. "
                "Please try again later."
            )

    def _post_process_response(self, response: str) -> str:
        """Post-process AI responses for quality and relevance."""
        # Ensure response starts professionally
        if not any(response.lower().startswith(starter) for starter in [
            "i apologize", "thank you", "hello", "hi", "greetings",
            "welcome", "certainly", "absolutely", "i understand", "i'd be happy"
        ]):
            response = "I'd be happy to help you. " + response
        
        # Add professional closing if needed
        if not any(response.lower().endswith(closer) for closer in [
            "assistance.", "help.", "questions.", "service.", "team."
        ]):
            response += " Please let me know if you need any additional information."
        
        return response

    def translate_text(self, text: str) -> str:
        """Enhanced translation with context preservation and caching."""
        if not self.user_language or self.user_language == 'english':
            return text
        
        # Check cache first
        cache_key = f"{text[:50]}_{self.user_language}"
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
        
        try:
            # Split text into sentences for better translation
            sentences = sent_tokenize(text)
            translated_sentences = []
            
            translator = GoogleTranslator(source='english', target=self.user_language)
            
            for sentence in sentences:
                try:
                    translated = translator.translate(sentence)
                    if translated:
                        translated_sentences.append(translated)
                    else:
                        translated_sentences.append(sentence)
                except Exception as e:
                    logger.warning(f"Error translating sentence: {e}")
                    translated_sentences.append(sentence)
            
            final_translation = ' '.join(translated_sentences)
            
            # Only cache successful translations
            if final_translation and final_translation != text:
                self.translation_cache[cache_key] = final_translation
            
            return final_translation
            
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text

    def select_language(self):
        """Enhanced language selection with user-friendly interface."""
        print(f"\n{Back.BLUE}{Fore.WHITE} Welcome to Language Selection {Style.RESET_ALL}")
        print("\nPlease select your preferred language for communication.")
        print("You can:")
        print("1. Type part of the language name to search")
        print("2. Use common language codes (e.g., 'en', 'es', 'fr')")
        print("3. Type the language name in your own language")
        
        self.user_language = self.language_manager.display_languages()
        print(f"\n{Fore.GREEN}✓ Language set to: {self.user_language}{Style.RESET_ALL}")
        
        # Save preference if user ID is available
        if hasattr(self, 'user_id'):
            self.language_manager.save_language_preference(self.user_id, self.user_language)

    def change_language(self):
        """Allow users to change their language preference during conversation."""
        print(f"\n{Fore.CYAN}Changing language preference{Style.RESET_ALL}")
        self.select_language()
        return self.translate_text("Language changed successfully. How may I assist you?")

    def _extract_query_keywords(self, query: str) -> List[str]:
        """Extract important keywords from the query with error handling."""
        try:
            # Basic tokenization
            tokens = word_tokenize(query.lower())
            tokens = [token for token in tokens if token not in self.stop_words and token.isalnum()]
            
            try:
                # Try to use POS tagging if available
                pos_tags = nltk.pos_tag(tokens)
                important_pos = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
                keywords = [word for word, pos in pos_tags if pos in important_pos]
                if keywords:
                    return keywords
            except Exception:
                # Fall back to basic filtering if POS tagging fails
                pass
            
            # Return all tokens if POS tagging failed or returned no keywords
            return tokens
            
        except Exception as e:
            print(f"{Fore.YELLOW}Warning: Error in keyword extraction: {e}. Using basic filtering.{Style.RESET_ALL}")
            # Fallback to simple word splitting
            return [word.lower() for word in query.split() if word.lower() not in self.stop_words]

    def _find_relevant_data(self, query: str, df: pd.DataFrame, threshold: float = 0.2) -> pd.DataFrame:
        """Find relevant rows in a DataFrame based on query similarity."""
        keywords = self._extract_query_keywords(query)
        
        # Convert DataFrame rows to text
        df_text = df.astype(str).agg(' '.join, axis=1)
        
        # Calculate similarity scores
        query_vector = self.vectorizer.transform([' '.join(keywords)])
        similarities = cosine_similarity(query_vector, self.vectorizer.transform(df_text))
        
        # Filter relevant rows
        relevant_indices = similarities[0] > threshold
        return df[relevant_indices]

    def _prepare_context(self, query: str) -> str:
        """Prepare relevant context from restaurant data based on the user query with advanced filtering."""
        context = []
        keywords = self._extract_query_keywords(query)
        
        # Add metadata context
        context.append("Restaurant Data Overview:")
        for dataset_name, metadata in self.data_metadata.items():
            context.append(f"\n{dataset_name}:")
            context.append(f"- Available fields: {', '.join(metadata['columns'])}")
            context.append(f"- Total records: {metadata['row_count']}")
        
        # Find and add relevant data from each dataset
        for dataset_name, df in self.restaurant_data.items():
            relevant_data = self._find_relevant_data(query, df)
            
            if not relevant_data.empty:
                context.append(f"\nRelevant {dataset_name} data:")
                
                # Add summary statistics for numeric columns
                numeric_cols = relevant_data.select_dtypes(include=[np.number]).columns
                if not numeric_cols.empty:
                    stats = relevant_data[numeric_cols].describe()
                    context.append("Summary statistics:")
                    context.append(stats.to_string())
                
                # Add sample records
                context.append("\nSample records:")
                sample = relevant_data.head(3).to_dict(orient='records')
                context.append(json.dumps(sample, indent=2))
                
                # Add keyword matches
                for keyword in keywords:
                    matches = df.astype(str).apply(lambda x: x.str.contains(keyword, case=False)).any()
                    if matches.any():
                        context.append(f"\nColumns containing '{keyword}': {', '.join(matches[matches].index)}")
        
        return "\n".join(context)

    def run_cli(self):
        """Run the enhanced command-line interface with improved user interaction."""
        print(f"\n{Back.BLUE}{Fore.WHITE} Welcome to QuantAI Restaurant AI Assistant {Style.RESET_ALL}")
        print("\nI'm here to help you with information about our restaurant's services, facilities, and general dining guidance.")
        
        # Select language
        self.select_language()
        
        # Main interaction loop
        while True:
            print("\n" + "="*50)
            print(f"\n{Fore.CYAN}You can:{Style.RESET_ALL}")
            print("1. Ask any question about QuantAI Restaurant")
            print("2. Type 'language' to change your language")
            print("3. Type 'quit' to exit")
            
            user_input = input(f"\n{Fore.YELLOW}How can I help you today?: {Style.RESET_ALL}")
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print(f"\n{Fore.CYAN}Thank you for using QuantAI Restaurant AI Assistant. Take care!{Style.RESET_ALL}")
                break
            
            if user_input.lower() == 'language':
                response = self.change_language()
                print(f"\n{Fore.GREEN}Response:{Style.RESET_ALL}")
                print(response)
                continue
            
            print(f"\n{Fore.CYAN}Generating response...{Style.RESET_ALL}")
            with tqdm(total=100, desc="Processing", ncols=75) as pbar:
                try:
                    response = self.generate_response(user_input)
                    pbar.update(100)
                    print(f"\n{Fore.GREEN}Response:{Style.RESET_ALL}")
                    print(response)
                except Exception as e:
                    pbar.update(100)
                    logger.error(f"Error generating response: {e}")
                    print(f"\n{Fore.RED}I apologize, but I encountered an error while processing your request.{Style.RESET_ALL}")
                    print("Please try again or contact our support team for assistance.")

def main():
    """Enhanced main entry point with better error handling and user guidance."""
    try:
        print(f"\n{Back.BLUE}{Fore.WHITE} Initializing QuantAI Restaurant AI Assistant {Style.RESET_ALL}")
        print("\nPlease wait while I set up the necessary components...")
        
        with tqdm(total=100, desc="Loading", ncols=75) as pbar:
            agent = QuantAIAgent()
            pbar.update(100)
        
        agent.run_cli()
        
    except ValueError as e:
        print(f"\n{Back.RED}{Fore.WHITE} Configuration Error {Style.RESET_ALL}")
        print(f"\n{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        print("\nPlease ensure you have:")
        print("1. Created a .env file with your OPENROUTER_API_KEY")
        print("2. Installed all required dependencies (pip install -r requirements.txt)")
        print("3. Have the necessary data files in the /data directory")
        
    except Exception as e:
        print(f"\n{Back.RED}{Fore.WHITE} Unexpected Error {Style.RESET_ALL}")
        print(f"\n{Fore.RED}An unexpected error occurred: {str(e)}{Style.RESET_ALL}")
        logger.error(f"Unexpected error in main: {e}", exc_info=True)
        print("\nPlease try:")
        print("1. Checking your internet connection")
        print("2. Verifying all dependencies are correctly installed")
        print("3. Contacting support if the issue persists")

if __name__ == "__main__":
    main() 