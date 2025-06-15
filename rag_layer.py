"""
RAG (Retrieval-Augmented Generation) Layer for QuantAI Restaurant
This module implements an advanced RAG system for restaurant data using QWEN AI via OpenRouter.
Features:
- Advanced NLP for query understanding
- Robust data cleaning and preprocessing
- Context-aware response generation
- Data privacy and security
- Real-time resource management
- Synthetic data integration
- User intent analysis
- Sentiment analysis
- Security verification
- Empathetic responses
- Structured formatting
- Contextual memory
- Direct and clear responses
- Engaging follow-up questions
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import aiohttp
import hashlib
import hmac
import base64
from cryptography.fernet import Fernet
import jwt
from functools import lru_cache
import re
import time
import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy
from fuzzywuzzy import fuzz
import concurrent.futures
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_layer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataLoader:
    """Loads and manages synthetic restaurant data."""
    
    def __init__(self):
        """Initialize the data loader."""
        self.data_dir = "data"
        self.data = {}
        self.load_data()
        
    def load_data(self):
        """Load all synthetic data files."""
        try:
            # Load restaurant infrastructure
            with open(os.path.join(self.data_dir, "restaurant_infrastructure.json"), 'r') as f:
                self.data['infrastructure'] = json.load(f)
                
            # Load CSV files
            csv_files = [
                'quantai_restaurant_customers.csv',
                'quantai_restaurant_reservations.csv',
                'quantai_restaurant_orders.csv',
                'quantai_restaurant_staff_schedule.csv',
                'quantai_restaurant_inventory.csv',
                'quantai_restaurant_menu.csv',
                'quantai_restaurant_financial.csv',
                'quantai_restaurant_analytics.csv'
            ]
            
            for file in csv_files:
                name = file.replace('quantai_restaurant_', '').replace('.csv', '')
                self.data[name] = pd.read_csv(os.path.join(self.data_dir, file))
                
            logger.info("Successfully loaded all synthetic restaurant data")
            
        except Exception as e:
            logger.error(f"Error loading synthetic data: {str(e)}")
            raise
            
    def get_relevant_data(self, query_type: str, entities: List[str]) -> Dict[str, Any]:
        """Get relevant data based on query type and entities."""
        try:
            relevant_data = {}
            
            # Map query types to relevant data sources
            query_data_map = {
                'reservation': ['reservations', 'staff_schedule'],
                'order': ['orders', 'menu'],
                'customer': ['customers', 'orders'],
                'menu': ['menu', 'inventory'],
                'staff': ['staff_schedule'],
                'financial': ['financial', 'analytics'],
                'inventory': ['inventory'],
                'general': ['infrastructure']
            }
            
            # Get relevant data sources
            data_sources = query_data_map.get(query_type, ['infrastructure'])
            
            # Extract relevant information
            for source in data_sources:
                if source in self.data:
                    if isinstance(self.data[source], pd.DataFrame):
                        # Filter DataFrame based on entities
                        filtered_data = self.data[source]
                        for entity in entities:
                            for col in filtered_data.columns:
                                if entity.lower() in filtered_data[col].astype(str).str.lower().values:
                                    filtered_data = filtered_data[filtered_data[col].astype(str).str.lower().str.contains(entity.lower())]
                        relevant_data[source] = filtered_data.to_dict('records')
                    else:
                        relevant_data[source] = self.data[source]
                        
            return relevant_data
            
        except Exception as e:
            logger.error(f"Error getting relevant data: {str(e)}")
            return {}

class QWENClient:
    """Client for interacting with QWEN AI via OpenRouter."""
    
    def __init__(self, api_key: str):
        """Initialize the QWEN client."""
        self.api_key = api_key
        self.session = None
        self.base_url = "https://openrouter.ai/api/v1"
        self.model = "qwen/qwen-2.5-7b-instruct:free"
        self.rate_limit = 10  # requests per minute
        self.last_request_time = 0
        
    async def initialize(self):
        """Initialize the aiohttp session."""
        if not self.session:
            self.session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://quantai-restaurant.com",
                    "X-Title": "QuantAI Restaurant Assistant"
                }
            )
            
    async def close(self):
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None
            
    async def _rate_limit(self):
        """Implement rate limiting."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < (60 / self.rate_limit):
            await asyncio.sleep((60 / self.rate_limit) - time_since_last_request)
        self.last_request_time = time.time()
        
    async def generate_response(self, prompt: str, context: Optional[str] = None, user_role: str = "customer") -> str:
        """Generate a response using QWEN AI via OpenRouter."""
        try:
            await self._rate_limit()
            
            if not self.session:
                await self.initialize()
                
            # Construct the system prompt
            system_prompt = """You are a specialized AI assistant for QuantAI Restaurant. Your responses should be:
            1. Specific to QuantAI Restaurant's services, menu, and policies
            2. Clear, concise, and professional
            3. Focused on dining experience and culinary information
            4. Privacy-conscious and secure
            5. Helpful and empathetic
            6. Based on the provided context and conversation history
            
            Do not provide information about other restaurants or general dining advice not specific to QuantAI Restaurant."""
            
            # Construct the messages
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            if context:
                messages.append({"role": "system", "content": f"Context: {context}"})
                
            messages.append({"role": "user", "content": prompt})
            
            # Prepare the request payload
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 500
            }
            
            # Make the API request
            async with self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    error_text = await response.text()
                    logger.error(f"API error: {error_text}")
                    raise Exception(f"API request failed with status {response.status}")
                    
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

class DataCleaner:
    """Handles data cleaning and preprocessing for restaurant data."""
    
    def __init__(self):
        """Initialize the data cleaner."""
        self.stop_words = set(stopwords.words('english'))
        self.nlp = spacy.load('en_core_web_sm')
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text data."""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove stop words
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in self.stop_words]
        
        return ' '.join(tokens)
        
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text."""
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
            
        return entities

class QueryProcessor:
    """Processes and analyzes user queries."""
    
    def __init__(self):
        """Initialize the query processor."""
        self.data_cleaner = DataCleaner()
        
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process and analyze a user query."""
        try:
            # Clean the query
            cleaned_query = self.data_cleaner.clean_text(query)
            
            # Extract entities
            entities = self.data_cleaner.extract_entities(query)
            
            # Determine query type
            query_type = self._determine_query_type(cleaned_query)
            
            # Extract key information
            key_info = self._extract_key_info(query, entities)
            
            # Analyze sentiment
            sentiment = self._analyze_sentiment(query)
            
            # Determine intent
            intent = self._determine_intent(query)
            
            return {
                'original_query': query,
                'cleaned_query': cleaned_query,
                'query_type': query_type,
                'entities': entities,
                'key_info': key_info,
                'sentiment': sentiment,
                'intent': intent
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {}
            
    def _determine_query_type(self, query: str) -> str:
        """Determine the type of query."""
        query = query.lower()
        
        if any(word in query for word in ['reserve', 'booking', 'table', 'reservation']):
            return 'reservation'
        elif any(word in query for word in ['order', 'menu', 'dish', 'food', 'drink']):
            return 'order'
        elif any(word in query for word in ['customer', 'guest', 'diner']):
            return 'customer'
        elif any(word in query for word in ['menu', 'dish', 'special', 'chef']):
            return 'menu'
        elif any(word in query for word in ['staff', 'server', 'chef', 'waiter']):
            return 'staff'
        elif any(word in query for word in ['price', 'cost', 'bill', 'payment']):
            return 'financial'
        elif any(word in query for word in ['inventory', 'stock', 'supply']):
            return 'inventory'
        else:
            return 'general'
            
    def _extract_key_info(self, query: str, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract key information from the query."""
        key_info = {
            'time': None,
            'date': None,
            'party_size': None,
            'special_requests': [],
            'menu_items': [],
            'prices': [],
            'locations': []
        }
        
        # Extract time and date
        time_pattern = r'\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b'
        date_pattern = r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{1,2}(?:st|nd|rd|th)?\b'
        
        time_matches = re.findall(time_pattern, query.lower())
        date_matches = re.findall(date_pattern, query.lower())
        
        if time_matches:
            key_info['time'] = time_matches[0]
        if date_matches:
            key_info['date'] = date_matches[0]
            
        # Extract party size
        party_pattern = r'\b(?:party of|table for|group of)\s*(\d+)\b'
        party_match = re.search(party_pattern, query.lower())
        if party_match:
            key_info['party_size'] = int(party_match.group(1))
            
        # Extract special requests
        special_request_keywords = ['allergy', 'dietary', 'preference', 'special']
        for word in special_request_keywords:
            if word in query.lower():
                key_info['special_requests'].append(word)
                
        # Extract menu items and prices
        for entity in entities:
            if entity['label'] in ['PRODUCT', 'DISH']:
                key_info['menu_items'].append(entity['text'])
            elif entity['label'] == 'MONEY':
                key_info['prices'].append(entity['text'])
            elif entity['label'] == 'GPE':
                key_info['locations'].append(entity['text'])
                
        return key_info
        
    def _analyze_sentiment(self, text: str) -> str:
        """Analyze the sentiment of the text."""
        analysis = TextBlob(text)
        score = analysis.sentiment.polarity
        
        if score > 0.1:
            return 'positive'
        elif score < -0.1:
            return 'negative'
        else:
            return 'neutral'
            
    def _determine_intent(self, query: str) -> str:
        """Determine the user's intent."""
        query = query.lower()
        
        if any(word in query for word in ['book', 'reserve', 'make reservation']):
            return 'make_reservation'
        elif any(word in query for word in ['cancel', 'change', 'modify reservation']):
            return 'modify_reservation'
        elif any(word in query for word in ['order', 'get', 'want', 'would like']):
            return 'place_order'
        elif any(word in query for word in ['menu', 'what do you have', 'what can i get']):
            return 'view_menu'
        elif any(word in query for word in ['price', 'cost', 'how much']):
            return 'check_price'
        elif any(word in query for word in ['special', 'recommendation', 'suggestion']):
            return 'get_recommendation'
        elif any(word in query for word in ['hours', 'open', 'close', 'when']):
            return 'check_hours'
        elif any(word in query for word in ['location', 'where', 'address']):
            return 'get_location'
        else:
            return 'general_inquiry'

class RestaurantRAG:
    """Main RAG system for restaurant data."""
    
    def __init__(self):
        """Initialize the RAG system."""
        self.data_loader = DataLoader()
        self.query_processor = QueryProcessor()
        self.qwen_client = None
        self.conversation_history = []
        self.load_environment()
        self.initialize_components()
        
    def load_environment(self):
        """Load environment variables."""
        load_dotenv()
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
            
    def initialize_components(self):
        """Initialize system components."""
        self.qwen_client = QWENClient(self.api_key)
        
    async def process_query(self, query: str, user_role: str = "customer") -> str:
        """Process a user query and generate a response."""
        try:
            # Process the query
            processed_query = self.query_processor.process_query(query)
            
            # Get relevant data
            relevant_data = self.data_loader.get_relevant_data(
                processed_query['query_type'],
                [entity['text'] for entity in processed_query['entities']]
            )
            
            # Generate context
            context = self._generate_context(processed_query, user_role, relevant_data)
            
            # Generate response
            response = await self.qwen_client.generate_response(
                query,
                context,
                user_role
            )
            
            # Format and clean response
            formatted_response = self._format_response(response, processed_query)
            cleaned_response = self._clean_response(formatted_response)
            
            # Get follow-up question
            follow_up = self._get_follow_up_question(
                processed_query['query_type'],
                processed_query
            )
            
            # Update conversation history
            self._update_conversation_history(query, cleaned_response, processed_query)
            
            # Log interaction
            self._log_interaction(query, cleaned_response, user_role)
            
            # Combine response and follow-up
            final_response = f"{cleaned_response}\n\n{follow_up}" if follow_up else cleaned_response
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return "I apologize, but I'm having trouble processing your request. Please try again."
            
    def _generate_context(self, processed_query: Dict[str, Any], user_role: str, relevant_data: Dict[str, Any]) -> str:
        """Generate context for the response."""
        context_parts = []
        
        # Add role-specific context
        if user_role == "customer":
            context_parts.append("You are speaking with a restaurant customer.")
        elif user_role == "staff":
            context_parts.append("You are speaking with a restaurant staff member.")
        elif user_role == "manager":
            context_parts.append("You are speaking with a restaurant manager.")
            
        # Add query type context
        query_type = processed_query['query_type']
        context_parts.append(f"The query is about {query_type}.")
        
        # Add relevant data context
        for source, data in relevant_data.items():
            if data:
                context_parts.append(f"Relevant {source} data: {json.dumps(data)}")
                
        # Add sentiment context
        sentiment = processed_query['sentiment']
        context_parts.append(f"The query has a {sentiment} sentiment.")
        
        # Add intent context
        intent = processed_query['intent']
        context_parts.append(f"The user's intent is to {intent}.")
        
        return "\n".join(context_parts)
        
    def _log_interaction(self, query: str, response: str, user_role: str):
        """Log the interaction."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_role': user_role,
            'query': query,
            'response': response
        }
        logger.info(f"Interaction logged: {json.dumps(log_entry)}")
        
    def _format_response(self, response: str, processed_query: Dict[str, Any]) -> str:
        """Format the response based on query type and intent."""
        formatted_response = response
        
        # Add role-specific formatting
        if processed_query['intent'] == 'make_reservation':
            formatted_response = f"Reservation Information:\n{formatted_response}"
        elif processed_query['intent'] == 'place_order':
            formatted_response = f"Order Information:\n{formatted_response}"
        elif processed_query['intent'] == 'view_menu':
            formatted_response = f"Menu Information:\n{formatted_response}"
            
        # Add sentiment-based formatting
        if processed_query['sentiment'] == 'negative':
            formatted_response = f"I understand your concern. {formatted_response}"
        elif processed_query['sentiment'] == 'positive':
            formatted_response = f"I'm glad to hear that! {formatted_response}"
            
        return formatted_response
        
    def _clean_response(self, response: str) -> str:
        """Clean and normalize the response."""
        # Remove any special characters
        response = re.sub(r'[^\w\s.,!?-]', '', response)
        
        # Fix spacing
        response = ' '.join(response.split())
        
        # Ensure proper sentence structure
        if not response.endswith(('.', '!', '?')):
            response += '.'
            
        return response
        
    def _get_follow_up_question(self, query_type: str, processed_query: Dict[str, Any]) -> str:
        """Generate a relevant follow-up question."""
        follow_ups = {
            'reservation': [
                "Would you like to know about our special events or private dining options?",
                "Can I help you with any dietary requirements for your reservation?",
                "Would you like information about our parking facilities?"
            ],
            'order': [
                "Would you like to know about our chef's specials?",
                "Can I help you with wine pairings for your order?",
                "Would you like to know about our seasonal menu items?"
            ],
            'menu': [
                "Would you like to know about our daily specials?",
                "Can I help you with any dietary restrictions?",
                "Would you like to know about our chef's recommendations?"
            ],
            'general': [
                "Is there anything else you'd like to know about our restaurant?",
                "Can I help you with anything else?",
                "Would you like to know about our upcoming events?"
            ]
        }
        
        # Get relevant follow-ups
        relevant_follow_ups = follow_ups.get(query_type, follow_ups['general'])
        
        # Select a random follow-up
        return random.choice(relevant_follow_ups)
        
    def _update_conversation_history(self, query: str, response: str, processed_query: Dict[str, Any]):
        """Update the conversation history."""
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': response,
            'query_type': processed_query['query_type'],
            'intent': processed_query['intent'],
            'sentiment': processed_query['sentiment']
        }
        
        self.conversation_history.append(history_entry)
        
        # Keep only the last 10 interactions
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
            
    async def close(self):
        """Close the RAG system."""
        if self.qwen_client:
            await self.qwen_client.close()

# Initialize the RAG system
rag_system = RestaurantRAG()

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords') 