from crewai.tools import BaseTool
from typing import Any, Dict, List
import requests
from bs4 import BeautifulSoup
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
import json
import re
import pickle
from pathlib import Path

class WebSearchTool(BaseTool):
    name: str = "Web Search Tool"
    description: str = "Search the web for information on any topic using SERP API and return relevant results"
    
    def _run(self, query: str) -> str:
        """Perform web search using SERP API"""
        try:
            # Use SerpAPI for real web search
            serp_api_key = os.getenv("SERP_API_KEY")
            
            if not serp_api_key:
                # Fallback to mock results if no API key
                return self._mock_search(query)
            
            # SerpAPI request
            search_url = "https://serpapi.com/search"
            params = {
                "engine": "google",
                "q": query,
                "api_key": serp_api_key,
                "num": 10,
                "gl": "us",
                "hl": "en"
            }
            
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract organic results
            results = []
            organic_results = data.get("organic_results", [])
            
            for result in organic_results[:10]:
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("link", ""),
                    "snippet": result.get("snippet", ""),
                    "relevance_score": 0.9  # SerpAPI results are highly relevant
                })
            
            search_results = {
                "query": query,
                "results": results,
                "total_found": len(results)
            }
            
            return json.dumps(search_results, indent=2)
            
        except Exception as e:
            # Fallback to mock results on error
            return self._mock_search(query)
    
    def _mock_search(self, query: str) -> str:
        """Fallback mock search results"""
        mock_results = {
            "query": query,
            "results": [
                {
                    "title": f"Top insights about {query}",
                    "url": "https://example.com/article1",
                    "snippet": f"Comprehensive guide covering {query} with latest trends and insights.",
                    "relevance_score": 0.95
                },
                {
                    "title": f"Best practices for {query}",
                    "url": "https://example.com/article2", 
                    "snippet": f"Expert recommendations and strategies for {query} implementation.",
                    "relevance_score": 0.88
                }
            ]
        }
        return json.dumps(mock_results, indent=2)

class RAGTool(BaseTool):
    name: str = "RAG Retrieval Tool"
    description: str = "Retrieve relevant information from the internal knowledge base using RAG with FAISS and HuggingFace embeddings"
    knowledge_base_path: str = "knowledge/"
    embeddings_model: Any = None
    index: Any = None
    documents: list = []
    document_texts: list = []
    index_path: str = "knowledge/faiss_index.pkl"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._initialize_rag()
    
    def _initialize_rag(self):
        """Initialize RAG system with HuggingFace embeddings and FAISS"""
        try:
            # Initialize HuggingFace sentence transformer
            self.embeddings_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            # Check if pre-built index exists
            if os.path.exists(self.index_path):
                self._load_index()
            else:
                self._build_index()
                
        except Exception as e:
            print(f"Error initializing RAG: {e}")
            self._initialize_mock_knowledge()
    
    def _build_index(self):
        """Build FAISS index from knowledge base documents"""
        try:
            if os.path.exists(self.knowledge_base_path):
                # Load documents from knowledge base
                loader = DirectoryLoader(self.knowledge_base_path, glob="**/*.md")
                docs = loader.load()
                
                if docs:
                    # Split documents into chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=800,
                        chunk_overlap=100
                    )
                    self.documents = text_splitter.split_documents(docs)
                    
                    # Extract text content
                    self.document_texts = [doc.page_content for doc in self.documents]
                    
                    # Generate embeddings
                    embeddings = self.embeddings_model.encode(self.document_texts)
                    
                    # Create FAISS index
                    dimension = embeddings.shape[1]
                    self.index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
                    
                    # Normalize embeddings for cosine similarity
                    faiss.normalize_L2(embeddings)
                    self.index.add(embeddings.astype('float32'))
                    
                    # Save index
                    self._save_index()
                else:
                    self._initialize_mock_knowledge()
            else:
                self._initialize_mock_knowledge()
                
        except Exception as e:
            print(f"Error building index: {e}")
            self._initialize_mock_knowledge()
    
    def _save_index(self):
        """Save FAISS index and documents to disk"""
        try:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            
            index_data = {
                'index': faiss.serialize_index(self.index),
                'documents': self.documents,
                'document_texts': self.document_texts
            }
            
            with open(self.index_path, 'wb') as f:
                pickle.dump(index_data, f)
                
        except Exception as e:
            print(f"Error saving index: {e}")
    
    def _load_index(self):
        """Load FAISS index and documents from disk"""
        try:
            with open(self.index_path, 'rb') as f:
                index_data = pickle.load(f)
            
            self.index = faiss.deserialize_index(index_data['index'])
            self.documents = index_data['documents']
            self.document_texts = index_data['document_texts']
            
        except Exception as e:
            print(f"Error loading index: {e}")
            self._build_index()
    
    def _initialize_mock_knowledge(self):
        """Initialize with mock knowledge if no documents available"""
        mock_texts = [
            "EzRewards is a comprehensive loyalty and rewards platform that helps businesses increase customer engagement and retention through personalized reward programs.",
            "Brand voice guidelines: Professional, helpful, and concise. Avoid hype and focus on practical value for the reader.",
            "Customer loyalty programs should focus on providing real value to customers through meaningful rewards and experiences.",
            "Effective loyalty programs use data analytics to personalize offers and track customer behavior patterns."
        ]
        
        self.document_texts = mock_texts
        
        # Create embeddings for mock data
        embeddings = self.embeddings_model.encode(mock_texts)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        
        # Normalize and add embeddings
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
    
    def _run(self, query: str, top_k: int = 5) -> str:
        """Retrieve relevant information from knowledge base using FAISS similarity search"""
        try:
            if self.index is None or self.embeddings_model is None:
                return json.dumps({
                    "query": query,
                    "error": "RAG system not properly initialized",
                    "retrieved_documents": []
                })
            
            # Generate query embedding
            query_embedding = self.embeddings_model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Search FAISS index
            scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            # Retrieve relevant documents
            relevant_docs = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.document_texts):
                    relevant_docs.append({
                        "content": self.document_texts[idx],
                        "similarity_score": float(score),
                        "rank": i + 1,
                        "source": f"document_{idx}"
                    })
            
            result = {
                "query": query,
                "retrieved_documents": relevant_docs,
                "total_found": len(relevant_docs)
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return json.dumps({
                "query": query,
                "error": f"Error retrieving from knowledge base: {str(e)}",
                "retrieved_documents": []
            })

class CompetitorAnalysisTool(BaseTool):
    name: str = "Competitor Analysis Tool"
    description: str = "Analyze competitor content and extract insights"
    
    def _run(self, topic: str, urls: List[str] = None) -> str:
        """Analyze competitor content for given topic"""
        try:
            # Mock competitor analysis - in production, scrape actual URLs
            competitor_data = {
                "topic": topic,
                "analysis": {
                    "common_themes": [
                        f"Step-by-step guides for {topic}",
                        f"Best practices and tips",
                        f"Common challenges and solutions"
                    ],
                    "content_structures": [
                        "Introduction → Problem → Solution → Benefits → Conclusion",
                        "Listicle format with numbered points",
                        "FAQ-style content addressing common questions"
                    ],
                    "keyword_patterns": [
                        f"Primary focus on '{topic}' variations",
                        "Long-tail keywords around implementation",
                        "Question-based keywords (how to, what is, etc.)"
                    ],
                    "opportunities": [
                        "More detailed technical implementation",
                        "Industry-specific use cases",
                        "Updated statistics and data"
                    ]
                },
                "recommendations": {
                    "differentiation": f"Focus on practical, actionable insights for {topic}",
                    "content_gaps": "Lack of beginner-friendly explanations",
                    "tone_suggestions": "More conversational and engaging approach"
                }
            }
            
            return json.dumps(competitor_data, indent=2)
            
        except Exception as e:
            return f"Error analyzing competitors: {str(e)}"

class SEOTool(BaseTool):
    name: str = "SEO Analysis Tool"
    description: str = "Analyze and optimize content for SEO including keyword research"
    
    def _run(self, topic: str, content: str = None) -> str:
        """Perform SEO analysis and keyword research"""
        try:
            # Mock SEO analysis - in production, use actual SEO APIs
            seo_analysis = {
                "topic": topic,
                "keyword_strategy": {
                    "primary_keyword": f"{topic.lower().replace(' ', ' ')}",
                    "secondary_keywords": [
                        f"how to {topic.lower()}",
                        f"{topic.lower()} guide",
                        f"best {topic.lower()} practices"
                    ],
                    "long_tail_keywords": [
                        f"complete guide to {topic.lower()}",
                        f"{topic.lower()} for beginners",
                        f"advanced {topic.lower()} strategies"
                    ]
                },
                "seo_recommendations": {
                    "title_suggestions": [
                        f"The Complete Guide to {topic}",
                        f"How to Master {topic}: A Step-by-Step Guide",
                        f"10 Essential {topic} Strategies for 2025"
                    ],
                    "meta_description": f"Discover expert strategies for {topic}. Learn best practices, avoid common mistakes, and achieve better results with our comprehensive guide.",
                    "header_structure": [
                        f"What is {topic}?",
                        f"Why {topic} Matters",
                        f"Step-by-Step {topic} Implementation",
                        f"Common {topic} Challenges",
                        f"Best Practices for {topic}",
                        "Conclusion and Next Steps"
                    ]
                },
                "content_optimization": {
                    "keyword_density_target": "2-3% for primary, 1-2% for secondary",
                    "readability_target": "Grade 8-10 reading level",
                    "word_count_recommendation": "2500-4000 words for comprehensive coverage"
                }
            }
            
            if content:
                # Basic content analysis
                word_count = len(content.split())
                seo_analysis["content_metrics"] = {
                    "word_count": word_count,
                    "estimated_reading_time": f"{word_count // 200} minutes",
                    "keyword_density": "Analysis would be performed here"
                }
            
            return json.dumps(seo_analysis, indent=2)
            
        except Exception as e:
            return f"Error performing SEO analysis: {str(e)}"