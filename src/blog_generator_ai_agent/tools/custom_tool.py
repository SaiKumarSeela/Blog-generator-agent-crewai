from crewai.tools import BaseTool
from typing import Any, Dict, List, Optional
import requests
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import json
import pickle
from datetime import datetime
import hashlib
from firecrawl import Firecrawl
import logging
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from pydantic import Field, BaseModel
from serpapi import GoogleSearch
from src.blog_generator_ai_agent.utils.constants import EMBEDDING_MODEL_RAG, CHUNK_SIZE, CHUNK_OVERLAP
import re
import json
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedRAGTool(BaseTool):
    name: str = "Enhanced RAG Tool"
    description: str = "RAG system supporting PDFs, Markdown, URLs with FAISS vector storage"
    
    # Declare Pydantic fields
    knowledge_base_path: str = Field(default="knowledge/")
    index_path: str = Field(default="")
    embeddings_model: Optional[Any] = Field(default=None, exclude=True)
    index: Optional[Any] = Field(default=None, exclude=True)
    documents: List[str] = Field(default_factory=list)
    document_metadata: List[Dict] = Field(default_factory=list)
    firecrawl_app: Optional[Any] = Field(default=None, exclude=True)
    
    def __init__(self, knowledge_base_path: str = "knowledge/", **kwargs):
        super().__init__(**kwargs)
        # Set the knowledge base path
        object.__setattr__(self, 'knowledge_base_path', knowledge_base_path)
        object.__setattr__(self, 'index_path', f"{knowledge_base_path}/faiss_index.pkl")
        object.__setattr__(self, 'embeddings_model', None)
        object.__setattr__(self, 'index', None)
        object.__setattr__(self, 'documents', [])
        object.__setattr__(self, 'document_metadata', [])
        object.__setattr__(self, 'firecrawl_app', None)
        
        self._initialize()
    
    def _initialize(self):
        """Initialize embeddings model and services"""
        try:
            embeddings_model = SentenceTransformer(EMBEDDING_MODEL_RAG)
            object.__setattr__(self, 'embeddings_model', embeddings_model)
            logger.info("Embeddings model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embeddings model: {e}")
            raise
        
        # Initialize FireCrawl
        firecrawl_key = os.getenv("FIRECRAWL_API_KEY")
        if firecrawl_key:
            try:
                firecrawl_app = Firecrawl(api_key=firecrawl_key)
                object.__setattr__(self, 'firecrawl_app', firecrawl_app)
                logger.info("FireCrawl initialized")
            except Exception as e:
                logger.warning(f"FireCrawl initialization failed: {e}")
        else:
            logger.warning("FIRECRAWL_API_KEY not found - URL scraping will use basic method")
        
        os.makedirs(self.knowledge_base_path, exist_ok=True)
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        """Load existing index or create new one"""
        if os.path.exists(self.index_path):
            try:
                self._load_index()
                logger.info(f"Loaded existing index with {len(self.documents)} documents")
            except Exception as e:
                logger.error(f"Failed to load index: {e}")
                self._create_empty_index()
        else:
            self._create_empty_index()
    
    def _create_empty_index(self):
        """Create empty FAISS index"""
        index = faiss.IndexFlatIP(384)  # MiniLM dimension
        object.__setattr__(self, 'index', index)
        object.__setattr__(self, 'documents', [])
        object.__setattr__(self, 'document_metadata', [])
        logger.info("Created empty FAISS index")
    
    def add_pdf_documents(self, pdf_paths: List[str]) -> Dict[str, Any]:
        """Add PDF documents to knowledge base"""
        results = {"added": 0, "failed": 0, "errors": []}
        
        for pdf_path in pdf_paths:
            try:
                if not os.path.exists(pdf_path):
                    results["errors"].append(f"PDF file not found: {pdf_path}")
                    results["failed"] += 1
                    continue
                
                loader = PyPDFLoader(pdf_path)
                pages = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP
                )
                
                for page in pages:
                    chunks = text_splitter.split_text(page.page_content)
                    
                    for i, chunk in enumerate(chunks):
                        metadata = {
                            "source": pdf_path,
                            "source_type": "pdf",
                            "page": page.metadata.get("page", 0),
                            "chunk_id": i,
                            "added_at": datetime.now().isoformat(),
                            "doc_id": self._generate_doc_id(f"{pdf_path}_page_{page.metadata.get('page', 0)}_chunk_{i}")
                        }
                        
                        # Append to documents and metadata
                        current_docs = list(self.documents)
                        current_meta = list(self.document_metadata)
                        current_docs.append(chunk)
                        current_meta.append(metadata)
                        object.__setattr__(self, 'documents', current_docs)
                        object.__setattr__(self, 'document_metadata', current_meta)
                
                results["added"] += 1
                logger.info(f"Added PDF: {pdf_path}")
                
            except Exception as e:
                results["errors"].append(f"Failed to process PDF {pdf_path}: {str(e)}")
                results["failed"] += 1
                logger.error(f"PDF processing error for {pdf_path}: {e}")
        
        if results["added"] > 0:
            self._rebuild_index()
        
        return results
    
    def add_markdown_documents(self, markdown_paths: List[str]) -> Dict[str, Any]:
        """Add Markdown documents to knowledge base"""
        results = {"added": 0, "failed": 0, "errors": []}
        
        for md_path in markdown_paths:
            try:
                if not os.path.exists(md_path):
                    results["errors"].append(f"Markdown file not found: {md_path}")
                    results["failed"] += 1
                    continue
                
                with open(md_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP
                )
                chunks = text_splitter.split_text(content)
                
                for i, chunk in enumerate(chunks):
                    metadata = {
                        "source": md_path,
                        "source_type": "markdown",
                        "chunk_id": i,
                        "added_at": datetime.now().isoformat(),
                        "doc_id": self._generate_doc_id(f"{md_path}_chunk_{i}")
                    }
                    
                    # Append to documents and metadata
                    current_docs = list(self.documents)
                    current_meta = list(self.document_metadata)
                    current_docs.append(chunk)
                    current_meta.append(metadata)
                    object.__setattr__(self, 'documents', current_docs)
                    object.__setattr__(self, 'document_metadata', current_meta)
                
                results["added"] += 1
                logger.info(f"Added Markdown: {md_path}")
                
            except Exception as e:
                results["errors"].append(f"Failed to process Markdown {md_path}: {str(e)}")
                results["failed"] += 1
                logger.error(f"Markdown processing error for {md_path}: {e}")
        
        if results["added"] > 0:
            self._rebuild_index()
        
        return results
    
    def add_url_documents(self, urls: List[str]) -> Dict[str, Any]:
        """Add web pages via URL scraping"""
        results = {"added": 0, "failed": 0, "errors": []}
        
        for url in urls:
            try:
                content = self._scrape_url(url)
                
                if not content:
                    results["errors"].append(f"No content extracted from URL: {url}")
                    results["failed"] += 1
                    continue
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP
                )
                chunks = text_splitter.split_text(content)
                
                for i, chunk in enumerate(chunks):
                    metadata = {
                        "source": url,
                        "source_type": "url",
                        "chunk_id": i,
                        "added_at": datetime.now().isoformat(),
                        "doc_id": self._generate_doc_id(f"{url}_chunk_{i}")
                    }
                    
                    # Append to documents and metadata
                    current_docs = list(self.documents)
                    current_meta = list(self.document_metadata)
                    current_docs.append(chunk)
                    current_meta.append(metadata)
                    object.__setattr__(self, 'documents', current_docs)
                    object.__setattr__(self, 'document_metadata', current_meta)
                
                results["added"] += 1
                logger.info(f"Added URL: {url}")
                
            except Exception as e:
                results["errors"].append(f"Failed to process URL {url}: {str(e)}")
                results["failed"] += 1
                logger.error(f"URL processing error for {url}: {e}")
        
        if results["added"] > 0:
            self._rebuild_index()
        
        return results
    
    def _scrape_url(self, url: str) -> str:
        """Scrape URL content using FireCrawl or basic scraping"""
        content = None
        try:
            
            if self.firecrawl_app:
                
                result =self.firecrawl_app.scrape(
                    url,
                    formats=["markdown"],
                    only_main_content=False,
                    timeout=120000      
                )
            
                return result.markdown
                
            # Fallback to basic scraping
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            # Try to find main content
            content_selectors = ['article', 'main', '.content', '#content']
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    return ' '.join([elem.get_text(strip=True) for elem in elements])
            
            return soup.get_text(strip=True)
            
        except Exception as e:
            logger.error(f"URL scraping failed for {url}: {e}")
            raise
    
    def _generate_doc_id(self, source: str) -> str:
        """Generate unique document ID"""
        return hashlib.md5(source.encode()).hexdigest()[:16]
    
    def _rebuild_index(self):
        """Rebuild FAISS index with current documents"""
        try:
            if not self.documents:
                logger.warning("No documents to index")
                return
            
            embeddings = self.embeddings_model.encode(self.documents)
            
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            
            faiss.normalize_L2(embeddings)
            index.add(embeddings.astype('float32'))
            
            object.__setattr__(self, 'index', index)
            self._save_index()
            logger.info(f"Index rebuilt with {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f"Index rebuild failed: {e}")
            raise
    
    def _save_index(self):
        """Save FAISS index and metadata"""
        try:
            index_data = {
                'index': faiss.serialize_index(self.index),
                'documents': self.documents,
                'document_metadata': self.document_metadata,
                'saved_at': datetime.now().isoformat()
            }
            
            with open(self.index_path, 'wb') as f:
                pickle.dump(index_data, f)
                
        except Exception as e:
            logger.error(f"Index save failed: {e}")
            raise
    
    def _load_index(self):
        """Load FAISS index and metadata"""
        try:
            with open(self.index_path, 'rb') as f:
                index_data = pickle.load(f)
            
            index = faiss.deserialize_index(index_data['index'])
            documents = index_data['documents']
            document_metadata = index_data['document_metadata']
            
            object.__setattr__(self, 'index', index)
            object.__setattr__(self, 'documents', documents)
            object.__setattr__(self, 'document_metadata', document_metadata)
            
        except Exception as e:
            logger.error(f"Index load failed: {e}")
            raise
    
    def retrieve(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Retrieve relevant documents"""
        try:
            if not self.index or not self.documents:
                return {
                    "query": query,
                    "retrieved_documents": [],
                    "total_found": 0,
                    "error": "No documents in knowledge base"
                }
            
            query_embedding = self.embeddings_model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            retrieved_docs = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.documents):
                    retrieved_docs.append({
                        "content": self.documents[idx],
                        "metadata": self.document_metadata[idx],
                        "similarity_score": float(score),
                        "rank": i + 1,
                        "source": self.document_metadata[idx]["source"],
                        "snippet": self.documents[idx][:200] + "...",
                        "tags": self._generate_tags(self.documents[idx], query),
                        "confidence": float(score)
                    })
            
            return {
                "query": query,
                "retrieved_documents": retrieved_docs,
                "total_found": len(retrieved_docs)
            }
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return {
                "query": query,
                "error": str(e),
                "retrieved_documents": [],
                "total_found": 0
            }
    
    def _generate_tags(self, content: str, query: str) -> List[str]:
        """Generate simple tags based on content"""
        tags = []
        content_lower = content.lower()
        query_words = query.lower().split()
        
        for word in query_words:
            if word in content_lower:
                tags.append(f"query:{word}")
        
        if "tutorial" in content_lower or "how to" in content_lower:
            tags.append("tutorial")
        if "example" in content_lower:
            tags.append("examples")
        
        return tags[:5]
    
    # Define explicit args schema for tool invocation
    class EnhancedRAGToolInput(BaseModel):
        query: str
        top_k: int = 5

    args_schema = EnhancedRAGToolInput

    def _run(self, query: str, top_k: int = 5) -> str:
        """Main RAG retrieval function"""
        result = self.retrieve(query, top_k)
        return json.dumps(result, indent=2)

class ResearchModeTool(BaseTool):
    name: str = "Research Mode Tool"
    description: str = "Research tool with SERP analysis, RAG retrieval, and reference article processing"
    
    # Declare Pydantic fields
    rag_tool: Optional[Any] = Field(default=None, exclude=True)
    serp_api_key: Optional[str] = Field(default=None)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'rag_tool', EnhancedRAGTool())
        object.__setattr__(self, 'serp_api_key', os.getenv("SERP_API_KEY"))
    
    def serp_analysis(self, topic: str, num_results: int = 10) -> Dict[str, Any]:
        """Perform SERP analysis"""
        try:
            if not self.serp_api_key:
                return {
                    "error": "SERP_API_KEY not configured",
                    "topic": topic,
                    "findings": []
                }
            
            params = {
                "q": topic,
                "hl": "en",
                "gl": "us",
                "num": num_results,
                "google_domain": "google.com",
                "api_key": self.serp_api_key
            }

            search = GoogleSearch(params)
            results = search.get_dict()
            print(results)
            
            findings = []
            for i, result in enumerate(results.get("organic_results", [])):
                finding = {
                    "source": result.get("link", ""),
                    "title": result.get("title", ""),
                    "snippet": result.get("snippet", ""),
                    "tags": self._analyze_serp_result(result, topic),
                    "confidence": 1.0 - (i * 0.05),  # Decrease confidence with rank
                    "rank": i + 1
                }
                findings.append(finding)
            
            return {
                "research_mode": "serp_analysis",
                "topic": topic,
                "findings": findings,
                "total_found": len(findings)
            }
            
        except Exception as e:
            logger.error(f"SERP analysis failed: {e}")
            return {
                "error": str(e),
                "topic": topic,
                "findings": []
            }
    
    def internal_knowledge_retrieval(self, topic: str, top_k: int = 5) -> Dict[str, Any]:
        """Retrieve from internal knowledge base"""
        try:
            # Check if knowledge base has documents
            if not self.rag_tool.documents or len(self.rag_tool.documents) == 0:
                return {
                    "research_mode": "internal_knowledge_base",
                    "error": "No documents in knowledge base. Please upload files first or use SERP mode for web research.",
                    "topic": topic,
                    "findings": [],
                    "suggestions": [
                        "Upload PDF, Markdown, or text files to build your knowledge base",
                        "Use SERP mode for web-based research instead",
                        "Check if files were properly uploaded and processed"
                    ]
                }
            
            rag_result = self.rag_tool.retrieve(topic, top_k)
            
            if "error" in rag_result:
                return {
                    "research_mode": "internal_knowledge_base",
                    "error": rag_result["error"],
                    "topic": topic,
                    "findings": []
                }
            
            findings = []
            for doc in rag_result.get("retrieved_documents", []):
                finding = {
                    "source": doc["source"],
                    "snippet": doc["snippet"],
                    "tags": doc["tags"],
                    "confidence": doc["confidence"],
                    "rank": doc["rank"]
                }
                findings.append(finding)
            
            return {
                "research_mode": "internal_knowledge_base",
                "topic": topic,
                "findings": findings,
                "total_found": len(findings)
            }
            
        except Exception as e:
            logger.error(f"Internal knowledge retrieval failed: {e}")
            return {
                "research_mode": "internal_knowledge_base",
                "error": str(e),
                "topic": topic,
                "findings": []
            }
    
    def process_reference_articles(self, content_list: List[Dict[str, str]], topic: str) -> Dict[str, Any]:
        """Process reference articles"""
        try:
            findings = []
            
            for i, content_item in enumerate(content_list):
                content_type = content_item.get("type", "text")
                content_data = content_item.get("content", "")
                source = content_item.get("source", f"reference_{i+1}")
                
                # Add to knowledge base based on type
                if content_type == "url":
                    result = self.rag_tool.add_url_documents([content_data])
                elif content_type == "pdf":
                    result = self.rag_tool.add_pdf_documents([content_data])
                elif content_type == "markdown":
                    temp_path = f"/tmp/ref_{i+1}.md"
                    with open(temp_path, 'w', encoding='utf-8') as f:
                        f.write(content_data)
                    result = self.rag_tool.add_markdown_documents([temp_path])
                
                if result.get("errors"):
                    logger.warning(f"Processing errors for {source}: {result['errors']}")
                
                finding = {
                    "source": source,
                    "snippet": content_data[:200] + "..." if len(content_data) > 200 else content_data,
                    "tags": ["reference", "uploaded"],
                    "confidence": 0.95,
                    "rank": i + 1
                }
                findings.append(finding)
            
            return {
                "research_mode": "reference_articles",
                "topic": topic,
                "findings": findings,
                "total_found": len(findings)
            }
            
        except Exception as e:
            logger.error(f"Reference article processing failed: {e}")
            return {
                "research_mode": "reference_articles",
                "error": str(e),
                "topic": topic,
                "findings": []
            }
    
    def _analyze_serp_result(self, result: Dict, topic: str) -> List[str]:
        """Generate tags for SERP result"""
        tags = []
        title = result.get("title", "").lower()
        snippet = result.get("snippet", "").lower()
        combined = f"{title} {snippet}"
        
        if "guide" in combined or "tutorial" in combined:
            tags.append("guide")
        if "best practices" in combined or "tips" in combined:
            tags.append("best-practices")
        if "2024" in combined or "2025" in combined:
            tags.append("recent")
        
        topic_words = topic.lower().split()
        for word in topic_words:
            if word in combined:
                tags.append(f"topic:{word}")
        
        return tags
    
    # Define explicit args schema for tool invocation
    class ResearchModeToolInput(BaseModel):
        mode: str
        topic: str
        num_results: int | None = 10
        top_k: int | None = 10
        content_list: List[Dict[str, str]] | None = None

    args_schema = ResearchModeToolInput

    def _run(self, mode: str, topic: str, **kwargs) -> str:
        """Execute research mode"""
        
        try:
            # Handle JSON input from agents
            if isinstance(mode, str) and mode.startswith('{'):
                try:
                    parsed_input = json.loads(mode)
                    mode = parsed_input.get('mode', mode)
                    topic = parsed_input.get('topic', topic)
                except json.JSONDecodeError:
                    pass
            
            # Handle dict input
            if isinstance(mode, dict):
                mode = mode.get('mode', 'serp')
                topic = mode.get('topic', topic)
            
            # Normalize mode names - handle various input formats
            if isinstance(mode, str):
                mode_lower = mode.lower().strip()
            else:
                mode_lower = str(mode).lower().strip()
            
            # Map various mode names to standard modes
            mode_mapping = {
                # SERP modes
                "serp": "serp",
                "a": "serp", 
                "serp_analysis": "serp",
                "web_search": "serp",
                "google_search": "serp",
                "search": "serp",
                
                # RAG modes
                "rag": "rag",
                "b": "rag",
                "internal_knowledge_base": "rag",
                "knowledge_base": "rag",
                "internal": "rag",
                "local": "rag",
                "documents": "rag",
                
                # Reference modes
                "reference": "reference",
                "c": "reference",
                "reference_articles": "reference",
                "upload": "reference",
                "uploaded": "reference",
                "files": "reference"
            }
            
            # Get the normalized mode
            normalized_mode = mode_mapping.get(mode_lower, mode_lower)
            
            # Execute based on normalized mode
            if normalized_mode == "serp":
                num_results = kwargs.get("num_results", 10)
                result = self.serp_analysis(topic, num_results)
            
            elif normalized_mode == "rag":
                top_k = kwargs.get("top_k", 10)
                result = self.internal_knowledge_retrieval(topic, top_k)
            
            elif normalized_mode == "reference":
                content_list = kwargs.get("content_list", [])
                result = self.process_reference_articles(content_list, topic)
            
            else:
                result = {
                    "error": f"Unknown research mode: '{mode}'. Valid modes are: 'serp' (for web search), 'rag' (for knowledge base), or 'reference' (for uploaded content)",
                    "valid_modes": ["serp", "rag", "reference"],
                    "mode_received": mode,
                    "normalized_mode": normalized_mode,
                    "topic": topic,
                    "suggestions": [
                        "Use 'serp' for web-based research",
                        "Use 'rag' for knowledge base queries",
                        "Use 'reference' for uploaded file analysis"
                    ]
                }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Research mode execution failed: {e}")
            return json.dumps({
                "error": str(e),
                "mode": mode,
                "topic": topic,
                "valid_modes": ["serp", "rag", "reference"],
                "suggestions": [
                    "Check the mode parameter format",
                    "Ensure topic is provided",
                    "Verify all required parameters are set"
                ]
            }, indent=2)
class CompetitorAnalysisTool(BaseTool):
    name: str = "Competitor Analysis Tool"
    description: str = "Analyze competitor blogs for tone, structure, keywords, and formatting"
    
    # Declare Pydantic fields
    serp_api_key: Optional[str] = Field(default=None)
    firecrawl_app: Optional[Any] = Field(default=None, exclude=True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'serp_api_key', os.getenv("SERP_API_KEY"))
        
        firecrawl_key = os.getenv("FIRECRAWL_API_KEY")
        if firecrawl_key:
            try:
                firecrawl_app = Firecrawl(api_key=firecrawl_key)
                object.__setattr__(self, 'firecrawl_app', firecrawl_app)
            except Exception as e:
                logger.warning(f"FireCrawl init failed: {e}")
                object.__setattr__(self, 'firecrawl_app', None)
        else:
            logger.warning("FIRECRAWL_API_KEY not found")
            object.__setattr__(self, 'firecrawl_app', None)
    
    def find_competitor_urls(self, topic: str, num_urls: int = 5) -> List[str]:
        """Find competitor URLs via search"""
        try:
            if not self.serp_api_key:
                # Return empty list instead of raising exception
                logger.warning("SERP API key not found, cannot search for competitor URLs")
                return []

            params = {
                "q": topic,
                "hl": "en",
                "gl": "us",
                "num": num_urls,
                "google_domain": "google.com",
                "api_key": self.serp_api_key
            }

            search = GoogleSearch(params)
            data = search.get_dict()
            
            # Add error handling for SERP API response
            if 'error' in data:
                logger.error(f"SERP API error: {data['error']}")
                return []

            urls = []
            organic_results = data.get("organic_results", [])
            
            if not organic_results:
                logger.warning("No organic results found in SERP response")
                return []
            
            for result in organic_results:
                url = result.get("link", "")
                if url and url.startswith(('http://', 'https://')):
                    urls.append(url)
            
            logger.info(f"Found {len(urls)} competitor URLs for topic: {topic}")
            return urls[:num_urls]
            
        except Exception as e:
            logger.error(f"Failed to find competitor URLs: {e}")
            return []  # Return empty list instead of raising
    
    def analyze_competitors(self, urls: List[str], topic: str) -> Dict[str, Any]:
        """Analyze competitor content with improved error handling"""
        try:
            if not urls:
                return {
                    "error": "No URLs provided for analysis",
                    "topic": topic,
                    "competitors": [],
                    "suggestions": [
                        "Provide competitor URLs manually",
                        "Check SERP API configuration",
                        "Verify topic search returns results"
                    ]
                }
            
            competitor_data = []
            failed_urls = []
            
            for url in urls:
                try:
                    logger.info(f"Analyzing competitor URL: {url}")
                    content = self._scrape_content(url)
                    
                    if not content or len(content.strip()) < 100:
                        logger.warning(f"Insufficient content scraped from {url}")
                        failed_urls.append(url)
                        continue
                    
                    analysis = self._analyze_content(content, url, topic)
                    competitor_data.append(analysis)
                    logger.info(f"Successfully analyzed {url}")
                    
                except Exception as e:
                    logger.error(f"Failed to analyze {url}: {e}")
                    failed_urls.append(url)
                    continue
            
            if not competitor_data:
                return {
                    "error": "No competitor content successfully analyzed",
                    "topic": topic,
                    "competitors": [],
                    "failed_urls": failed_urls,
                    "suggestions": [
                        "Check if URLs are accessible",
                        "Verify FireCrawl API configuration",
                        "Try with different competitor URLs"
                    ]
                }
            
            comparison = self._generate_comparison(competitor_data, topic)
            
            return {
                "topic": topic,
                "competitors": competitor_data,
                "comparison_grid": comparison,
                "total_analyzed": len(competitor_data),
                "failed_urls": failed_urls,
                "success_rate": f"{len(competitor_data)}/{len(urls)}"
            }
            
        except Exception as e:
            logger.error(f"Competitor analysis failed: {e}")
            return {
                "error": str(e),
                "topic": topic,
                "competitors": [],
                "suggestions": [
                    "Check tool configuration",
                    "Verify API keys",
                    "Try with simpler topic"
                ]
            }
    
    def _scrape_content(self, url: str) -> Optional[str]:
        """Scrape content with improved fallback and error handling"""
        content = None
        
        # Try FireCrawl first
        if self.firecrawl_app:
            try:
                logger.info(f"Attempting FireCrawl scraping for: {url}")
                
                result = self.firecrawl_app.scrape(
                    url,
                    formats=["markdown"],
                    only_main_content=False,
                    timeout=120000      
                )
                
                return result.markdown
            
            except Exception as e:
                logger.warning(f"FireCrawl failed for {url}: {e}")
        
        # Fallback to basic scraping
        try:
            logger.info(f"Attempting basic scraping for: {url}")
            return self._basic_scrape(url)
        except Exception as e:
            logger.error(f"Basic scraping also failed for {url}: {e}")
            return None
        
    def _basic_scrape(self, url: str) -> Optional[str]:
        """Basic web scraping with improved error handling"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                logger.warning(f"URL {url} returned non-HTML content: {content_type}")
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()
            
            # Try to find main content with multiple selectors
            content_selectors = [
                'article', 'main', '.content', '#content', 
                '.post-content', '.entry-content', '.blog-content',
                '[role="main"]', '.container'
            ]
            
            content = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    content = ' '.join([elem.get_text(strip=True, separator=' ') for elem in elements])
                    if len(content.strip()) > 100:  # Minimum content threshold
                        break
            
            # Fallback to body text if no specific content found
            if not content or len(content.strip()) < 100:
                content = soup.get_text(strip=True, separator=' ')
            
            # Clean up the content
            content = re.sub(r'\s+', ' ', content.strip())
            
            return content if len(content.strip()) >= 100 else None
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Basic scraping failed for {url}: {e}")
            return None

    def _analyze_content(self, content: str, url: str, topic: str) -> Dict[str, Any]:
        """Analyze individual competitor content with error handling"""
        try:
            words = content.split()
            
            # Basic validation
            if len(words) < 10:
                logger.warning(f"Content too short for analysis: {url}")
                return self._create_minimal_analysis(url, topic)
            
            return {
                "url": url,
                "domain": self._extract_domain(url),
                "word_count": len(words),
                "char_count": len(content),
                "tone": self._analyze_tone(content),
                "structure": self._analyze_structure(content),
                "keywords": self._analyze_keywords(content, topic),
                "formatting": self._analyze_formatting(content),
                "readability": self._analyze_readability(content),
                "content_quality": self._assess_content_quality(content, topic)
            }
        except Exception as e:
            logger.error(f"Content analysis failed for {url}: {e}")
            return self._create_minimal_analysis(url, topic, error=str(e))
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL safely"""
        try:
            if '://' in url:
                return url.split('/')[2]
            return url.split('/')[0]
        except:
            return "unknown"
    
    def _create_minimal_analysis(self, url: str, topic: str, error: str = None) -> Dict[str, Any]:
        """Create minimal analysis structure for failed cases"""
        return {
            "url": url,
            "domain": self._extract_domain(url),
            "word_count": 0,
            "error": error,
            "tone": {"primary_tone": "unknown", "confidence": 0},
            "structure": {"header_count": 0, "paragraph_count": 0},
            "keywords": {"topic_keyword_usage": {}, "relevance_score": 0},
            "formatting": {"readability_score": 0}
        }
    
    def _analyze_tone(self, content: str) -> Dict[str, Any]:
        """Analyze content tone with improved detection"""
        try:
            content_lower = content.lower()
            
            # Formal indicators
            formal_indicators = [
                'furthermore', 'therefore', 'consequently', 'however', 'moreover',
                'nevertheless', 'thus', 'hence', 'accordingly', 'subsequently'
            ]
            
            # Casual indicators
            casual_indicators = [
                "you'll", "we'll", "let's", "here's", "don't", "can't", "won't",
                "hey", "okay", "cool", "awesome", "yeah"
            ]
            
            formal_count = sum(1 for word in formal_indicators if word in content_lower)
            casual_count = sum(1 for word in casual_indicators if word in content_lower)
            
            # Sentence analysis
            sentences = [s.strip() for s in content.split('.') if s.strip()]
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
            
            # Determine primary tone
            if formal_count > casual_count:
                primary_tone = "formal"
                confidence = min(0.9, formal_count / max(1, formal_count + casual_count))
            elif casual_count > formal_count:
                primary_tone = "casual"
                confidence = min(0.9, casual_count / max(1, formal_count + casual_count))
            else:
                primary_tone = "neutral"
                confidence = 0.5
            
            return {
                "primary_tone": primary_tone,
                "confidence": round(confidence, 2),
                "avg_sentence_length": round(avg_sentence_length, 1),
                "uses_second_person": " you " in content_lower or "your " in content_lower,
                "question_count": content.count('?'),
                "exclamation_count": content.count('!'),
                "formal_score": formal_count,
                "casual_score": casual_count
            }
        except Exception as e:
            logger.error(f"Tone analysis failed: {e}")
            return {"primary_tone": "unknown", "error": str(e)}
    
    def _analyze_structure(self, content: str) -> Dict[str, Any]:
        """Analyze content structure with improved detection"""
        try:
            lines = content.split('\n')
            
            # Count different types of headers
            markdown_headers = len([line for line in lines if line.strip().startswith('#')])
            
            # Count list items (improved detection)
            list_patterns = [r'^\s*[-*•]\s+', r'^\s*\d+\.\s+', r'^\s*[a-zA-Z]\.\s+']
            list_items = 0
            for line in lines:
                for pattern in list_patterns:
                    if re.match(pattern, line):
                        list_items += 1
                        break
            
            # Analyze paragraphs
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip() and len(p.strip()) > 10]
            avg_paragraph_length = sum(len(p.split()) for p in paragraphs) / len(paragraphs) if paragraphs else 0
            
            return {
                "header_count": markdown_headers,
                "list_items": list_items,
                "paragraph_count": len(paragraphs),
                "avg_paragraph_length": round(avg_paragraph_length, 1),
                "line_count": len(lines),
                "structure_score": min(10, markdown_headers + (list_items / 5) + (len(paragraphs) / 10))
            }
        except Exception as e:
            logger.error(f"Structure analysis failed: {e}")
            return {"header_count": 0, "paragraph_count": 0, "error": str(e)}
    
    def _analyze_keywords(self, content: str, topic: str) -> Dict[str, Any]:
        """Analyze keyword usage with improved relevance scoring"""
        try:
            content_lower = content.lower()
            words = re.findall(r'\b\w+\b', content_lower)
            
            if not words:
                return {"topic_keyword_usage": {}, "relevance_score": 0}
            
            # Topic keyword analysis
            topic_words = re.findall(r'\b\w+\b', topic.lower())
            topic_mentions = {}
            
            for word in topic_words:
                if len(word) > 2:  # Skip short words
                    count = content_lower.count(word)
                    if count > 0:
                        topic_mentions[word] = count
            
            # Calculate relevance score
            total_topic_mentions = sum(topic_mentions.values())
            relevance_score = min(1.0, total_topic_mentions / len(words) * 100) if words else 0
            
            # Get word frequency
            from collections import Counter
            word_freq = Counter(words)
            
            # Filter out common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
            filtered_freq = {word: count for word, count in word_freq.items() if word not in stop_words and len(word) > 2}
            
            top_keywords = [{"word": word, "count": count, "density": round(count/len(words)*100, 2)} 
                          for word, count in Counter(filtered_freq).most_common(10)]
            
            return {
                "topic_keyword_usage": topic_mentions,
                "relevance_score": round(relevance_score, 3),
                "top_keywords": top_keywords,
                "total_words": len(words),
                "unique_words": len(set(words)),
                "keyword_density": round(len(set(words)) / len(words), 3) if words else 0
            }
        except Exception as e:
            logger.error(f"Keyword analysis failed: {e}")
            return {"topic_keyword_usage": {}, "error": str(e)}
    
    def _analyze_formatting(self, content: str) -> Dict[str, Any]:
        """Analyze formatting elements with improved detection"""
        try:
            return {
                "bold_usage": content.count('**') // 2,
                "italic_usage": content.count('*') - (content.count('**')),
                "link_count": len(re.findall(r'https?://\S+', content)) + content.count('['),
                "code_blocks": content.count('```') // 2,
                "inline_code": content.count('`') - (content.count('```') * 3),
                "lists": content.count('\n-') + content.count('\n*') + len(re.findall(r'\n\d+\.', content)),
                "line_breaks": content.count('\n'),
                "formatting_richness": self._calculate_formatting_score(content)
            }
        except Exception as e:
            logger.error(f"Formatting analysis failed: {e}")
            return {"formatting_richness": 0, "error": str(e)}
    
    def _calculate_formatting_score(self, content: str) -> float:
        """Calculate overall formatting richness score"""
        try:
            score = 0
            score += min(5, content.count('**') // 2)  # Bold text
            score += min(3, content.count('`'))  # Code snippets  
            score += min(4, content.count('\n#'))  # Headers
            score += min(3, content.count('\n-') + content.count('\n*'))  # Lists
            score += min(2, len(re.findall(r'https?://\S+', content)))  # Links
            return min(10, score)
        except:
            return 0
    
    def _analyze_readability(self, content: str) -> Dict[str, Any]:
        """Basic readability analysis"""
        try:
            sentences = [s.strip() for s in re.split(r'[.!?]+', content) if s.strip()]
            words = content.split()
            
            if not sentences or not words:
                return {"readability_score": 0}
            
            avg_words_per_sentence = len(words) / len(sentences)
            
            # Simple readability score (lower is easier to read)
            if avg_words_per_sentence < 15:
                readability = "Easy"
                score = 8
            elif avg_words_per_sentence < 20:
                readability = "Medium"
                score = 6
            else:
                readability = "Difficult"
                score = 4
            
            return {
                "avg_words_per_sentence": round(avg_words_per_sentence, 1),
                "sentence_count": len(sentences),
                "readability_level": readability,
                "readability_score": score
            }
        except Exception as e:
            logger.error(f"Readability analysis failed: {e}")
            return {"readability_score": 0, "error": str(e)}
    
    def _assess_content_quality(self, content: str, topic: str) -> Dict[str, Any]:
        """Assess overall content quality"""
        try:
            quality_score = 0
            
            # Length check
            word_count = len(content.split())
            if word_count > 300:
                quality_score += 2
            if word_count > 800:
                quality_score += 2
            
            # Topic relevance
            topic_words = topic.lower().split()
            content_lower = content.lower()
            relevance_count = sum(1 for word in topic_words if word in content_lower)
            if relevance_count >= len(topic_words) * 0.5:
                quality_score += 3
            
            # Structure elements
            if '#' in content or content.count('\n\n') > 3:
                quality_score += 2
            
            # Engagement elements
            if '?' in content:
                quality_score += 1
            
            quality_level = "Low"
            if quality_score >= 7:
                quality_level = "High"
            elif quality_score >= 4:
                quality_level = "Medium"
            
            return {
                "quality_score": min(10, quality_score),
                "quality_level": quality_level,
                "word_count_adequate": word_count >= 300,
                "topic_relevant": relevance_count >= len(topic_words) * 0.3,
                "well_structured": '#' in content or content.count('\n\n') > 2
            }
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return {"quality_score": 0, "error": str(e)}
    
    def _generate_comparison(self, competitor_data: List[Dict], topic: str) -> Dict[str, Any]:
        """Generate comprehensive comparison insights"""
        try:
            if not competitor_data:
                return {"error": "No competitor data to compare"}
            
            # Calculate averages
            valid_data = [comp for comp in competitor_data if comp.get("word_count", 0) > 0]
            
            if not valid_data:
                return {"error": "No valid competitor data for comparison"}
            
            avg_word_count = sum(comp["word_count"] for comp in valid_data) / len(valid_data)
            
            # Analyze tones
            tones = [comp.get("tone", {}).get("primary_tone", "unknown") for comp in valid_data]
            most_common_tone = max(set(tones), key=tones.count) if tones else "unknown"
            
            # Structure analysis
            avg_headers = sum(comp.get("structure", {}).get("header_count", 0) for comp in valid_data) / len(valid_data)
            avg_paragraphs = sum(comp.get("structure", {}).get("paragraph_count", 0) for comp in valid_data) / len(valid_data)
            
            # Quality scores
            quality_scores = [comp.get("content_quality", {}).get("quality_score", 0) for comp in valid_data]
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
            
            return {
                "average_metrics": {
                    "word_count": round(avg_word_count),
                    "header_count": round(avg_headers, 1),
                    "paragraph_count": round(avg_paragraphs, 1),
                    "quality_score": round(avg_quality, 1),
                    "common_tone": most_common_tone
                },
                "recommendations": {
                    "target_word_count": f"{int(avg_word_count * 1.1)}-{int(avg_word_count * 1.3)} words",
                    "tone_suggestion": f"Consider {most_common_tone} tone to match competitors",
                    "structure_advice": f"Aim for {int(avg_headers)} headers and {int(avg_paragraphs)} paragraphs",
                    "quality_target": f"Target quality score: {min(10, int(avg_quality + 1))}/10"
                },
                "competitive_gaps": self._identify_gaps(valid_data, topic),
                "best_performers": self._identify_top_performers(valid_data)
            }
        except Exception as e:
            logger.error(f"Comparison generation failed: {e}")
            return {"error": str(e)}
    
    def _identify_gaps(self, competitor_data: List[Dict], topic: str) -> List[str]:
        """Identify potential competitive gaps"""
        gaps = []
        
        try:
            # Check for common weaknesses
            low_quality_count = sum(1 for comp in competitor_data 
                                  if comp.get("content_quality", {}).get("quality_score", 0) < 6)
            
            if low_quality_count > len(competitor_data) * 0.5:
                gaps.append("Many competitors have low content quality - opportunity for high-quality content")
            
            # Check word count distribution
            word_counts = [comp.get("word_count", 0) for comp in competitor_data]
            if max(word_counts) - min(word_counts) > 1000:
                gaps.append("Wide variation in content length - opportunity for optimal length content")
            
            # Check topic relevance
            low_relevance = sum(1 for comp in competitor_data 
                              if comp.get("keywords", {}).get("relevance_score", 0) < 0.1)
            
            if low_relevance > 0:
                gaps.append("Some competitors have low topic relevance - opportunity for focused content")
                
        except Exception as e:
            logger.error(f"Gap identification failed: {e}")
            gaps.append("Unable to identify gaps due to analysis error")
        
        return gaps[:5]  # Limit to top 5 gaps
    
    def _identify_top_performers(self, competitor_data: List[Dict]) -> List[Dict]:
        """Identify top performing competitors"""
        try:
            # Sort by quality score and word count
            sorted_competitors = sorted(
                competitor_data,
                key=lambda x: (
                    x.get("content_quality", {}).get("quality_score", 0),
                    x.get("word_count", 0)
                ),
                reverse=True
            )
            
            top_performers = []
            for comp in sorted_competitors[:3]:  # Top 3
                top_performers.append({
                    "url": comp.get("url", ""),
                    "domain": comp.get("domain", ""),
                    "word_count": comp.get("word_count", 0),
                    "quality_score": comp.get("content_quality", {}).get("quality_score", 0),
                    "why_top": self._explain_top_performance(comp)
                })
            
            return top_performers
            
        except Exception as e:
            logger.error(f"Top performer identification failed: {e}")
            return []
    
    def _explain_top_performance(self, competitor: Dict) -> str:
        """Explain why a competitor is a top performer"""
        reasons = []
        
        try:
            word_count = competitor.get("word_count", 0)
            quality_score = competitor.get("content_quality", {}).get("quality_score", 0)
            structure = competitor.get("structure", {})
            
            if word_count > 800:
                reasons.append("comprehensive content")
            if quality_score >= 7:
                reasons.append("high quality")
            if structure.get("header_count", 0) > 3:
                reasons.append("well-structured")
            if structure.get("list_items", 0) > 5:
                reasons.append("good use of lists")
            
            return ", ".join(reasons) if reasons else "overall strong performance"
            
        except:
            return "strong overall metrics"
    
    # Define explicit args schema for tool invocation
    class CompetitorAnalysisToolInput(BaseModel):
        topic: str
        competitor_urls: Optional[List[str]] = None

    args_schema = CompetitorAnalysisToolInput

    def _run(self, topic: str, competitor_urls: Optional[List[str]] = None) -> str:
        """Execute competitor analysis with comprehensive error handling"""
        num_competitors = 0
        try:
            logger.info(f"Starting competitor analysis for topic: {topic}")
            logger.info(f"Received competitor_urls: {competitor_urls} (type: {type(competitor_urls)})")
            
            # Handle various input formats from agent
            if competitor_urls is None or competitor_urls == "null" or competitor_urls == []:
                competitor_urls = None
                num_competitors = 5  # Default number for auto-discovery
                logger.info("No competitor URLs provided, will auto-discover")
            elif isinstance(competitor_urls, str):
                # Handle case where agent passes string instead of list
                if competitor_urls.strip() == "" or competitor_urls.strip() == "null":
                    competitor_urls = None
                    num_competitors = 5
                else:
                    # Try to parse as JSON or split by comma
                    try:
                        competitor_urls = json.loads(competitor_urls)
                    except:
                        competitor_urls = [url.strip() for url in competitor_urls.split(',') if url.strip()]
                    num_competitors = len(competitor_urls)
            else:
                num_competitors = len(competitor_urls)
            
            # Validate inputs
            if not topic or not isinstance(topic, str):
                return json.dumps({
                    "error": "Invalid topic provided",
                    "topic": topic,
                    "suggestions": ["Provide a valid topic string"]
                }, indent=2)
            
            # Find URLs if not provided
            if not competitor_urls:
                logger.info(f"Finding competitor URLs for topic: {topic}")
                competitor_urls = self.find_competitor_urls(topic, num_competitors)
                
                if not competitor_urls:
                    return json.dumps({
                        "error": "No competitor URLs found via search",
                        "topic": topic,
                        "suggestions": [
                            "Provide competitor URLs manually",
                            "Check SERP API configuration",
                            "Try a more specific topic",
                            "Verify SERP_API_KEY environment variable"
                        ]
                    }, indent=2)
            
            # Validate URLs
            valid_urls = [url for url in competitor_urls if url.startswith(('http://', 'https://'))]
            if not valid_urls:
                return json.dumps({
                    "error": "No valid URLs provided",
                    "invalid_urls": competitor_urls,
                    "suggestions": ["Ensure URLs start with http:// or https://"]
                }, indent=2)
            
            logger.info(f"Analyzing {len(valid_urls)} competitor URLs")
            result = self.analyze_competitors(valid_urls, topic)
            
            # Add execution metadata
            result["execution_metadata"] = {
                "timestamp": datetime.now().isoformat(),
                "urls_attempted": len(valid_urls),
                "analysis_successful": "competitors" in result and len(result.get("competitors", [])) > 0
            }
            
            logger.info(f"Competitor analysis completed. Success: {result['execution_metadata']['analysis_successful']}")
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Competitor analysis execution failed: {e}")
            import traceback
            traceback.print_exc()
            
            return json.dumps({
                "error": f"Competitor analysis execution failed: {str(e)}",
                "topic": topic,
                "traceback": traceback.format_exc()[-500:],  # Last 500 chars of traceback
                "suggestions": [
                    "Check API keys configuration",
                    "Verify internet connectivity", 
                    "Try with simpler topic or manual URLs",
                    "Check tool initialization"
                ]
            }, indent=2)
        

class WebSearchTool(BaseTool):
    name: str = "Web Search Tool"
    description: str = "Search the web using SERP API and return relevant results"
    
    # Define explicit args schema so CrewAI passes concrete values (not schema dict)
    class WebSearchToolInput(BaseModel):
        query: str
        num_results: int = 5

    args_schema = WebSearchToolInput
    
    def _run(self, query: str, num_results: int = 5) -> str:
        """Perform web search using SERP API"""
        try:
            serp_api_key = os.getenv("SERP_API_KEY")
            
            if not serp_api_key:
                return json.dumps({
                    "error": "SERP_API_KEY environment variable not set",
                    "query": query,
                    "results": []
                }, indent=2)
            
            params = {
                "engine": "google",
                "q": query,
                "api_key": serp_api_key,
                "num": num_results,
                "gl": "us",
                "hl": "en"
            }
            
            response = requests.get("https://serpapi.com/search", params=params)
            response.raise_for_status()
            
            data = response.json()
            
            results = []
            organic_results = data.get("organic_results", [])
            
            for result in organic_results:
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("link", ""),
                    "snippet": result.get("snippet", ""),
                    "relevance_score": 0.9  # High relevance for SERP results
                })
            
            search_results = {
                "query": query,
                "results": results,
                "total_found": len(results),
                "timestamp": datetime.now().isoformat()
            }
            
            return json.dumps(search_results, indent=2)
            
        except requests.RequestException as e:
            logger.error(f"SERP API request failed: {e}")
            return json.dumps({
                "error": f"SERP API request failed: {str(e)}",
                "query": query,
                "results": []
            }, indent=2)
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return json.dumps({
                "error": f"Web search failed: {str(e)}",
                "query": query,
                "results": []
            }, indent=2)