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
from pydantic import Field
from serpapi import GoogleSearch
from src.blog_generator_ai_agent.utils.constants import EMBEDDING_MODEL_RAG, CHUNK_SIZE, CHUNK_OVERLAP

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
        try:
            if self.firecrawl_app:
                result = self.firecrawl_app.scrape(url, formats= ['markdown'])
                
                if result and 'markdown' in result:
                    return result['markdown']
            
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
                raise ValueError("SERP API key not found. Please set the SERP_API_KEY environment variable.")

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
            print(data)

            urls = []
            for result in data.get("organic_results", []):
                url = result.get("link", "")
                if url:
                    urls.append(url)
            
            return urls[:num_urls]
            
        except Exception as e:
            logger.error(f"Failed to find competitor URLs: {e}")
            raise
    
    def analyze_competitors(self, urls: List[str], topic: str) -> Dict[str, Any]:
        """Analyze competitor content"""
        try:
            competitor_data = []
            
            for url in urls:
                try:
                    content = self._scrape_content(url)
                    if not content:
                        logger.warning(f"No content scraped from {url}")
                        continue
                    
                    analysis = self._analyze_content(content, url, topic)
                    competitor_data.append(analysis)
                    
                except Exception as e:
                    logger.error(f"Failed to analyze {url}: {e}")
                    continue
            
            if not competitor_data:
                return {
                    "error": "No competitor content successfully analyzed",
                    "topic": topic,
                    "competitors": []
                }
            
            comparison = self._generate_comparison(competitor_data, topic)
            
            return {
                "topic": topic,
                "competitors": competitor_data,
                "comparison_grid": comparison,
                "total_analyzed": len(competitor_data)
            }
            
        except Exception as e:
            logger.error(f"Competitor analysis failed: {e}")
            return {
                "error": str(e),
                "topic": topic,
                "competitors": []
            }
    
    def _scrape_content(self, url: str) -> Optional[str]:
        """Scrape content from URL using Firecrawl"""
        if not self.firecrawl_app:
            logger.warning("Firecrawl not initialized")
            return None
            
        try:
            # Use the correct Firecrawl API format
            result = self.firecrawl_app.scrape(
                url,
                formats=["markdown"]
            )
            return result.get('markdown')
        except Exception as e:
            logger.error(f"URL scraping failed for {url}: {e}")
            return None
    
    def _analyze_content(self, content: str, url: str, topic: str) -> Dict[str, Any]:
        """Analyze individual competitor content"""
        words = content.split()
        
        return {
            "url": url,
            "domain": url.split('/')[2] if '://' in url else url,
            "word_count": len(words),
            "tone": self._analyze_tone(content),
            "structure": self._analyze_structure(content),
            "keywords": self._analyze_keywords(content, topic),
            "formatting": self._analyze_formatting(content)
        }
    
    def _analyze_tone(self, content: str) -> Dict[str, Any]:
        """Analyze content tone"""
        content_lower = content.lower()
        
        formal_words = ['furthermore', 'therefore', 'consequently', 'however']
        casual_words = ["you'll", "we'll", "let's", "here's"]
        
        formal_count = sum(1 for word in formal_words if word in content_lower)
        casual_count = sum(1 for word in casual_words if word in content_lower)
        
        sentences = content.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        return {
            "primary_tone": "formal" if formal_count > casual_count else "casual",
            "avg_sentence_length": round(avg_sentence_length, 1),
            "uses_second_person": " you " in content_lower,
            "question_count": content.count('?')
        }
    
    def _analyze_structure(self, content: str) -> Dict[str, Any]:
        """Analyze content structure"""
        lines = content.split('\n')
        
        # Count headers (lines starting with # or being short and uppercase)
        headers = [line for line in lines if line.startswith('#') or (len(line) < 100 and line.isupper())]
        
        # Count lists
        list_items = sum(1 for line in lines if line.strip().startswith(('•', '-', '*', '1.', '2.', '3.')))
        
        paragraphs = [p for p in content.split('\n\n') if p.strip()]
        
        return {
            "header_count": len(headers),
            "list_items": list_items,
            "paragraph_count": len(paragraphs),
            "avg_paragraph_length": sum(len(p.split()) for p in paragraphs) / len(paragraphs) if paragraphs else 0
        }
    
    def _analyze_keywords(self, content: str, topic: str) -> Dict[str, Any]:
        """Analyze keyword usage"""
        content_lower = content.lower()
        words = content_lower.split()
        
        topic_words = topic.lower().split()
        topic_mentions = {}
        
        for word in topic_words:
            if len(word) > 2:
                count = content_lower.count(word)
                topic_mentions[word] = count
        
        # Get word frequency
        from collections import Counter
        word_freq = Counter(words)
        common_words = word_freq.most_common(10)
        
        return {
            "topic_keyword_usage": topic_mentions,
            "top_keywords": [{"word": word, "count": count} for word, count in common_words],
            "total_words": len(words)
        }
    
    def _analyze_formatting(self, content: str) -> Dict[str, Any]:
        """Analyze formatting elements"""
        return {
            "bold_usage": content.count('**') // 2,
            "italic_usage": content.count('*') - (content.count('**') // 2 * 2),
            "link_count": content.count('http') + content.count('['),
            "code_blocks": content.count('```') // 2,
            "lists": content.count('\n-') + content.count('\n*')
        }
    
    def _generate_comparison(self, competitor_data: List[Dict], topic: str) -> Dict[str, Any]:
        """Generate comparison insights"""
        if not competitor_data:
            return {}
        
        avg_word_count = sum(comp["word_count"] for comp in competitor_data) / len(competitor_data)
        
        common_tones = [comp["tone"]["primary_tone"] for comp in competitor_data]
        most_common_tone = max(set(common_tones), key=common_tones.count) if common_tones else "unknown"
        
        return {
            "average_metrics": {
                "word_count": round(avg_word_count),
                "common_tone": most_common_tone,
                "avg_headers": round(sum(comp["structure"]["header_count"] for comp in competitor_data) / len(competitor_data))
            },
            "recommendations": {
                "target_word_count": f"{int(avg_word_count * 1.2)}-{int(avg_word_count * 1.5)} words",
                "tone_suggestion": f"Consider {most_common_tone} tone to match competitors",
                "structure_advice": "Use clear headers and formatting based on competitor analysis"
            }
        }
    
    def _run(self, topic: str, competitor_urls: List[str] = None, num_competitors: int = 5) -> str:
        """Execute competitor analysis"""
        try:
            # Find URLs if not provided
            if not competitor_urls:
                competitor_urls = self.find_competitor_urls(topic, num_competitors)
            
            if not competitor_urls:
                return json.dumps({
                    "error": "No competitor URLs found",
                    "topic": topic
                }, indent=2)
            
            result = self.analyze_competitors(competitor_urls, topic)
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Competitor analysis execution failed: {e}")
            return json.dumps({
                "error": str(e),
                "topic": topic
            }, indent=2)

class WebSearchTool(BaseTool):
    name: str = "Web Search Tool"
    description: str = "Search the web using SERP API and return relevant results"
    
    def _run(self, query: str, num_results: int = 10) -> str:
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

