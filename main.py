from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any
import uuid
import json
import os
import tempfile
import asyncio
import re
from datetime import datetime
from pathlib import Path
import logging
import traceback

from src.blog_generator_ai_agent.crew import BlogGeneratorCrew
from src.blog_generator_ai_agent.models.pydantic_models import (
    TopicGenerationOutput, ResearchOutput, CompetitorAnalysisOutput,
    KeywordStrategyOutput, TitleGenerationOutput, ContentStructureOutput,
    BlogGenerationOutput, CompleteWorkflowOutput
)
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Blog Generator AI Agent",
    description="Agentic RAG-enabled blog generation system with structured outputs",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage for sessions (In production, use Redis/Database)
sessions: Dict[str, Dict] = {}
results_storage: Dict[str, Dict] = {}

# Initialize the blog generator crew
blog_crew = BlogGeneratorCrew()

# Enhanced Error Handling Utilities
class UserFriendlyError(Exception):
    """Custom exception for user-friendly error messages"""
    def __init__(self, message: str, technical_details: str = None, error_code: str = None):
        self.message = message
        self.technical_details = technical_details
        self.error_code = error_code
        super().__init__(self.message)

def extract_user_friendly_error(error: Exception) -> Dict[str, Any]:
    """Extract user-friendly error message from various exception types"""
    error_str = str(error).lower()
    error_traceback = traceback.format_exc()
    
    # Common AI/API errors
    if "rate limit" in error_str or "quota" in error_str or "429" in error_str:
        return {
            "user_message": "We've hit our AI service limit for today. Please try again tomorrow or upgrade your plan.",
            "technical_details": "Rate limit exceeded - API quota reached",
            "error_code": "RATE_LIMIT_EXCEEDED",
            "suggestions": [
                "Wait a few minutes and try again",
                "Check your AI service plan and billing",
                "Consider upgrading to a higher tier plan"
            ]
        }
    
    elif "authentication" in error_str or "unauthorized" in error_str or "401" in error_str:
        return {
            "user_message": "Authentication failed. Please check your API keys and credentials.",
            "technical_details": "Authentication/authorization error",
            "error_code": "AUTHENTICATION_FAILED",
            "suggestions": [
                "Verify your API keys are correct",
                "Check if your account has proper permissions",
                "Ensure your subscription is active"
            ]
        }
    
    elif "timeout" in error_str or "timed out" in error_str:
        return {
            "user_message": "The request took too long to complete. Please try again with a simpler request.",
            "technical_details": "Request timeout exceeded",
            "error_code": "REQUEST_TIMEOUT",
            "suggestions": [
                "Try breaking down your request into smaller parts",
                "Check your internet connection",
                "Wait a few minutes and try again"
            ]
        }
    
    elif "network" in error_str or "connection" in error_str:
        return {
            "user_message": "Network connection issue. Please check your internet connection and try again.",
            "technical_details": "Network connectivity problem",
            "error_code": "NETWORK_ERROR",
            "suggestions": [
                "Check your internet connection",
                "Try refreshing the page",
                "Check if the service is available"
            ]
        }
    
    elif "memory" in error_str or "out of memory" in error_str:
        return {
            "user_message": "The system is running low on memory. Please try a simpler request or contact support.",
            "technical_details": "Memory allocation error",
            "error_code": "MEMORY_ERROR",
            "suggestions": [
                "Try a simpler request",
                "Close other applications",
                "Contact support if the issue persists"
            ]
        }
    
    elif "validation" in error_str or "invalid" in error_str:
        return {
            "user_message": "The input data is not valid. Please check your input and try again.",
            "technical_details": "Input validation error",
            "error_code": "VALIDATION_ERROR",
            "suggestions": [
                "Review your input data",
                "Check required fields are filled",
                "Ensure data format is correct"
            ]
        }
    
    elif "litellm" in error_str or "vertexai" in error_str or "gemini" in error_str:
        if "quota" in error_str or "free_tier" in error_str:
            return {
                "user_message": "You've reached the free tier limit for AI services. Please upgrade your plan or try again tomorrow.",
                "technical_details": "AI service quota exceeded",
                "error_code": "AI_QUOTA_EXCEEDED",
                "suggestions": [
                    "Upgrade to a paid plan for higher limits",
                    "Wait until tomorrow when limits reset",
                    "Use a different AI service provider"
                ]
            }
        else:
            return {
                "user_message": "AI service is temporarily unavailable. Please try again in a few minutes.",
                "technical_details": "AI service error",
                "error_code": "AI_SERVICE_ERROR",
                "suggestions": [
                    "Wait a few minutes and try again",
                    "Check if the AI service is operational",
                    "Contact support if the issue persists"
                ]
            }
    
    # Default error message
    else:
        return {
            "user_message": "Something went wrong. Please try again or contact support if the issue persists.",
            "technical_details": f"Unexpected error: {str(error)}",
            "error_code": "UNKNOWN_ERROR",
            "suggestions": [
                "Try refreshing the page",
                "Check your input data",
                "Contact support with error details"
            ]
        }

def create_error_response(error: Exception, session_id: str = None) -> Dict[str, Any]:
    """Create a standardized error response"""
    error_info = extract_user_friendly_error(error)
    
    # Log the full error for debugging
    logger.error(f"Error occurred: {str(error)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    
    response = {
        "status": "error",
        "error": error_info,
        "timestamp": datetime.now().isoformat()
    }
    
    if session_id:
        response["session_id"] = session_id
    
    return response

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors"""
    error_response = create_error_response(exc)
    return JSONResponse(
        status_code=500,
        content=error_response
    )

# Enhanced Request Models
class TopicRequest(BaseModel):
    pillar: str = Field(..., description="Content pillar for topic generation")

class ResearchRequest(BaseModel):
    """Request model for research endpoint."""
    topic: str = Field(..., min_length=3, max_length=200, description="Topic to research")
    mode: str = Field(..., pattern="^(SERP|RAG|reference)$", description="Research mode: SERP, RAG, or reference")
    urls: List[HttpUrl] = Field(default_factory=list, description="List of URLs to include in research")
    uploads: List[str] = Field(default_factory=list, description="List of uploaded file references")
    research_method: str = Field("SERP", pattern="^(SERP|RAG|reference)$", 
                               description="Research method to use (SERP, RAG, reference)")
    pillar: Optional[str] = Field(None, min_length=3, max_length=100, 
                                 description="Content pillar for the research")

class CompetitorAnalysisRequest(BaseModel):
    urls: List[str] = Field(..., description="Competitor URLs to analyze")
    topic: str = Field(..., description="Topic for competitor analysis")
    research_method: str = Field("SERP", description="Research method to use (SERP, RAG, reference)")

class KeywordRequest(BaseModel):
    topic: str = Field(..., description="Topic for keyword generation")
    findings: Optional[str] = Field(None, description="Research findings")
    primary_keyword: Optional[str] = Field(None, description="Optional primary keyword")
    research_method: str = Field(default="SERP", description="Research method to use")
    pillar: Optional[str] = Field(None, description="Content pillar")

class TitleRequest(BaseModel):
    topic: str = Field(..., description="Topic for title generation")
    primary: str = Field(..., description="Primary keyword")
    secondary: List[str] = Field(..., description="Secondary keywords")
    research_method: str = Field("SERP", description="Research method to use")

class StructureRequest(BaseModel):
    topic: str = Field(..., description="Topic for structure")
    structure_type: str = Field(..., description="Structure type")
    keywords: Dict[str, Any] = Field(..., description="Keyword strategy")
    primary_keyword: str = Field(..., description="Primary keyword for the content")
    research_method: str = Field("SERP", description="Research method to use")

class OutlineRequest(BaseModel):
    topic: str = Field(..., description="Topic for outline")
    structure: str = Field(..., description="Content structure type")
    keywords: Dict[str, Any] = Field(..., description="Keyword strategy")
    research_method: str = Field("SERP", description="Research method to use")

class BlogGenerationRequest(BaseModel):
    topic: str = Field(..., description="Blog topic")
    structure_type: str = Field(..., description="Structure type")
    primary_keyword: str = Field(..., description="Primary keyword")
    keywords: Dict[str, Any] = Field(..., description="Keywords")
    brand_voice: str = Field(default="professional, helpful, concise", description="Brand voice guidelines")
    research_method: str = Field("SERP", description="Research method to use")

class WorkflowRequest(BaseModel):
    topic: str = Field(..., description="Blog topic")
    pillar: str = Field(..., description="Content pillar")
    mode: str = Field(default="SERP", description="Research mode")
    urls: Optional[List[str]] = None
    uploads: Optional[List[str]] = None
    structure: str = Field(default="How-to Guide", description="Blog structure type")
    brand_voice: str = Field(default="professional, helpful, concise", description="Brand voice")

# Enhanced Response Models using Pydantic models
class SessionResponse(BaseModel):
    session_id: str
    status: str
    message: str

# Utility functions
def create_session() -> str:
    """Create a new session ID"""
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "created_at": datetime.now().isoformat(),
        "status": "active",
        "steps_completed": []
    }
    return session_id

def get_session(session_id: str) -> Dict:
    """Get session data"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]

def update_session(session_id: str, step: str, data: Any):
    """Update session with step completion"""
    session = get_session(session_id)
    session["steps_completed"].append(step)
    session["last_updated"] = datetime.now().isoformat()
    results_storage[f"{session_id}_{step}"] = data

def convert_pydantic_to_dict(data):
    """Convert Pydantic model to dictionary for JSON serialization"""
    if hasattr(data, 'model_dump'):
        return data.model_dump()
    elif hasattr(data, 'dict'):
        return data.dict()
    else:
        return data

# API Endpoints with Structured Outputs
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/test/crew")
async def test_crew():
    """Test endpoint to check crew functionality"""
    try:
        test_inputs = {
            "topic": "test topic",
            "competitor_urls": ["https://example.com"],
            "pillar": "test"
        }
        
        logger.info("Testing crew functionality...")
        
        if not blog_crew:
            return {"status": "error", "message": "Blog crew not initialized"}
        
        crew_methods = [method for method in dir(blog_crew) if not method.startswith('_')]
        
        return {
            "status": "healthy",
            "crew_available": True,
            "crew_methods": crew_methods,
            "test_inputs": test_inputs,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error testing crew: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/topic/generate")
async def generate_topics(request: TopicRequest):
    """Generate topic suggestions based on pillar with structured output"""
    try:
        session_id = create_session()
        
        inputs = {
            "pillar": request.pillar,
            "topic": f"topics for {request.pillar}",
            "research_method": "SERP"
        }
        
        logger.info(f"Starting topic generation for pillar: {request.pillar}")
        
        # Get direct output from crew (now returns clean extracted result)
        result_data = blog_crew.run_topic_generation(inputs)
        
        # The result_data should now be the actual JSON content, not wrapped
        update_session(session_id, "topic_generation", result_data)
        
        topics_count = len(result_data.get("topics", []))
        logger.info(f"Generated {topics_count} topics for session {session_id}")
        
        # Return the actual data directly in the response
        return {
            "session_id": session_id,
            "status": "completed",
            "message": f"Generated {topics_count} topic suggestions",
            **result_data  # Merge the actual data directly into response
        }
        
    except Exception as e:
        logger.error(f"Error in topic generation: {e}", exc_info=True)
        
        # Use enhanced error handling
        error_response = create_error_response(e, session_id)
        
        # Create fallback response with error info
        fallback_data = {
            "topics": [],
            "pillar": request.pillar,
            "generation_date": datetime.now().strftime("%Y-%m-%d"),
            "total_suggestions": 0,
            "error_info": error_response["error"]
        }
        
        return {
            "session_id": session_id,
            "status": "completed_with_fallback", 
            "message": "Topic generation completed with fallback data due to processing error",
            **fallback_data
        }

@app.post("/research/run", status_code=200)
async def run_research(request: ResearchRequest):
    """Run research using specified mode with structured output"""
    session_id = create_session()
    
    try:
        logger.info(f"Starting research session: {session_id}")
        
        inputs = {
            "topic": request.topic.strip(),
            "research_method": request.research_method.upper(),
            "mode": request.mode.upper(),
            "urls": [str(url) for url in request.urls if url],
            "uploads": [upload for upload in request.uploads if upload],
            "pillar": (request.pillar or request.topic).strip()
        }
        
        logger.info(f"Research inputs: {inputs}")
        
        # Get direct output from crew (now returns clean extracted result)
        result_data = blog_crew.run_research_only(inputs)
        
        update_session(session_id, "research", result_data)
        
        findings_count = len(result_data.get("findings", []))
        logger.info(f"Research completed successfully for session {session_id} with {findings_count} findings")
        
        # Return the actual data directly in the response
        return {
            "session_id": session_id,
            "status": "completed",
            "message": f"Research completed with {findings_count} findings",
            **result_data  # Merge the actual data directly into response
        }
        
    except Exception as e:
        error_msg = f"Error during research: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Use enhanced error handling
        error_response = create_error_response(e, session_id)
        
        # Create fallback response structure
        fallback_data = {
            "topic": request.topic,
            "research_method": request.research_method,
            "key_insights": [],
            "statistics": [],
            "findings": [],
            "content_gaps": [],
            "talking_points": [],
            "sources": [],
            "research_date": datetime.now().isoformat(),
            "error_info": error_response["error"]
        }
        
        update_session(session_id, "research", fallback_data)
        
        return {
            "session_id": session_id,
            "status": "completed_with_fallback",
            "message": "Research completed with fallback data due to processing error",
            **fallback_data
        }

@app.post("/competitors/analyse")
async def analyze_competitors(request: CompetitorAnalysisRequest):
    """Analyze competitor content with structured output"""
    try:
        session_id = create_session()
        
        inputs = {
            "topic": request.topic,
            "competitor_urls": request.urls,
            "pillar": request.topic
        }
        
        logger.info(f"Starting competitor analysis for {len(request.urls)} URLs")
        
        # Get direct output from crew (now returns clean extracted result)
        result_data = blog_crew.run_competitor_analysis(inputs)
        
        update_session(session_id, "competitor_analysis", result_data)
        
        competitors_count = len(result_data.get("competitor_summaries", []))
        logger.info(f"Competitor analysis completed for {len(request.urls)} URLs")
        
        # Return the actual data directly in the response
        return {
            "session_id": session_id,
            "status": "completed", 
            "message": f"Analyzed {competitors_count} competitors successfully",
            **result_data  # Merge the actual data directly into response
        }
        
    except Exception as e:
        logger.error(f"Error in competitor analysis: {e}", exc_info=True)
        
        # Use enhanced error handling
        error_response = create_error_response(e, create_session())
        
        # Create fallback response
        fallback_data = {
            "topic": request.topic,
            "competitor_summaries": [],
            "analysis_date": datetime.now().isoformat(),
            "error_info": error_response["error"]
        }
        
        return {
            "session_id": error_response["session_id"],
            "status": "completed_with_fallback",
            "message": "Competitor analysis completed with fallback data due to processing error",
            **fallback_data
        }

@app.post("/seo/keywords")
async def generate_keywords(request: KeywordRequest):
    """Generate SEO keywords with structured output"""
    try:
        session_id = create_session()
        
        primary_keyword = request.primary_keyword or request.topic.replace(" ", "-").lower()
        inputs = {
            "topic": request.topic,
            "research_findings": request.findings or "",
            "primary_keyword": primary_keyword,
            "pillar": request.pillar or request.topic,
            "research_method": request.research_method
        }
        
        logger.info(f"Starting keyword strategy for topic: {request.topic}")
        
        # Get direct output from crew (now returns clean extracted result)
        result_data = blog_crew.run_keyword_strategy(inputs)
        
        update_session(session_id, "keyword_strategy", result_data)
        
        primary_kw = result_data.get("strategy", {}).get("primary_keyword", primary_keyword)
        logger.info(f"Keyword strategy completed for topic: {request.topic}")
        
        # Return the actual data directly in the response
        return {
            "session_id": session_id,
            "status": "completed",
            "message": f"Keyword strategy generated with primary: '{primary_kw}'",
            **result_data  # Merge the actual data directly into response
        }
        
    except Exception as e:
        logger.error(f"Error in keyword generation: {e}", exc_info=True)
        
        # Use enhanced error handling
        error_response = create_error_response(e, create_session())
        
        # Create fallback response
        fallback_data = {
            "strategy": {
                "primary_keyword": request.primary_keyword or request.topic.lower(),
                "secondary_keywords": [],
                "long_tail_keywords": []
            },
            "topic": request.topic,
            "generation_date": datetime.now().isoformat(),
            "error_info": error_response["error"]
        }
        
        return {
            "session_id": error_response["session_id"],
            "status": "completed_with_fallback",
            "message": "Keyword strategy completed with fallback data due to processing error",
            **fallback_data
        }

@app.post("/titles/generate")
async def generate_titles(request: TitleRequest):
    """Generate title options with structured output"""
    try:
        session_id = create_session()
        
        inputs = {
            "topic": request.topic,
            "primary_keyword": request.primary,
            "secondary_keywords": request.secondary,
            "pillar": request.topic,
            "research_method": request.research_method
        }
        
        logger.info(f"Starting title generation for topic: {request.topic}")
        
        # Get direct output from crew (now returns clean extracted result)
        result_data = blog_crew.run_title_generation(inputs)
        
        update_session(session_id, "title_generation", result_data)
        
        titles_count = len(result_data.get("title_options", []))
        logger.info(f"Generated {titles_count} title options")
        
        # Return the actual data directly in the response
        return {
            "session_id": session_id,
            "status": "completed",
            "message": f"Generated {titles_count} title options",
            **result_data  # Merge the actual data directly into response
        }
        
    except Exception as e:
        logger.error(f"Error in title generation: {e}", exc_info=True)
        
        # Use enhanced error handling
        error_response = create_error_response(e, create_session())
        
        # Create fallback response
        fallback_data = {
            "title_options": [],
            "topic": request.topic,
            "generation_date": datetime.now().isoformat(),
            "error_info": error_response["error"]
        }
        
        return {
            "session_id": error_response["session_id"],
            "status": "completed_with_fallback",
            "message": "Title generation completed with fallback data due to processing error",
            **fallback_data
        }

@app.post("/structure/create")
async def create_structure(request: StructureRequest):
    """Create content structure with structured output"""
    try:
        session_id = create_session()
        
        inputs = {
            "topic": request.topic,
            "structure_type": request.structure_type,
            "keywords": request.keywords,
            "primary_keyword": request.primary_keyword,
            "pillar": request.topic,
            "research_method": request.research_method
        }
        
        logger.info(f"Creating structure for topic: {request.topic}, type: {request.structure_type}")
        
        # Get direct output from crew (now returns clean extracted result)
        result_data = blog_crew.run_content_structure(inputs)
        
        update_session(session_id, "content_structure", result_data)
        
        sections_count = len(result_data.get("sections", []))
        logger.info(f"Content structure created with {sections_count} sections")
        
        # Return the actual data directly in the response
        return {
            "session_id": session_id,
            "status": "completed",
            "message": f"Content structure created with {sections_count} sections",
            **result_data  # Merge the actual data directly into response
        }
        
    except Exception as e:
        logger.error(f"Error in structure creation: {e}", exc_info=True)
        
        # Use enhanced error handling
        error_response = create_error_response(e, create_session())
        
        # Create fallback response
        fallback_data = {
            "sections": [],
            "topic": request.topic,
            "structure_type": request.structure_type,
            "generation_date": datetime.now().isoformat(),
            "error_info": error_response["error"]
        }
        
        return {
            "session_id": error_response["session_id"],
            "status": "completed_with_fallback",
            "message": "Content structure completed with fallback data due to processing error",
            **fallback_data
        }

@app.post("/blog/generate")
async def generate_blog(request: BlogGenerationRequest):
    """Generate full blog post with structured output"""
    try:
        session_id = create_session()
        
        inputs = {
            "topic": request.topic,
            "structure_type": request.structure_type,
            "primary_keyword": request.primary_keyword,
            "keywords": request.keywords,
            "brand_voice": request.brand_voice,
            "pillar": request.topic,
            "research_method": request.research_method
        }
        
        logger.info("Starting blog generation")
        
        # Get direct output from crew (now returns clean extracted result)
        result_data = blog_crew.run_blog_writing(inputs)
        
        update_session(session_id, "blog_generation", result_data)
        
        # Calculate word count more reliably
        word_count = 0
        if isinstance(result_data, dict):
            if 'blog_content' in result_data and result_data['blog_content']:
                # Strip HTML tags for more accurate word count
                import re
                clean_content = re.sub(r'<[^>]+>', '', result_data['blog_content'])
                word_count = len(clean_content.split())
                
                # Update metadata with correct word count
                if 'metadata' not in result_data:
                    result_data['metadata'] = {}
                result_data['metadata']['word_count'] = word_count
                
                logger.info(f"Blog generated successfully: {word_count} words")
        
        if word_count == 0:
            logger.warning("Blog generation completed but word count is 0 - possible parsing issue")
        
        # Return the actual data directly in the response
        return {
            "session_id": session_id,
            "status": "completed",
            "message": f"Blog generated successfully with {word_count} words",
            **result_data  # Merge the actual data directly into response
        }
        
    except Exception as e:
        logger.error(f"Error in blog generation: {e}", exc_info=True)
        
        # Use enhanced error handling
        error_response = create_error_response(e, create_session())
        
        # Create fallback response
        fallback_data = {
            "blog_content": "",
            "metadata": {
                "word_count": 0,
                "topic": request.topic
            },
            "generation_date": datetime.now().isoformat(),
            "error_info": error_response["error"]
        }
        
        return {
            "session_id": error_response["session_id"],
            "status": "completed_with_fallback",
            "message": "Blog generation completed with fallback data due to processing error",
            **fallback_data
        }
    
@app.post("/workflow/run")
async def run_complete_workflow(request: WorkflowRequest):
    """Run complete workflow from topic to blog with structured output"""
    try:
        session_id = create_session()
        
        inputs = {
            "session_id": session_id,
            "topic": request.topic,
            "pillar": request.pillar,
            "research_method": request.mode,
            "structure_type": request.structure,
            "brand_voice": request.brand_voice
        }
        
        if request.urls:
            inputs["urls"] = request.urls
        if request.uploads:
            inputs["uploads"] = request.uploads
        
        logger.info(f"Starting complete workflow for topic: {request.topic}")
        
        # Get direct output from crew (now returns clean extracted result)
        result_data = blog_crew.run_workflow(inputs)
        
        update_session(session_id, "complete_workflow", result_data)
        
        logger.info(f"Complete workflow executed successfully")
        
        # Return the actual data directly in the response
        return {
            "session_id": session_id,
            "status": result_data.get("final_status", "completed"),
            "message": "Complete workflow executed successfully",
            "execution_time": result_data.get("total_execution_time", "unknown"),
            **result_data  # Merge the actual data directly into response
        }
        
    except Exception as e:
        logger.error(f"Error in complete workflow: {e}", exc_info=True)
        
        # Use enhanced error handling
        error_response = create_error_response(e, session_id)
        
        # Create fallback response
        fallback_data = {
            "topic": request.topic,
            "pillar": request.pillar,
            "final_status": "failed",
            "total_execution_time": "unknown",
            "success_metrics": {"workflow_completed": False, "error_info": error_response["error"]},
            "error_info": error_response["error"]
        }
        
        return {
            "session_id": session_id,
            "status": "completed_with_fallback",
            "message": "Complete workflow completed with fallback data due to processing error",
            **fallback_data
        }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload file for research"""
    try:
        # Validate file type
        allowed_extensions = {'.pdf', '.md', '.txt', '.docx'}
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"File type {file_extension} not supported. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Save file to temp directory
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file.filename}")
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"File uploaded successfully: {file.filename}")
        
        return {
            "filename": file.filename,
            "file_path": file_path,
            "size": len(content),
            "content_type": file.content_type,
            "extension": file_extension,
            "status": "uploaded"
        }
        
    except Exception as e:
        logger.error(f"Error in file upload: {e}")
        
        # Use enhanced error handling
        error_response = create_error_response(e)
        
        # Return structured error response
        return JSONResponse(
            status_code=500,
            content=error_response
        )


@app.get("/sessions/{session_id}")
async def get_session_status(session_id: str):
    """Get session status and completed steps"""
    try:
        session = get_session(session_id)
        
        # Get detailed results for each completed step
        results = {}
        for step in session.get("steps_completed", []):
            key = f"{session_id}_{step}"
            if key in results_storage:
                data = results_storage[key]
                if isinstance(data, dict):
                    # Provide summary of structured data
                    results[step] = {
                        "type": "structured_output",
                        "keys": list(data.keys()),
                        "data_preview": {k: v for k, v in list(data.items())[:3]}  # First 3 items
                    }
                else:
                    results[step] = {"content_length": len(str(data)), "type": "string"}
        
        return {
            "session_id": session_id,
            "status": session["status"],
            "created_at": session["created_at"],
            "last_updated": session.get("last_updated"),
            "steps_completed": session["steps_completed"],
            "results_summary": results
        }
        
    except Exception as e:
        logger.error(f"Error getting session status: {e}")
        
        # Use enhanced error handling
        error_response = create_error_response(e)
        
        # Return structured error response
        return JSONResponse(
            status_code=500,
            content=error_response
        )

@app.get("/sessions/{session_id}/step/{step_name}")
async def get_step_output(session_id: str, step_name: str):
    """Get detailed output for a specific workflow step"""
    try:
        session = get_session(session_id)
        
        if step_name not in session.get("steps_completed", []):
            raise HTTPException(status_code=404, detail=f"Step '{step_name}' not found or not completed")
        
        key = f"{session_id}_{step_name}"
        if key not in results_storage:
            raise HTTPException(status_code=404, detail=f"Output for step '{step_name}' not found")
        
        step_data = results_storage[key]
        
        # Return the actual step data directly
        return {
            "session_id": session_id,
            "step_name": step_name,
            "status": "completed",
            "retrieved_at": datetime.now().isoformat(),
            **step_data  # Merge the actual step data directly into response
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting step output: {e}")
        
        # Use enhanced error handling
        error_response = create_error_response(e)
        
        # Return structured error response
        return JSONResponse(
            status_code=500,
            content=error_response
        )

@app.get("/output/files")
async def list_output_files():
    """List all JSON output files created by the system"""
    try:
        output_dir = Path('output')
        if not output_dir.exists():
            return {"files": [], "message": "No output directory found"}
        
        json_files = list(output_dir.glob('*.json'))
        file_info = []
        
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    file_info.append({
                        "filename": file_path.name,
                        "size": file_path.stat().st_size,
                        "created": datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
                        "type": type(data).__name__,
                        "preview": str(data)[:200] + "..." if len(str(data)) > 200 else str(data)
                    })
            except Exception as e:
                file_info.append({
                    "filename": file_path.name,
                    "size": file_path.stat().st_size,
                    "created": datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
                    "error": f"Could not read file: {e}"
                })
        
        return {
            "files": file_info,
            "total_files": len(json_files),
            "output_directory": str(output_dir.absolute())
        }
        
    except Exception as e:
        logger.error(f"Error listing output files: {e}")
        
        # Use enhanced error handling
        error_response = create_error_response(e)
        
        # Return structured error response
        return JSONResponse(
            status_code=500,
            content=error_response
        )

@app.get("/output/download/{filename}")
async def download_output_file(filename: str):
    """Download a specific JSON output file"""
    try:
        output_dir = Path('output')
        file_path = output_dir / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        if not filename.endswith('.json'):
            raise HTTPException(status_code=400, detail="Only JSON files can be downloaded")
        
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type='application/json'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        
        # Use enhanced error handling
        error_response = create_error_response(e)
        
        # Return structured error response
        return JSONResponse(
            status_code=500,
            content=error_response
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8085)