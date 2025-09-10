from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List
import uuid
import json
import os
import tempfile
import re
from datetime import datetime
from pathlib import Path
import logging
from src.blog_generator_ai_agent.utils.constants import API_HOST,API_PORT, LAYOUT_TEMPLATES
from src.blog_generator_ai_agent.utils.blog_generation_utils import build_orchestration_context_from_session_data
from src.blog_generator_ai_agent.api.models import (
    TopicRequest,
    ResearchRequest,
    CompetitorAnalysisRequest,
    KeywordRequest,
    TitleRequest,
    StructureRequest,
    OutlineRequest,
    BlogGenerationRequest,
    WorkflowRequest,
    SessionResponse,
)
from src.blog_generator_ai_agent.api.exceptions import create_error_response, global_exception_handler
from src.blog_generator_ai_agent.utils.utils import create_session, get_session, update_session, results_storage, save_json_output, load_json_output

from dotenv import load_dotenv
from src.blog_generator_ai_agent.logger import get_logger, LOGS_DIR 
from src.blog_generator_ai_agent.utils.setup_telemetry import setup_telemetry, instrument_fastapi_app, log_with_custom_dimensions
from src.blog_generator_ai_agent.utils.constants import LLM_MODEL
from langchain_google_genai import ChatGoogleGenerativeAI

from src.blog_generator_ai_agent.utils.blog_generation_utils import (
    get_generate_topics,
    get_run_research,
    get_analyze_competitors,
    get_generate_keywords,
    get_generate_titles,
    get_create_structure,
    get_generate_blog,
    get_run_complete_workflow
)

load_dotenv()

app = FastAPI(
    title="Blog Generator AI Agent",
    description="Agentic RAG-enabled blog generation system with structured outputs",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_exception_handler(Exception, global_exception_handler)

logger = get_logger("Blog-generator-fastapi-azure-insights")
logger.info(f"Logger initialized. Logs directory: {LOGS_DIR}")

# Initialize the gemini model
google_api_key = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(
            model= "gemini-1.5-flash",
            google_api_key = google_api_key,
            temperature=0.7,
            max_tokens=8192
        )

session_id = create_session()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        logger.debug("Health check requested")
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}"
        )
    
@app.get("/test/crew")
async def test_crew():
    """Test endpoint to check crew functionality"""
    try:
        logger.info("/test/crew called")
        test_inputs = {
            "topic": "test topic",
            "competitor_urls": ["https://example.com"],
            "pillar": "test"
        }
        
        logger.info("Testing crew functionality...")
        
        return {
            "status": "healthy",
            "crew_available": False, 
            "crew_methods": [],
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
        
        logger.info(f"/topic/generate started | session_id={session_id} | pillar={request.pillar}")
        
        inputs = {
            "pillar": request.pillar,
        }
        
        logger.debug(f"Topic generation inputs: {inputs}")
        
        response = get_generate_topics(request.pillar, llm=llm)

        response_dict = response.model_dump()

        response_dict.update({
            "pillar": request.pillar,
            "generation_date": datetime.now().strftime("%Y-%m-%d"),
            "total_suggestions": len(response.topics)
        })
        
        update_session(session_id, "topic_generation", response_dict)
        save_json_output(session_id, "topic_generation", response_dict)
        
        topics_count = len(response.topics)
        logger.info(f"/topic/generate completed | session_id={session_id} | topics_count={topics_count}")
      
        return response_dict
        
    except Exception as e:
        logger.error(f"Error in topic generation: {e}", exc_info=True)
                
        error_response = create_error_response(e, session_id)
        
        fallback_data = {
            "topics": [],
            "pillar": request.pillar,
            "generation_date": datetime.now().strftime("%Y-%m-%d"),
            "total_suggestions": 0,
            "error_info": error_response["error"]
        }
        
        save_json_output(session_id, "topic_generation", fallback_data)
        return fallback_data

@app.post("/research/run", status_code=200)
async def run_research(request: ResearchRequest):
    """Run research using specified mode with structured output"""
    
    
    try:
        logger.info(f"/research/run started | session_id={session_id} | topic={request.topic} | mode={request.mode}")
        
        response = get_run_research(
            topic=request.topic.strip(),
            mode=request.mode.upper(),
            urls=[str(url) for url in request.urls if url],
            uploads=[upload for upload in request.uploads if upload],
            llm=llm
        )
             
        response_dict = response.model_dump()
            
        response_dict.update({
            "session_id": session_id,
            "status": "completed",
            "message": f"Research completed with {len(response.findings)} findings"
        })
           
        update_session(session_id, "research", response_dict)
        save_json_output(session_id, "research", response_dict)
        
        findings_count = len(response.findings)
        logger.info(f"/research/run completed | session_id={session_id} | findings_count={findings_count}")
            
        return response_dict
        
    except Exception as e:
        error_msg = f"Error during research: {str(e)}"
        logger.error(error_msg, exc_info=True)
             
        error_response = create_error_response(e, session_id)
          
        fallback_data = {
            "session_id": session_id,
            "status": "completed_with_fallback",
            "message": "Research completed with fallback data due to processing error",
            "topic": request.topic,
            "research_method": request.mode,
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
        save_json_output(session_id, "research", fallback_data)
        
        return fallback_data

@app.post("/competitors/analyse")
async def analyze_competitors(request: CompetitorAnalysisRequest):
    """Analyze competitor content with structured output"""
    try:
        
        
        logger.info(f"/competitors/analyse started | session_id={session_id} | urls_count={len(request.urls)}")
        
        response = get_analyze_competitors(
            topic=request.topic,
            competitor_urls=request.urls,
            llm=llm
        )
         
        response_dict = response.model_dump()
 
        response_dict.update({
            "session_id": session_id,
            "status": "completed", 
            "message": f"Analyzed {len(response.competitor_summaries)} competitors successfully"
        })
        
        
        update_session(session_id, "competitor_analysis", response_dict)
        save_json_output(session_id, "competitor_analysis", response_dict)
        
        competitors_count = len(response.competitor_summaries)
        logger.info(f"/competitors/analyse completed | session_id={session_id} | analyzed_count={competitors_count}")
        
        return response_dict
        
    except Exception as e:
        logger.error(f"Error in competitor analysis: {e}", exc_info=True)
        
        error_response = create_error_response(e, create_session())
        
        fallback_data = {
            "session_id": error_response["session_id"],
            "status": "completed_with_fallback",
            "message": "Competitor analysis completed with fallback data due to processing error",
            "topic": request.topic,
            "competitor_summaries": [],
            "analysis_date": datetime.now().isoformat(),
            "error_info": error_response["error"]
        }
        
        save_json_output(error_response["session_id"], "competitor_analysis", fallback_data)
        return fallback_data

@app.post("/seo/keywords")
async def generate_keywords(request: KeywordRequest):
    """Generate SEO keywords with structured output"""
    try:
        
        
        primary_keyword = request.primary_keyword or request.topic.replace(" ", "-").lower()
        
        logger.info(f"/seo/keywords started | session_id={session_id} | topic={request.topic}")
        
        
        response = get_generate_keywords(
            topic=request.topic,
            primary_keyword=primary_keyword,
            findings=request.findings,
            competitor_urls=request.competitor_urls,
            llm=llm
        )
        
        
        response_dict = response.model_dump()
        
        
        response_dict.update({
            "session_id": session_id,
            "status": "completed",
            "message": f"Keyword strategy generated with primary: '{response.strategy.primary_keyword}'"
        })
        
        
        update_session(session_id, "keyword_strategy", response_dict)
        save_json_output(session_id, "keyword_strategy", response_dict)
        
        primary_kw = response.strategy.primary_keyword
        logger.info(f"/seo/keywords completed | session_id={session_id} | primary='{primary_kw}'")
        
        
        return response_dict
        
    except Exception as e:
        logger.error(f"Error in keyword generation: {e}", exc_info=True)
        
        
        error_response = create_error_response(e, create_session())
        
        
        fallback_data = {
            "session_id": error_response["session_id"],
            "status": "completed_with_fallback",
            "message": "Keyword strategy completed with fallback data due to processing error",
            "strategy": {
                "primary_keyword": request.primary_keyword or request.topic.lower(),
                "secondary_keywords": [],
                "long_tail_keywords": []
            },
            "topic": request.topic,
            "generation_date": datetime.now().isoformat(),
            "error_info": error_response["error"]
        }
        
        save_json_output(error_response["session_id"], "keyword_strategy", fallback_data)
        return fallback_data

@app.post("/titles/generate")
async def generate_titles(request: TitleRequest):
    """Generate title options with structured output"""
    try:
        
        
        logger.info(f"/titles/generate started | session_id={session_id} | topic={request.topic}")
        
        
        response = get_generate_titles(
            topic=request.topic,
            primary_keyword=request.primary,
            secondary_keywords=request.secondary,
            competitor_urls=request.competitor_urls,
            llm=llm
        )
        
        
        response_dict = response.model_dump()
        
        
        response_dict.update({
            "session_id": session_id,
            "status": "completed",
            "message": f"Generated {len(response.title_options)} title options"
        })
        
        
        update_session(session_id, "title_generation", response_dict)
        save_json_output(session_id, "title_generation", response_dict)
        
        titles_count = len(response.title_options)
        logger.info(f"/titles/generate completed | session_id={session_id} | titles_count={titles_count}")
        
        
        return response_dict
        
    except Exception as e:
        logger.error(f"Error in title generation: {e}", exc_info=True)
        
        
        error_response = create_error_response(e, create_session())
        
        
        fallback_data = {
            "session_id": error_response["session_id"],
            "status": "completed_with_fallback",
            "message": "Title generation completed with fallback data due to processing error",
            "title_options": [],
            "topic": request.topic,
            "generation_date": datetime.now().isoformat(),
            "error_info": error_response["error"]
        }
        
        save_json_output(error_response["session_id"], "title_generation", fallback_data)
        return fallback_data

@app.post("/structure/select")
async def select_structure(request: StructureRequest):
    structure_type = request.type
    if structure_type not in LAYOUT_TEMPLATES:
        raise HTTPException(status_code=400, detail="Invalid structure type")
    return {"type": structure_type, "layout": LAYOUT_TEMPLATES[structure_type]}

@app.post("/outline/create")
async def create_structure(request: OutlineRequest):
    """Create content structure with structured output"""
    try:
        
        
        logger.info(f"/outline/create started | session_id={session_id} | topic={request.topic} | type={request.structure_type}")
        
        
        response = get_create_structure(
            topic=request.topic,
            structure_type=request.structure_type,
            keywords=request.keywords,
            primary_keyword=request.primary_keyword,
            competitor_urls=request.competitor_urls,
            llm=llm
        )
        
        
        response_dict = response.model_dump()
        
        
        response_dict.update({
            "session_id": session_id,
            "status": "completed",
            "message": f"Content structure created with {len(response.sections)} sections"
        })
        
        
        update_session(session_id, "content_structure", response_dict)
        save_json_output(session_id, "content_structure", response_dict)
        
        sections_count = len(response.sections)
        logger.info(f"/outline/create completed | session_id={session_id} | sections_count={sections_count}")
        
        
        return response_dict
        
    except Exception as e:
        logger.error(f"Error in structure creation: {e}", exc_info=True)
        
        
        error_response = create_error_response(e, create_session())
        
        
        fallback_data = {
            "session_id": error_response["session_id"],
            "status": "completed_with_fallback",
            "message": "Content structure completed with fallback data due to processing error",
            "sections": [],
            "topic": request.topic,
            "structure_type": request.structure_type,
            "generation_date": datetime.now().isoformat(),
            "error_info": error_response["error"]
        }
        
        save_json_output(error_response["session_id"], "content_structure", fallback_data)
        return fallback_data

@app.post("/blog/generate")
async def generate_blog(request: BlogGenerationRequest):
    """Generate full blog post with structured output using session context"""
    try:
        logger.info(f"/blog/generate started | session_id={session_id} | topic={request.topic} | structure={request.structure_type}")
        
        # Get session data to build orchestration context
        session = get_session(session_id)
        session_data = {}
        
        # Collect data from previous workflow steps
        for step in ["research", "competitor_analysis", "keywords_strategy", "content_structure", "title_generation"]:
            if step in session.get("steps_completed", []):
                key = f"{session_id}_{step}"
                if key in results_storage:
                    session_data[step] = results_storage[key]
        
       
        orchestration_context = build_orchestration_context_from_session_data(session_data)
        
        # Get selected title from title generation if available
        selected_title = ""
        if "title_generation" in session_data:
            title_options = session_data["title_generation"].get("title_options", [])
            if title_options:
                # Auto-select best title based on SEO score
                best_title = title_options[0]
                best_score = 0
                for option in title_options:
                    score = option.get("seo_score", 0)
                    if isinstance(score, (int, float)) and score > best_score:
                        best_score = score
                        best_title = option
                selected_title = best_title.get("title", "")
        
        # Get content structure if available
        content_structure_str = ""
        if "content_structure" in session_data:
            content_structure_str = str(session_data["content_structure"])
        
        # Get research findings if available
        research_findings_str = ""
        if "research" in session_data:
            research_findings_str = str(session_data["research"])
        
        # Get keyword strategy if available
        keyword_strategy_str = ""
        if "keywords_strategy" in session_data:
            keyword_strategy_str = str(session_data["keywords_strategy"])
        
        logger.info(f"Using orchestration context length: {len(orchestration_context)} characters")
        logger.info(f"Selected title: {selected_title}")
        
        response = get_generate_blog(
            topic=request.topic,
            structure_type=request.structure_type,
            primary_keyword=request.primary_keyword,
            keywords=request.keywords,
            brand_voice=request.brand_voice,
            selected_title=selected_title,
            orchestration_context=orchestration_context,
            content_structure=content_structure_str,
            research_findings=research_findings_str,
            keyword_strategy=keyword_strategy_str,
            competitor_urls=request.competitor_urls,
            llm=llm
        )
        
        
        response_dict = response.model_dump()
        
        # Calculate word count more reliably
        word_count = 0
        if response.blog_content:
            # Strip HTML tags for more accurate word count
            clean_content = re.sub(r'<[^>]+>', '', response.blog_content)
            word_count = len(clean_content.split())
               
            if response.metadata:
                response.metadata.word_count = word_count
        
        response_dict.update({
            "session_id": session_id,
            "status": "completed",
            "message": f"Blog generated successfully with {word_count} words"
        })
           
        update_session(session_id, "blog_generation", response_dict)
        save_json_output(session_id, "blog_generation", response_dict)
        
        logger.info(f"/blog/generate completed | session_id={session_id} | word_count={word_count}")
        
        return response_dict
        
    except Exception as e:
        logger.error(f"Error in blog generation: {e}", exc_info=True)
            
        error_response = create_error_response(e, create_session())
        
        fallback_data = {
            "session_id": error_response["session_id"],
            "status": "completed_with_fallback",
            "message": "Blog generation completed with fallback data due to processing error",
            "blog_content": "",
            "metadata": {
                "word_count": 0,
                "topic": request.topic
            },
            "generation_date": datetime.now().isoformat(),
            "error_info": error_response["error"]
        }
        
        save_json_output(error_response["session_id"], "blog_generation", fallback_data)
        return fallback_data
    
@app.post("/workflow/run")
async def run_complete_workflow(request: WorkflowRequest):
    """Run complete workflow from topic to blog with structured output"""
    try:
        
        
        inputs = {
            "session_id": session_id,
            "topic": request.topic,
            "pillar": request.pillar,
            "research_method": request.mode,
            "structure_type": request.structure,
            "brand_voice": request.brand_voice,
            "competitor_urls": request.competitor_urls
        }
        
        if request.urls:
            inputs["urls"] = request.urls
        if request.uploads:
            inputs["uploads"] = request.uploads
        
        logger.info(f"/workflow/run started | session_id={session_id} | topic={request.topic} | mode={request.mode}")
        
       
        result_data = get_run_complete_workflow(
            topic=request.topic,
            pillar=request.pillar,
            mode=request.mode,
            structure_type=request.structure,
            brand_voice=request.brand_voice,
            competitor_urls=request.competitor_urls,
            urls=request.urls,
            uploads=request.uploads,
            llm=llm
        )
        
        update_session(session_id, "complete_workflow", result_data)
        
        logger.info(f"/workflow/run completed | session_id={session_id} | final_status={result_data.get('final_status', 'completed')} | total_time={result_data.get('total_execution_time', 'unknown')}")
        
        
        response_payload = {
            "session_id": session_id,
            "status": result_data.get("final_status", "completed"),
            "message": "Complete workflow executed successfully",
            "execution_time": result_data.get("total_execution_time", "unknown"),
            **result_data
        }
        save_json_output(session_id, "complete_workflow", response_payload)
        return response_payload
        
    except Exception as e:
        logger.error(f"Error in complete workflow: {e}", exc_info=True)
        
        
        error_response = create_error_response(e, session_id)
        
        
        fallback_data = {
            "topic": request.topic,
            "pillar": request.pillar,
            "final_status": "failed",
            "total_execution_time": "unknown",
            "success_metrics": {"workflow_completed": False, "error_info": error_response["error"]},
            "error_info": error_response["error"]
        }
        
        response_payload = {
            "session_id": session_id,
            "status": "completed_with_fallback",
            "message": "Complete workflow completed with fallback data due to processing error",
            **fallback_data
        }
        save_json_output(session_id, "complete_workflow", response_payload)
        return response_payload

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
        
        logger.info(f"/upload completed | filename={file.filename} | size={len(content)} | saved_to={file_path}")
        
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
        
        
        error_response = create_error_response(e)
        
        # Return structured error response
        return JSONResponse(
            status_code=500,
            content=error_response
        )

@app.get("/export/{session_id}")
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
        
        
        error_response = create_error_response(e)
        
        return JSONResponse(
            status_code=500,
            content=error_response
        )
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)