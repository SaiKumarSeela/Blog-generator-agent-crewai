from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import uuid
import json
import os
import tempfile
import re
from datetime import datetime
from pathlib import Path
import logging
from src.blog_generator_ai_agent.utils.constants import API_HOST,API_PORT, LAYOUT_TEMPLATES
from src.blog_generator_ai_agent.crew import BlogGeneratorCrew

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
from src.blog_generator_ai_agent.utils.utils import create_session, get_session, update_session, results_storage

from dotenv import load_dotenv
from src.blog_generator_ai_agent.logger import get_logger, LOGS_DIR 
from src.blog_generator_ai_agent.utils.setup_telemetry import setup_telemetry, instrument_fastapi_app, log_with_custom_dimensions


load_dotenv()

APPLICATIONINSIGHTS_CONNECTION_STRING = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
if not APPLICATIONINSIGHTS_CONNECTION_STRING:
    raise ValueError("APPLICATIONINSIGHTS_CONNECTION_STRING environment variable is required")


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

# Register global exception handler
app.add_exception_handler(Exception, global_exception_handler)

# Initialize the blog generator crew
blog_crew = BlogGeneratorCrew()

tracer = setup_telemetry(service_name="Blog-generator-fastapi-azure-insights",connection_string=APPLICATIONINSIGHTS_CONNECTION_STRING )
instrument_fastapi_app(app)

# Configure custom logger
logger = get_logger("Blog-generator-fastapi-azure-insights")
logger.info(f"Logger initialized. Logs directory: {LOGS_DIR}")


# API Endpoints with Structured Outputs
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    with tracer.start_as_current_span("health_check_endpoint") as span:
        try:
            span.set_attribute("Http.Status",200)
            
            logger.debug("Health check requested")
            log_with_custom_dimensions(logger, logging.INFO, "Fastapi run sucessfully",{
                "operation":"health_check",
                "status":200
            })

            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        except Exception as e:
            span.record_exception(e)
            span.set_attribute("error.type", type(e).__name__)
            span.set_attribute("error.message", str(e))
            log_with_custom_dimensions(logger, logging.INFO, "Error during health check",{
               "operation":"health_check",
                "status":500
            })
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
        
        if not blog_crew:
            return {"status": "error", "message": "Blog crew not initialized"}
        
        crew_methods = [method for method in dir(blog_crew) if not method.startswith('_')]
        
        logger.info(f"Crew methods discovered: {len(crew_methods)}")
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
        logger.info(f"/topic/generate started | session_id={session_id} | pillar={request.pillar}")
        
        inputs = {
            "pillar": request.pillar,
            "topic": f"topics for {request.pillar}",
            "research_method": "SERP"
        }
        
        logger.debug(f"Topic generation inputs: {inputs}")
        
        # Get direct output from crew (now returns clean extracted result)
        result_data = blog_crew.run_topic_generation(inputs)
        
        # The result_data should now be the actual JSON content, not wrapped
        update_session(session_id, "topic_generation", result_data)
        
        topics_count = len(result_data.get("topics", []))
        logger.info(f"/topic/generate completed | session_id={session_id} | topics_count={topics_count}")
        
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
        logger.info(f"/research/run started | session_id={session_id} | topic={request.topic} | mode={request.mode}")
        
        inputs = {
            "topic": request.topic.strip(),
            "research_method": request.mode.upper(),  # Use mode as research_method
            "mode": request.mode.upper(),
            "urls": [str(url) for url in request.urls if url],
            "uploads": [upload for upload in request.uploads if upload],
            "pillar": (request.pillar or request.topic).strip()
        }
        
        logger.debug(f"Research inputs: urls_count={len(inputs['urls'])}, uploads_count={len(inputs['uploads'])}")
        
        # Get direct output from crew (now returns clean extracted result)
        result_data = blog_crew.run_research_only(inputs)
        
        update_session(session_id, "research", result_data)
        
        findings_count = len(result_data.get("findings", []))
        logger.info(f"/research/run completed | session_id={session_id} | findings_count={findings_count}")
        
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
        
        logger.info(f"/competitors/analyse started | session_id={session_id} | urls_count={len(request.urls)}")
        
        # Get direct output from crew (now returns clean extracted result)
        result_data = blog_crew.run_competitor_analysis(inputs)
        
        update_session(session_id, "competitor_analysis", result_data)
        
        competitors_count = len(result_data.get("competitor_summaries", []))
        logger.info(f"/competitors/analyse completed | session_id={session_id} | analyzed_count={competitors_count}")
        
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
            "research_method": request.research_method,
            "competitor_urls": request.competitor_urls
        }
        
        logger.info(f"/seo/keywords started | session_id={session_id} | topic={request.topic}")
        
        # Get direct output from crew (now returns clean extracted result)
        result_data = blog_crew.run_keyword_strategy(inputs)
        
        update_session(session_id, "keyword_strategy", result_data)
        
        primary_kw = result_data.get("strategy", {}).get("primary_keyword", primary_keyword)
        logger.info(f"/seo/keywords completed | session_id={session_id} | primary='{primary_kw}'")
        
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
            "research_method": request.research_method,
            "competitor_urls": request.competitor_urls
        }
        
        logger.info(f"/titles/generate started | session_id={session_id} | topic={request.topic}")
        
        # Get direct output from crew (now returns clean extracted result)
        result_data = blog_crew.run_title_generation(inputs)
        
        update_session(session_id, "title_generation", result_data)
        
        titles_count = len(result_data.get("title_options", []))
        logger.info(f"/titles/generate completed | session_id={session_id} | titles_count={titles_count}")
        
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
        session_id = create_session()
        
        inputs = {
            "topic": request.topic,
            "structure_type": request.structure_type,
            "keywords": request.keywords,
            "primary_keyword": request.primary_keyword,
            "pillar": request.topic,
            "research_method": request.research_method,
            "competitor_urls": request.competitor_urls
        }
        
        logger.info(f"/outline/create started | session_id={session_id} | topic={request.topic} | type={request.structure_type}")
        
        # Get direct output from crew (now returns clean extracted result)
        result_data = blog_crew.run_content_structure(inputs)
        
        update_session(session_id, "content_structure", result_data)
        
        sections_count = len(result_data.get("sections", []))
        logger.info(f"/outline/create completed | session_id={session_id} | sections_count={sections_count}")
        
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
            "research_method": request.research_method,
            "competitor_urls": request.competitor_urls
        }
        
        logger.info(f"/blog/generate started | session_id={session_id} | topic={request.topic} | structure={request.structure_type}")
        
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
                
                logger.info(f"/blog/generate completed | session_id={session_id} | word_count={word_count}")
        
        if word_count == 0:
            logger.warning(f"/blog/generate warning | session_id={session_id} | word_count=0 (possible parsing issue)")
        
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
            "brand_voice": request.brand_voice,
            "competitor_urls": request.competitor_urls
        }
        
        if request.urls:
            inputs["urls"] = request.urls
        if request.uploads:
            inputs["uploads"] = request.uploads
        
        logger.info(f"/workflow/run started | session_id={session_id} | topic={request.topic} | mode={request.mode}")
        
        # Get direct output from crew (now returns clean extracted result)
        result_data = blog_crew.run_workflow(inputs)
        
        update_session(session_id, "complete_workflow", result_data)
        
        logger.info(f"/workflow/run completed | session_id={session_id} | final_status={result_data.get('final_status', 'completed')} | total_time={result_data.get('total_execution_time', 'unknown')}")
        
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
        
        # Use enhanced error handling
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
        
        # Use enhanced error handling
        error_response = create_error_response(e)
        
        # Return structured error response
        return JSONResponse(
            status_code=500,
            content=error_response
        )

@app.get("/export/{session_id}/step/{step_name}")
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
        output_dir = Path('Artifacts')
        if not output_dir.exists():
            logger.info("/output/files | Artifacts directory not found")
            return {"files": [], "message": "No output directory found"}
        
        json_files = list(output_dir.glob('*.json'))
        logger.info(f"/output/files | found {len(json_files)} json files in {output_dir}")
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
                logger.warning(f"/output/files | failed to read {file_path}: {e}")
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
        output_dir = Path('Artifacts')
        file_path = output_dir / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        if not filename.endswith('.json'):
            raise HTTPException(status_code=400, detail="Only JSON files can be downloaded")
        
        logger.info(f"/output/download | filename={filename} | path={file_path}")
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
    uvicorn.run(app, host=API_HOST, port=API_PORT)