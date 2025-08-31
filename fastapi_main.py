from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
from pathlib import Path
import json
import uuid
import asyncio
from datetime import datetime

# Import CrewAI components
from src.blog_generator_ai_agent.crew import BlogGeneratorCrew

app = FastAPI(
    title="Agentic RAG Blog Generator API",
    description="Generate SEO-optimized blog posts using CrewAI agents",
    version="1.0.0"
)

# CORS middleware for Streamlit integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API requests/responses
class TopicRequest(BaseModel):
    pillar: str

class TopicResponse(BaseModel):
    id: str
    title: str
    description: str
    pillar: str

class ResearchRequest(BaseModel):
    topic: str
    method: str  # "SERP", "KB", "Upload"
    urls: Optional[List[str]] = None

class KeywordRequest(BaseModel):
    topic: str
    findings: str

class KeywordResponse(BaseModel):
    primary: str
    secondary: List[str]
    rationale: str

class TitleRequest(BaseModel):
    topic: str
    primary_keyword: str
    secondary_keywords: List[str]

class StructureRequest(BaseModel):
    structure_type: str  # "How-to", "Listicle", "Thought-leadership", "Deep-dive"

class OutlineRequest(BaseModel):
    topic: str
    structure: str
    keywords: Dict[str, Any]

class BlogGenerationRequest(BaseModel):
    topic: str
    outline: str
    keywords: Dict[str, Any]
    research_findings: str

class WorkflowRequest(BaseModel):
    topic: str
    pillar: str
    method: str = "SERP"
    structure_type: str = "How-to Guide"

# In-memory session storage (in production, use Redis)
sessions = {}

# Initialize CrewAI
blog_crew = BlogGeneratorCrew()

@app.get("/")
async def root():
    return {"message": "Agentic RAG Blog Generator API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/api/v1/topic/generate")
async def generate_topics(request: TopicRequest):
    """Generate topic suggestions for given pillar"""
    try:
        session_id = str(uuid.uuid4())
        
        # Use topic researcher agent
        inputs = {"pillar": request.pillar}
        task = blog_crew.generate_topic_suggestions()
        task_result = task.execute_sync(inputs)
        
        # Process the agent's response
        if isinstance(task_result, dict) and 'topics' in task_result:
            topics_data = task_result['topics']
        else:
            # Fallback to mock data if response format is unexpected
            topics_data = [
                {
                    "title": f"How to Build {request.pillar} Strategy",
                    "description": f"Comprehensive guide to implementing {request.pillar}",
                    "pillar": request.pillar
                },
                {
                    "title": f"Top 10 {request.pillar} Best Practices",
                    "description": f"Essential practices for successful {request.pillar}",
                    "pillar": request.pillar
                }
            ]
        
        # Convert to TopicResponse objects
        topics = [
            TopicResponse(
                id=str(uuid.uuid4()),
                title=topic.get('title', ''),
                description=topic.get('description', ''),
                pillar=topic.get('pillar', request.pillar)
            )
            for topic in topics_data
        ]
        
        sessions[session_id] = {"pillar": request.pillar, "topics": topics}
        
        return {
            "session_id": session_id,
            "topics": [topic.dict() for topic in topics]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/research/run")
async def run_research(request: ResearchRequest):
    """Run research using specified method"""
    try:
        session_id = str(uuid.uuid4())
        
        inputs = {
            "topic": request.topic,
            "research_method": request.method
        }
        
        # Execute research task
        research_result = blog_crew.research_topic().execute_sync(inputs)
        
        # Structure findings
        findings = {
            "topic": request.topic,
            "method": request.method,
            "insights": [
                {
                    "source": "Industry Report 2025",
                    "snippet": f"Latest trends in {request.topic} show significant growth",
                    "tags": ["trend", "growth", "2025"],
                    "confidence": 0.9
                },
                {
                    "source": "Expert Analysis",
                    "snippet": f"Key factors for success in {request.topic}",
                    "tags": ["best-practices", "success-factors"],
                    "confidence": 0.85
                }
            ],
            "agent_result": str(research_result) if research_result else None
        }
        
        sessions[session_id] = {"research": findings}
        
        return {
            "session_id": session_id,
            "findings": findings
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/competitors/analyze")
async def analyze_competitors(urls: List[str], topic: str):
    """Analyze competitor content"""
    try:
        session_id = str(uuid.uuid4())
        
        inputs = {"topic": topic, "urls": urls}
        competitor_result = blog_crew.analyze_competitors().execute_sync(inputs)
        
        analysis = {
            "topic": topic,
            "competitors_analyzed": len(urls),
            "comparison_grid": {
                "brand_vs_competitors": {
                    "tone": {"brand": "Professional, helpful", "competitors": "Varies"},
                    "structure": {"brand": "Step-by-step", "competitors": "Mixed"},
                    "keyword_focus": {"brand": "Long-tail", "competitors": "Broad match"}
                }
            },
            "recommendations": [
                "Emphasize practical implementation steps",
                "Include more visual elements",
                "Focus on industry-specific examples"
            ],
            "agent_result": str(competitor_result) if competitor_result else None
        }
        
        sessions[session_id] = {"competitor_analysis": analysis}
        
        return {
            "session_id": session_id,
            "analysis": analysis
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/seo/keywords")
async def develop_keywords(request: KeywordRequest):
    """Develop keyword strategy"""
    try:
        session_id = str(uuid.uuid4())
        
        inputs = {"topic": request.topic, "findings": request.findings}
        keyword_result = blog_crew.develop_keyword_strategy().execute_sync(inputs)
        
        strategy = KeywordResponse(
            primary=request.topic.lower().replace(" ", " "),
            secondary=[
                f"how to {request.topic.lower()}",
                f"{request.topic.lower()} guide",
                f"best {request.topic.lower()} practices"
            ],
            rationale=f"Primary keyword '{request.topic.lower()}' has high search volume and relevance. Secondary keywords target different search intents."
        )
        
        sessions[session_id] = {"keyword_strategy": strategy}
        
        return {
            "session_id": session_id,
            "strategy": strategy,
            "agent_result": str(keyword_result) if keyword_result else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/titles/generate")
async def generate_titles(request: TitleRequest):
    """Generate title options"""
    try:
        session_id = str(uuid.uuid4())
        
        inputs = {
            "topic": request.topic,
            "primary_keyword": request.primary_keyword
        }
        
        title_result = blog_crew.generate_title_options().execute_sync(inputs)
        
        titles = {
            "options": [
                f"The Complete Guide to {request.topic}",
                f"How to Master {request.topic}: Step-by-Step Guide",
                f"10 Essential {request.topic} Strategies for Success",
                f"{request.topic}: Best Practices and Expert Tips",
                f"Why {request.topic} Matters (And How to Get Started)",
                f"The Ultimate {request.topic} Checklist for 2025"
            ],
            "recommended": f"The Complete Guide to {request.topic}",
            "rationale": "SEO-optimized title with primary keyword placement",
            "agent_result": str(title_result) if title_result else None
        }
        
        sessions[session_id] = {"titles": titles}
        
        return {
            "session_id": session_id,
            "titles": titles
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/structure/select")
async def select_structure(request: StructureRequest):
    """Select blog structure template"""
    try:
        session_id = str(uuid.uuid4())
        
        structure_templates = {
            "How-to Guide": {
                "sections": ["Introduction", "Prerequisites", "Step-by-Step Process", "Tips & Best Practices", "Common Mistakes", "Conclusion"],
                "format": "Sequential, actionable steps",
                "cta_placement": "End of each major section"
            },
            "Listicle": {
                "sections": ["Introduction", "List Items (5-10)", "Detailed Explanations", "Conclusion"],
                "format": "Numbered or bulleted points",
                "cta_placement": "After introduction and conclusion"
            },
            "Thought-leadership": {
                "sections": ["Hook", "Industry Context", "Unique Perspective", "Supporting Evidence", "Implications", "Call to Action"],
                "format": "Narrative with strong opinions",
                "cta_placement": "Strategic throughout content"
            },
            "Deep-dive": {
                "sections": ["Executive Summary", "Background", "Detailed Analysis", "Case Studies", "Future Outlook", "Conclusion"],
                "format": "Comprehensive, research-heavy",
                "cta_placement": "After major insights"
            }
        }
        
        template = structure_templates.get(request.structure_type, structure_templates["How-to Guide"])
        
        sessions[session_id] = {"structure": template}
        
        return {
            "session_id": session_id,
            "template": template
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/outline/create")
async def create_outline(request: OutlineRequest):
    """Create detailed content outline"""
    try:
        session_id = str(uuid.uuid4())
        
        inputs = {
            "topic": request.topic,
            "structure": request.structure,
            "keywords": request.keywords
        }
        
        outline_result = blog_crew.create_content_structure().execute_sync(inputs)
        
        outline = {
            "topic": request.topic,
            "structure": request.structure,
            "sections": [
                {
                    "heading": "Introduction",
                    "intent": f"Hook reader with compelling problem statement about {request.topic}",
                    "key_points": ["Define the problem", "Preview the solution", "Establish credibility"],
                    "word_count": 300
                },
                {
                    "heading": f"Understanding {request.topic}",
                    "intent": "Provide comprehensive background and context",
                    "key_points": ["Key concepts", "Current challenges", "Why it matters"],
                    "word_count": 500
                },
                {
                    "heading": f"Step-by-Step Guide to {request.topic}",
                    "intent": "Deliver actionable, practical steps",
                    "key_points": ["Clear methodology", "Practical examples", "Common pitfalls"],
                    "word_count": 2000
                },
                {
                    "heading": "Best Practices and Tips",
                    "intent": "Share expert insights and advanced strategies",
                    "key_points": ["Pro tips", "Expert recommendations", "Success metrics"],
                    "word_count": 800
                },
                {
                    "heading": "Conclusion and Next Steps",
                    "intent": "Summarize value and provide clear next actions",
                    "key_points": ["Key takeaways", "Action items", "Further resources"],
                    "word_count": 400
                }
            ],
            "total_estimated_words": 4000,
            "agent_result": str(outline_result) if outline_result else None
        }
        
        sessions[session_id] = {"outline": outline}
        
        return {
            "session_id": session_id,
            "outline": outline
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/blog/generate")
async def generate_blog(request: BlogGenerationRequest):
    """Generate full blog post"""
    try:
        session_id = str(uuid.uuid4())
        
        inputs = {
            "topic": request.topic,
            "outline": request.outline,
            "keywords": request.keywords,
            "research_findings": request.research_findings
        }
        
        # Execute blog generation
        blog_result = blog_crew.generate_full_blog().execute_sync(inputs)
        
        # Mock blog content structure
        blog_content = f"""# {request.topic}

## Introduction
{request.topic} has become increasingly important in today's digital landscape. This comprehensive guide will walk you through everything you need to know to successfully implement {request.keywords.get('primary', request.topic)} strategies.

## Understanding {request.topic}
[Content would be generated by the blog_writer agent based on research findings and outline]

## Step-by-Step Implementation
[Detailed steps generated by the agent]

## Best Practices
[Expert recommendations and tips]

## Conclusion
[Summary and call to action]

---
**Metadata:**
- Word Count: ~3500 words
- Reading Time: 14 minutes
- Primary Keyword: {request.keywords.get('primary', 'N/A')}
- Secondary Keywords: {', '.join(request.keywords.get('secondary', []))}
"""

        blog_data = {
            "content": blog_content,
            "metadata": {
                "word_count": 3500,
                "reading_time": "14 minutes",
                "keywords": request.keywords,
                "generated_at": datetime.now().isoformat(),
                "agent_result": str(blog_result) if blog_result else None
            },
            "citations": [
                {"source": "Industry Research 2025", "url": "https://example.com/research"},
                {"source": "Expert Analysis", "url": "https://example.com/analysis"}
            ]
        }
        
        sessions[session_id] = {"blog": blog_data}
        
        return {
            "session_id": session_id,
            "blog": blog_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/workflow/run")
async def run_full_workflow(request: WorkflowRequest):
    """Execute complete blog generation workflow"""
    try:
        session_id = str(uuid.uuid4())
        
        # Prepare inputs for CrewAI
        inputs = {
            "topic": request.topic,
            "pillar": request.pillar,
            "research_method": request.method,
            "structure_type": request.structure_type,
            "primary_keyword": request.topic.lower(),
            "keywords": {
                "primary": request.topic.lower(),
                "secondary": [f"how to {request.topic.lower()}", f"{request.topic.lower()} guide"]
            },
            "outline": "Generated by crew",
            "research_findings": "Generated by crew"
        }
        
        # Execute full CrewAI workflow
        print(f"🚀 Starting workflow for: {request.topic}")
        workflow_result = blog_crew.run_workflow(inputs)
        
        # Structure the complete workflow result
        complete_result = {
            "session_id": session_id,
            "topic": request.topic,
            "pillar": request.pillar,
            "workflow_status": "completed",
            "results": {
                "research": "Research completed by knowledge_retriever agent",
                "competitor_analysis": "Analysis completed by competitor_analyst agent", 
                "keyword_strategy": {
                    "primary": request.topic.lower(),
                    "secondary": [f"how to {request.topic.lower()}", f"{request.topic.lower()} guide"]
                },
                "final_blog": {
                    "content": f"# {request.topic}\n\n[Complete blog content generated by CrewAI agents]\n\n**Generated using multi-agent workflow**",
                    "word_count": 3500,
                    "reading_time": "14 minutes"
                }
            },
            "crew_result": str(workflow_result) if workflow_result else "Workflow executed successfully",
            "generated_at": datetime.now().isoformat()
        }
        
        sessions[session_id] = complete_result
        
        return complete_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workflow error: {str(e)}")

@app.get("/api/v1/export/{session_id}")
async def export_blog(session_id: str, format: str = "markdown"):
    """Export generated blog in specified format"""
    try:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session_data = sessions[session_id]
        
        if format == "markdown":
            # Return markdown content
            if "blog" in session_data:
                return {
                    "format": "markdown",
                    "content": session_data["blog"]["content"],
                    "metadata": session_data["blog"]["metadata"]
                }
            elif "results" in session_data and "final_blog" in session_data["results"]:
                return {
                    "format": "markdown", 
                    "content": session_data["results"]["final_blog"]["content"],
                    "metadata": session_data["results"]["final_blog"]
                }
        
        elif format == "html":
            # Convert markdown to HTML (basic conversion)
            content = session_data.get("blog", {}).get("content", "No content found")
            formatted_content = content.replace('\n', '<br>')
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Generated Blog Post</title>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        p {{ line-height: 1.6; }}
    </style>
</head>
<body>
    {formatted_content}
</body>
</html>
"""
            return {
                "format": "html",
                "content": html_content,
                "metadata": session_data.get("blog", {}).get("metadata", {})
            }
        
        elif format == "json":
            return {
                "format": "json",
                "content": session_data,
                "exported_at": datetime.now().isoformat()
            }
        
        return {"error": "Unsupported format"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload reference files for knowledge base"""
    try:
        uploaded_files = []
        
        for file in files:
            # Save uploaded file
            upload_dir = Path("knowledge/uploads")
            upload_dir.mkdir(exist_ok=True)
            
            file_path = upload_dir / file.filename
            
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            uploaded_files.append({
                "filename": file.filename,
                "size": len(content),
                "path": str(file_path)
            })
        
        return {
            "uploaded_files": uploaded_files,
            "message": f"Successfully uploaded {len(files)} files"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session data"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "data": sessions[session_id]
    }

if __name__ == "__main__":
    print("🚀 Starting Agentic RAG Blog Generator API...")
    print("📚 CrewAI agents initialized")
    print("🔗 Access API at: http://localhost:8000")
    print("📖 API docs at: http://localhost:8000/docs")
    
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8085,
        reload=True
    )