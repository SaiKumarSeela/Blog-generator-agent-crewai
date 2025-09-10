import json
import re
from typing import Dict, Any, List
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from src.blog_generator_ai_agent.logger import get_logger
from src.blog_generator_ai_agent.tools.custom_tool import (
    WebSearchTool, 
    ResearchModeTool, 
    CompetitorAnalysisTool
)
from src.blog_generator_ai_agent.utils.prompts import (
    topic_generation_prompt,
    research_topic_prompt,
    competitor_analysis_prompt,
    keyword_strategy_prompt,
    title_generation_prompt,
    content_structure_prompt,
    blog_generation_prompt
)
from src.blog_generator_ai_agent.models.pydantic_models import (
    TopicSuggestion, TopicSuggestionsResponse,
    ResearchOutput, ResearchFinding,
    CompetitorAnalysisOutput, CompetitorContent,
    KeywordStrategyOutput, KeywordStrategy,
    TitleGenerationOutput, TitleOption,
    ContentStructureOutput, ContentSection,
    BlogGenerationOutput, BlogMetadata
)
from SeoKeywordResearch import SeoKeywordResearch
import os 

competitor_analysis_tool = CompetitorAnalysisTool()
web_tool = WebSearchTool()
research_mode_tool = ResearchModeTool()


logger = get_logger("blog_generation_utils")

def get_generate_topics(pillar: str, num_results: int = 5, llm: ChatGoogleGenerativeAI = None) -> TopicSuggestionsResponse:

    search_json = web_tool._run(query=f"latest {pillar} trends 2025", num_results=num_results)
    search_data = json.loads(search_json)
    logger.info(f"Search data: {json.dumps(search_data, indent=2)}")

    results = search_data.get("results", [])
    if not results:
        logger.warning("No search results found, falling back to empty context.")
    
    search_context = "\n".join(
        [f"- Title: {r.get('title')}\n  Snippet: {r.get('snippet')}" for r in results]
    )

    logger.info(f"Search Context:\n{search_context}")

    structured_llm = llm.with_structured_output(TopicSuggestionsResponse)

    response: TopicSuggestionsResponse = structured_llm.invoke(
        topic_generation_prompt.format(
            pillar=pillar,
            web_results=search_context  
        )
    )

    for idx, t in enumerate(response.topics, start=1):
        if not t.id:
            t.id = f"topic_{idx}" 

    return response

def get_run_research(topic: str, mode: str, urls: List[str] = None, uploads: List[str] = None, llm: ChatGoogleGenerativeAI = None) -> ResearchOutput:
    """Run research using specified mode with structured output"""
    try:
        research_result = research_mode_tool._run(
            mode=mode.lower(),
            topic=topic,
            num_results=5,
            content_list=[{"type": "url", "content": url} for url in urls] if urls else []
        )
        
        research_data = json.loads(research_result)
        logger.info(f"Research data: {json.dumps(research_data, indent=2)}")
        
        findings = research_data.get("findings", [])
        if not findings:
            logger.warning("No research findings found, falling back to empty context.")
        
        research_context = "\n".join([
            f"- Source: {f.get('source', 'Unknown')}\n  Snippet: {f.get('snippet', 'No snippet')}\n  Confidence: {f.get('confidence', 0)}"
            for f in findings
        ])
        
        logger.info(f"Research Context:\n{research_context}")
        
        structured_llm = llm.with_structured_output(ResearchOutput)
        
        response: ResearchOutput = structured_llm.invoke(
            research_topic_prompt.format(
                topic=topic,
                research_method=mode.upper(),
                research_context=research_context
            )
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Research failed: {e}")

        return ResearchOutput(
            topic=topic,
            research_method=mode.upper(),
            key_insights=[],
            statistics=[],
            findings=[],
            content_gaps=[],
            talking_points=[],
            sources=[]
        )

def get_analyze_competitors(topic: str, competitor_urls: List[str] = None, llm: ChatGoogleGenerativeAI = None) -> CompetitorAnalysisOutput:
    """Analyze competitor content with structured output"""
    try:
   
        competitor_result = competitor_analysis_tool._run(
            topic=topic,
            competitor_urls=competitor_urls
        )
        
        competitor_data = json.loads(competitor_result)
        logger.info(f"Competitor data: {json.dumps(competitor_data, indent=2)}")
        

        competitors = competitor_data.get("competitors", [])
        if not competitors:
            logger.warning("No competitor data found, falling back to empty context.")
        
        competitor_context = "\n".join([
            f"- URL: {c.get('url', 'Unknown')}\n  Domain: {c.get('domain', 'Unknown')}\n  Word Count: {c.get('word_count', 0)}\n  Quality Score: {c.get('content_quality', {}).get('quality_score', 0)}"
            for c in competitors
        ])
        
        logger.info(f"Competitor Context:\n{competitor_context}")
        

        structured_llm = llm.with_structured_output(CompetitorAnalysisOutput)
        
        response: CompetitorAnalysisOutput = structured_llm.invoke(
            competitor_analysis_prompt.format(
                topic=topic,
                competitor_urls=competitor_urls or [],
                competitor_context=competitor_context
            )
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Competitor analysis failed: {e}")
        # Return fallback response
        return CompetitorAnalysisOutput(
            topic=topic,
            competitor_summaries=[],
            common_patterns=[],
            tone_analysis={},
            structure_patterns=[],
            keyword_strategies=[],
            differentiation_opportunities=[],
            what_to_replicate=[],
            what_to_avoid=[]
        )

def get_generate_keywords(topic: str, findings: str = None, primary_keyword: str = None, competitor_urls: List[str] = None, llm: ChatGoogleGenerativeAI = None) -> KeywordStrategyOutput:
    """Generate SEO keywords with structured output"""
    try:

        keyword_research = SeoKeywordResearch(
            query= topic,
            api_key= os.getenv("SERP_API_KEY"),
            lang='en',
            country='us',
            domain='google.com'
        )
       
        structured_llm = llm.with_structured_output(KeywordStrategyOutput)
        primary_keyword = None
        response: KeywordStrategyOutput = structured_llm.invoke(
            keyword_strategy_prompt.format(
                topic=topic,
                primary_keyword=primary_keyword or topic.lower().replace(" ", "-"),
                research_findings=findings or ""
            )
        )
        
        response.serp_results = {
            'auto_complete': keyword_research.get_auto_complete(),
            'related_searches': keyword_research.get_related_searches(),
            'related_questions': keyword_research.get_related_questions(depth_limit=1)
        }
        return response
        
    except Exception as e:
        logger.error(f"Keyword generation failed: {e}")

        return KeywordStrategyOutput(
            topic=topic,
            strategy=KeywordStrategy(
                primary_keyword=primary_keyword or topic.lower().replace(" ", "-"),
                primary_rationale="Fallback keyword selection",
                secondary_keywords=[],
                long_tail_opportunities=[],
                search_intent="informational",
                competition_level="medium",
                recommended_density="2-3%",
                semantic_keywords=[]
            ),
            implementation_notes=[],
            serp_results={
                'auto_complete': [],
                'related_searches': [],
                'related_questions': []
            }
        )

def get_generate_titles(topic: str, primary_keyword: str, secondary_keywords: List[str], competitor_urls: List[str] = None, llm: ChatGoogleGenerativeAI = None) -> TitleGenerationOutput:
    """Generate title options with structured output"""
    try:
        structured_llm = llm.with_structured_output(TitleGenerationOutput)
        
        response: TitleGenerationOutput = structured_llm.invoke(
            title_generation_prompt.format(
                topic=topic,
                primary_keyword=primary_keyword,
                secondary_keywords=", ".join(secondary_keywords),
            )
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Title generation failed: {e}")
  
        return TitleGenerationOutput(
            topic=topic,
            title_options=[],
            recommended_title=f"{topic}: A Comprehensive Guide",
            recommendation_rationale="Fallback title based on topic",
            seo_considerations=[]
        )

def get_create_structure(topic: str, structure_type: str, keywords: Dict[str, Any], primary_keyword: str, competitor_urls: List[str] = None, llm: ChatGoogleGenerativeAI = None) -> ContentStructureOutput:
    """Create content structure with structured output"""
    try:
       
        structured_llm = llm.with_structured_output(ContentStructureOutput)
        
        response: ContentStructureOutput = structured_llm.invoke(
            content_structure_prompt.format(
                topic=topic,
                structure_type=structure_type,
                keywords=str(keywords),
                primary_keyword=primary_keyword,
            )
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Structure creation failed: {e}")

        return ContentStructureOutput(
            topic=topic,
            structure_type=structure_type,
            heading_structure=[],
            sections=[],
            cta_placements=[],
            faq_section=None,
            total_estimated_words=0,
            tone_notes="Professional and informative"
        )

def build_orchestration_context_from_session_data(session_data: Dict[str, Any]) -> str:
    """Build orchestration context from session data for blog generation"""
    oc_parts: List[str] = []
    
    # Research data
    research_data = session_data.get('research', {})
    if research_data:
        if research_data.get('key_insights'):
            oc_parts.append("Research - Key Insights:\n" + "\n".join(f"- {s}" for s in research_data.get('key_insights', [])[:5]))
        if research_data.get('statistics'):
            oc_parts.append("Research - Statistics:\n" + "\n".join(f"- {s}" for s in research_data.get('statistics', [])[:5]))
        if research_data.get('content_gaps'):
            oc_parts.append("Research - Content Gaps:\n" + "\n".join(f"- {s}" for s in research_data.get('content_gaps', [])[:5]))
        if research_data.get('sources'):
            oc_parts.append(
                "Sources (top):\n" + "\n".join(
                    f"- {s.get('title', s.get('url', ''))}: {s.get('url', '')}" for s in research_data.get('sources', [])[:5]
                )
            )
    
    # Competitor analysis data
    competitor_data = session_data.get('competitor_analysis', {})
    if competitor_data:
        if competitor_data.get('differentiation_opportunities'):
            oc_parts.append("Competitor - Differentiation Opportunities:\n" + "\n".join(f"- {s}" for s in competitor_data.get('differentiation_opportunities', [])[:5]))
        if competitor_data.get('common_patterns'):
            oc_parts.append("Competitor - Common Patterns:\n" + "\n".join(f"- {s}" for s in competitor_data.get('common_patterns', [])[:5]))
    
    # Keyword strategy data
    keyword_data = session_data.get('keywords_strategy', {})
    if keyword_data and keyword_data.get('strategy'):
        strategy = keyword_data['strategy']
        secondary_keywords = strategy.get('secondary_keywords', [])
        oc_parts.append(
            "Keyword Strategy:\n" + "\n".join([
                f"- Primary: {strategy.get('primary_keyword', '')}",
                f"- Secondary: {', '.join(secondary_keywords)}",
                f"- Long-tail: {', '.join(strategy.get('long_tail_opportunities', [])[:5])}",
                f"- Intent: {strategy.get('search_intent', '')}",
                f"- Density: {strategy.get('recommended_density', '')}",
            ])
        )
    
    # Content structure data
    structure_data = session_data.get('content_structure', {})
    if structure_data and structure_data.get('heading_structure'):
        oc_parts.append("Outline - Headings:\n" + "\n".join(f"- {h}" for h in structure_data.get('heading_structure', [])[:15]))
    
    return "\n\n".join([p for p in oc_parts if p])

def get_generate_blog(
    topic: str,
    structure_type: str,
    primary_keyword: str,
    secondary_keywords: List[str] = None,
    keywords: Dict[str, Any] = None,
    brand_voice: str = "",
    selected_title: str = "",
    orchestration_context: str = "",
    content_structure: str = "",
    research_findings: str = "",
    keyword_strategy: str = "",
    competitor_urls: List[str] = None,
    llm: ChatGoogleGenerativeAI = None
) -> BlogGenerationOutput:
    """Generate full blog post with structured output using prior step context"""
    try:

        
        structured_llm = llm.with_structured_output(BlogGenerationOutput)
        
        response: BlogGenerationOutput = structured_llm.invoke(
            blog_generation_prompt.format(
                topic=topic,
                structure_type=structure_type,
                primary_keyword=primary_keyword,
                secondary_keywords=", ".join(secondary_keywords or []),
                keywords=str(keywords),
                research_method="SERP",
                brand_voice=brand_voice,
                selected_title=selected_title,
                orchestration_context=orchestration_context,
                content_structure=content_structure,
                research_findings=research_findings,
                keyword_strategy=keyword_strategy,
            )
        )
        
        try:
            if selected_title and isinstance(response.blog_content, str):
                content = response.blog_content or ""
                lines = content.splitlines()
                
                # Check if content already starts with the selected title
                if lines and re.match(rf"^\s*#\s*{re.escape(selected_title)}\s*$", lines[0]):
                    # Title is already correct, don't modify
                    pass
                else:
                    # Remove any existing H1 title and replace with selected title
                    if lines and re.match(r"^\s*#\s+", lines[0]):
                        lines[0] = f"# {selected_title}"
                    else:
                        lines.insert(0, f"# {selected_title}")
                    content = "\n".join(lines)
                
                response.blog_content = content
        except Exception as _:
            # Do not fail workflow for enforcement attempts
            pass

        return response
        
    except Exception as e:
        logger.error(f"Blog generation failed: {e}")
        return BlogGenerationOutput(
            topic=topic,
            blog_content=f"# {topic}\n\nThis is a comprehensive guide about {topic}.",
            metadata=BlogMetadata(
                title=topic,
                meta_description=f"Learn about {topic} in this comprehensive guide.",
                word_count=100,
                reading_time="1 min",
                primary_keyword=primary_keyword,
                secondary_keywords=secondary_keywords or [],
                internal_links=[],
                seo_score=5.0,
                readability_score="Good"
            ),
            content_quality_assessment={},
            brand_voice_alignment="Good",
            sources_used=[],
            optimization_suggestions=[]
        )

def get_run_complete_workflow(
    topic: str, 
    pillar: str, 
    mode: str = "SERP", 
    structure_type: str = "How-to Guide", 
    brand_voice: str = "professional, helpful, concise",
    competitor_urls: List[str] = None,
    urls: List[str] = None,
    uploads: List[str] = None,
    llm: ChatGoogleGenerativeAI = None
) -> Dict[str, Any]:
    """Run complete workflow from topic to blog using LLM and tools"""
    try:
        logger.info(f"Starting complete workflow for topic: {topic}")
        
        # Step 1: Topic suggestions (only when topic not provided)
        topic_suggestions = TopicSuggestionsResponse(topics=[])
        if not topic:
            logger.info("Step 1: Generating topic suggestions...")
            topic_suggestions = get_generate_topics(pillar, llm=llm)
        
        logger.info("Step 2: Running research...")
        research_output = get_run_research(
            topic=topic,
            mode=mode,
            urls=urls,
            uploads=uploads,
            llm=llm
        )

        # Step 3: Competitor analysis (only when competitor URLs provided)
        competitor_analysis = CompetitorAnalysisOutput(
            topic=topic,
            competitor_summaries=[],
            common_patterns=[],
            tone_analysis={},
            structure_patterns=[],
            keyword_strategies=[],
            differentiation_opportunities=[],
            what_to_replicate=[],
            what_to_avoid=[]
        )
        if competitor_urls:
            logger.info("Step 3: Analyzing competitors...")
            competitor_analysis = get_analyze_competitors(
                topic=topic,
                competitor_urls=competitor_urls,
                llm=llm
            )
        else:
            logger.info("Skipping Step 3: Analyze competitors, Competitor URLs not provided")

        logger.info("Step 4: Generating keyword strategy...")
        keyword_strategy = get_generate_keywords(
            topic=topic,
            findings=str(research_output.findings),
            competitor_urls=competitor_urls,
            llm=llm
        )
        

        logger.info("Step 5: Generating title options...")
        title_options = get_generate_titles(
            topic=topic,
            primary_keyword=keyword_strategy.strategy.primary_keyword,
            secondary_keywords=keyword_strategy.strategy.secondary_keywords,
            competitor_urls=competitor_urls,
            llm=llm
        )
        

        logger.info("Step 6: Creating content structure...")
        content_structure = get_create_structure(
            topic=topic,
            structure_type=structure_type,
            keywords=keyword_strategy.model_dump(),
            primary_keyword=keyword_strategy.strategy.primary_keyword,
            competitor_urls=competitor_urls,
            llm=llm
        )
        
        
        secondary_keywords_list = keyword_strategy.strategy.secondary_keywords or []
        recommended_title = title_options.recommended_title or (
            title_options.title_options[0].title if title_options.title_options else topic
        )
        oc_parts: List[str] = []
        if research_output.key_insights:
            oc_parts.append("Research - Key Insights:\n" + "\n".join(f"- {s}" for s in research_output.key_insights[:5]))
        if research_output.statistics:
            oc_parts.append("Research - Statistics:\n" + "\n".join(f"- {s}" for s in research_output.statistics[:5]))
        if research_output.content_gaps:
            oc_parts.append("Research - Content Gaps:\n" + "\n".join(f"- {s}" for s in research_output.content_gaps[:5]))
        if competitor_analysis.differentiation_opportunities:
            oc_parts.append("Competitor - Differentiation Opportunities:\n" + "\n".join(f"- {s}" for s in competitor_analysis.differentiation_opportunities[:5]))
        if competitor_analysis.common_patterns:
            oc_parts.append("Competitor - Common Patterns:\n" + "\n".join(f"- {s}" for s in competitor_analysis.common_patterns[:5]))
        if keyword_strategy and keyword_strategy.strategy:
            oc_parts.append(
                "Keyword Strategy:\n" + "\n".join([
                    f"- Primary: {keyword_strategy.strategy.primary_keyword}",
                    f"- Secondary: {', '.join(secondary_keywords_list)}",
                    f"- Long-tail: {', '.join(keyword_strategy.strategy.long_tail_opportunities[:5])}",
                    f"- Intent: {keyword_strategy.strategy.search_intent}",
                    f"- Density: {keyword_strategy.strategy.recommended_density}",
                ])
            )
        if content_structure.heading_structure:
            oc_parts.append("Outline - Headings:\n" + "\n".join(f"- {h}" for h in content_structure.heading_structure[:15]))
        if research_output.sources:
            oc_parts.append(
                "Sources (top):\n" + "\n".join(
                    f"- {s.get('title', s.get('url', ''))}: {s.get('url', '')}" for s in research_output.sources[:5]
                )
            )
        orchestration_context = "\n\n".join([p for p in oc_parts if p])
        
        logger.info("Step 7: Generating blog post...")
        blog_output = get_generate_blog(
            topic=topic,
            structure_type=structure_type,
            primary_keyword=keyword_strategy.strategy.primary_keyword,
            secondary_keywords=secondary_keywords_list,
            keywords=keyword_strategy.model_dump(),
            brand_voice=brand_voice,
            selected_title=recommended_title,
            orchestration_context=orchestration_context,
            content_structure=str(content_structure.model_dump()),
            research_findings=str(research_output.model_dump()),
            keyword_strategy=str(keyword_strategy.model_dump()),
            competitor_urls=competitor_urls,
            llm=llm
        )

        final_results = {
            "topic": topic,
            "pillar": pillar,
            "research_method": mode,
            "structure_type": structure_type,
            "brand_voice": brand_voice,
            "final_status": "completed",
            "total_execution_time": "completed",
            "created_date": datetime.now().isoformat(),
            "topic_suggestions": topic_suggestions.model_dump(),
            "research_findings": research_output.model_dump(),
            "competitor_analysis": competitor_analysis.model_dump(),
            "keyword_strategy": keyword_strategy.model_dump(),
            "title_options": title_options.model_dump(),
            "content_structure": content_structure.model_dump(),
            "blog_post": blog_output.model_dump(),
            "final_blog_content": blog_output.blog_content,
            "final_metadata": blog_output.metadata.model_dump(),
            "success_metrics": {
                "workflow_completed": True,
                "total_steps": 7,
                "word_count": blog_output.metadata.word_count,
                "seo_score": blog_output.metadata.seo_score
            }
        }
        
        logger.info(f"Complete workflow finished successfully for topic: {topic}")
        return final_results
        
    except Exception as e:
        logger.error(f"Complete workflow failed: {e}")
        return {
            "topic": topic,
            "pillar": pillar,
            "final_status": "failed",
            "total_execution_time": "failed",
            "error": str(e),
            "success_metrics": {
                "workflow_completed": False,
                "error_info": str(e)
            }
        } 
