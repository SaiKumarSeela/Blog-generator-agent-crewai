from langchain.prompts import PromptTemplate

# Prompt Templates
topic_generation_prompt = PromptTemplate(
    input_variables=["pillar", "web_results"],
    template="""
Generate 6 engaging and contextually relevant topic suggestions strictly aligned with the given content pillar: {pillar}.

You are provided with real-time web search results about {pillar}:
{web_results}

CRITICAL INSTRUCTIONS:
- Use the specific information from the web search results to create concrete, specific topics
- DO NOT use placeholder text like "{pillar}" in your topic titles or descriptions
- Replace {pillar} with the actual pillar name: {pillar}
- Base your topics on the actual trends, technologies, and developments mentioned in the search results
- Create specific, actionable topic titles that reflect real current trends in {pillar}
- Make descriptions concrete and specific, not generic
- Ensure topics are directly connected to the actual content of the search results

FORMAT REQUIREMENTS:
- Generate exactly 6 topics
- Use ID format: "topic_1", "topic_2", "topic_3", "topic_4", "topic_5", "topic_6"
- Set content_pillar_alignment to exactly "{pillar}" (the actual pillar name)
- Use relevance_score as decimal numbers (e.g., 9.5, 9.0, 8.5, 8.0, 9.2, 7.8)
- Use estimated_engagement_potential as "High" or "Medium" only
- Make target_audience specific and professional (e.g., "Technology professionals, policymakers, investors")

Expected Output:
A structured list of exactly 6 topic suggestions with fields:
- id (string): "topic_1", "topic_2", "topic_3", "topic_4", "topic_5", "topic_6"
- title (string): Specific, engaging topic titles based on search results
- description (string): Detailed descriptions explaining the topic's importance
- target_audience (string): Specific professional audiences
- content_pillar_alignment (string): Must be exactly "{pillar}"
- estimated_engagement_potential (string): "High" or "Medium" only
- relevance_score (float): Decimal numbers between 7.0-10.0 (e.g., 9.5, 8.5, 9.2)
"""
)

research_topic_prompt = PromptTemplate(
    input_variables=["topic", "research_method", "research_context"],
    template="""
Research the selected topic: {topic} using the specified method: {research_method}.

You are provided with research context from tools:
{research_context}

Extract key insights, statistics, and supporting information from the research context.
Analyze the findings and identify content gaps and opportunities.

Expected Output:
Comprehensive research findings including:
- Key insights and statistics
- Supporting evidence with sources
- Content gaps and opportunities
- Recommended talking points
- Source citations with confidence scores
"""
)

competitor_analysis_prompt = PromptTemplate(
    input_variables=["topic", "competitor_urls", "competitor_context"],
    template="""
Analyze competitor content for the topic: {topic}.

You are provided with competitor analysis context from tools:
{competitor_context}

Extract tone, structure, keyword usage, content gaps, and differentiation opportunities.

Expected Output:
Competitor analysis report with:
- Competitor content summaries
- Tone and style analysis
- Content structure patterns
- Keyword strategies used
- Opportunities for differentiation
- What to replicate vs avoid
"""
)

keyword_strategy_prompt = PromptTemplate(
    input_variables=["topic", "primary_keyword", "research_findings"],
    template="""
Develop a comprehensive keyword strategy for the topic: {topic}.
Use the keyword research context and research findings to inform your strategy.

Research Findings:
{research_findings}

Identify 1 primary keyword and 2-3 secondary keywords.
Consider search volume, competition, and alignment with topic and research findings.
Use the primary keyword: {primary_keyword} as the main target if provided, otherwise suggest one.

Expected Output:
Keyword strategy including:
- Primary keyword with rationale
- Secondary keywords (2-3)
- Long-tail keyword opportunities
- Search intent analysis
- Recommended keyword density
- Integration strategy for content
"""
)

title_generation_prompt = PromptTemplate(
    input_variables=["topic", "primary_keyword", "secondary_keywords"],
    template="""
Generate 5-7 compelling title options for the topic: {topic} using the primary keyword: {primary_keyword}.

Use keyword strategy to inform your title generation.
Create variations including listicles, how-to guides, thought leadership, and SEO-optimized titles.
Consider click-through rate optimization and audience engagement factors.

Expected Output:
List of title options with for each item:
- title (string)
- type (string; one of: SEO-optimized, Listicle, How-to, Thought leadership)
- seo_score (float 0.0-10.0)
- engagement_score (float 0.0-10.0)
- rationale (string; briefly explain why this title would perform well)
- recommended_title (string; overall best choice)
- recommendation_rationale (string; why the recommended title is best)
- seo_considerations (list of strings)
"""
)

content_structure_prompt = PromptTemplate(
    input_variables=["topic", "structure_type", "keywords", "primary_keyword"],
    template="""
Create a detailed content structure and outline for the topic/title: {topic} using the selected structure type: {structure_type}.

Use research findings and keyword strategy to inform your structure.
Include section headings, key points, and content intent for each section.
This will serve as the blueprint for the final blog post.

Expected Output:
Comprehensive content outline with:
- H1, H2, H3 heading structure
- Key points for each section
- Content intent and tone notes
- CTA placement recommendations
- FAQ section if applicable
- Estimated word count per section
- Keyword placement strategy
"""
)
blog_generation_prompt = PromptTemplate(
    input_variables=[
        "topic",
        "structure_type", 
        "primary_keyword",
        "secondary_keywords",
        "keywords",
        "research_method",
        "brand_voice",
        "selected_title",
        "orchestration_context",
        "content_structure",
        "research_findings",
        "keyword_strategy"
    ],
    template="""
Generate a complete, high-quality blog post following these STRICT requirements:

TITLE COMPLIANCE (MANDATORY):
- Title: {selected_title}
- YOU MUST use this EXACT title as your H1 header
- DO NOT modify, change, or substitute this title

CONTENT STRUCTURE COMPLIANCE (MANDATORY):
Content Structure to Follow:
{content_structure}

- Follow the provided content structure EXACTLY
- Use the EXACT heading hierarchy and text specified in the structure
- DO NOT add, remove, or reorder sections
- Fill in content under each heading as specified in the structure

RESEARCH INTEGRATION (MANDATORY):
Research Findings to Incorporate:
{research_findings}

- Use ALL key insights from research findings
- Incorporate specific statistics and data points mentioned
- Reference talking points provided in research findings
- If using RAG mode, prioritize insights from uploaded documents

KEYWORD INTEGRATION (MANDATORY):
Keyword Strategy:
{keyword_strategy}

- Primary Keyword: {primary_keyword} (use throughout content naturally)
- Secondary Keywords: {secondary_keywords} (integrate in subheadings and content)
- Additional Keywords: {keywords}
- Follow the keyword density and placement strategy specified

ADDITIONAL CONTEXT:
- Topic: {topic}
- Structure Type: {structure_type}
- Research Method: {research_method}
- Brand Voice: {brand_voice}

Prior Outputs Context:
{orchestration_context}

CONTENT REQUIREMENTS:
- Create engaging, informative content (2000-4000 words)
- Use natural keyword integration based on keyword strategy
- Include proper citations and sources from research findings
- Maintain brand voice consistency: {brand_voice}
- Ensure content follows the specified structure type format
- Include internal linking suggestions where appropriate

QUALITY STANDARDS:
- Ensure factual accuracy based on research findings
- Maintain readability and coherence
- Provide actionable insights and practical value
- Include examples and case studies from research context where available

Expected Output:
Complete blog post with:
- H1: {selected_title} (EXACT match required)
- Content following the exact structure provided
- Natural integration of ALL research findings
- Proper keyword optimization based on strategy
- Meta description optimized for primary keyword
- Reading time and word count
- SEO analysis summary
- Content quality metrics
- sections_rendered: List of headings exactly as they appear in content structure
- research_utilization_report: How research findings were integrated
- keyword_integration_report: How keywords were used throughout content
"""
)

orchestration_prompt = PromptTemplate(
    input_variables=["topic", "pillar", "research_method", "structure_type", "primary_keyword", "keywords", "competitor_urls", "uploads"],
    template="""
Execute the complete blog generation workflow for topic: {topic}.
Coordinate all agents to deliver a comprehensive, SEO-optimized blog post
following all brand guidelines and quality standards.

Parameters:
- Topic: {topic}
- Pillar: {pillar}
- Research Method: {research_method}
- Structure Type: {structure_type}
- Primary Keyword: {primary_keyword}
- Keywords: {keywords}
- Competitor URLs: {competitor_urls} (optional)
- Uploads: {uploads} (for RAG mode)

Expected Output:
Complete workflow results including:
- Final blog post (Markdown format)
- Keyword strategy summary
- Research sources and citations
- SEO optimization metrics
- Content quality assessment
- Metadata (word count, reading time, etc.)
- Workflow execution summary
"""
)