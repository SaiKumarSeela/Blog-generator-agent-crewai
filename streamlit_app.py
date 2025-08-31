import streamlit as st
import requests
import json
from typing import Dict, List, Any
import time

# Configure Streamlit page
st.set_page_config(
    page_title="Agentic RAG Blog Generator",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# FastAPI backend URL
API_BASE_URL = "http://localhost:8085/api/v1"

# Initialize session state
if 'workflow_step' not in st.session_state:
    st.session_state.workflow_step = 1
if 'session_data' not in st.session_state:
    st.session_state.session_data = {}
if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = None

def call_api(endpoint: str, method: str = "GET", data: dict = None):
    """Helper function to call FastAPI endpoints"""
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to FastAPI backend. Make sure it's running on http://localhost:8000")
        return None
    except Exception as e:
        st.error(f"Error calling API: {str(e)}")
        return None

def display_workflow_progress():
    """Display workflow progress sidebar"""
    st.sidebar.title("📋 Workflow Progress")
    
    steps = [
        "1️⃣ Topic Selection",
        "2️⃣ Research Method", 
        "3️⃣ Competitor Analysis",
        "4️⃣ Keyword Strategy",
        "5️⃣ Title Generation",
        "6️⃣ Content Structure",
        "7️⃣ Outline Creation",
        "8️⃣ Blog Generation"
    ]
    
    for i, step in enumerate(steps, 1):
        if i < st.session_state.workflow_step:
            st.sidebar.success(step)
        elif i == st.session_state.workflow_step:
            st.sidebar.info(f"**{step}** ← Current")
        else:
            st.sidebar.write(step)

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🤖 Agentic RAG Blog Generator")
    st.subheader("Powered by CrewAI, LangChain & Gemini")
    
    # Display workflow progress
    display_workflow_progress()
    
    # Main content area
    if st.session_state.workflow_step == 1:
        step_1_topic_selection()
    elif st.session_state.workflow_step == 2:
        step_2_research_method()
    elif st.session_state.workflow_step == 3:
        step_3_competitor_analysis()
    elif st.session_state.workflow_step == 4:
        step_4_keyword_strategy()
    elif st.session_state.workflow_step == 5:
        step_5_title_generation()
    elif st.session_state.workflow_step == 6:
        step_6_content_structure()
    elif st.session_state.workflow_step == 7:
        step_7_outline_creation()
    elif st.session_state.workflow_step == 8:
        step_8_blog_generation()

def step_1_topic_selection():
    """Step 1: Topic Selection"""
    st.header("1️⃣ Topic Selection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Choose Your Approach")
        
        approach = st.radio(
            "How would you like to select your topic?",
            ["Manual Entry", "Generate Suggestions", "Quick Workflow"]
        )
        
        if approach == "Manual Entry":
            topic = st.text_input("Enter your blog topic:", placeholder="e.g., How to Build Customer Loyalty Programs")
            pillar = st.selectbox("Content Pillar:", ["Customer Loyalty", "Product Features", "Industry Insights", "Best Practices"])
            
            if st.button("Proceed with Topic") and topic:
                st.session_state.session_data['topic'] = topic
                st.session_state.session_data['pillar'] = pillar
                st.session_state.workflow_step = 2
                st.success(f"✅ Topic selected: {topic}")
                st.rerun()
        
        elif approach == "Generate Suggestions":
            pillar = st.selectbox("Select Content Pillar:", ["Customer Loyalty", "Product Features", "Industry Insights", "Best Practices"])
            
            if st.button("Generate Topic Suggestions"):
                with st.spinner("🤖 CrewAI agents generating topic suggestions..."):
                    result = call_api("/topic/generate", "POST", {"pillar": pillar})
                    
                    if result:
                        st.session_state.session_data['topic_suggestions'] = result
                        st.success("✅ Topic suggestions generated!")
                        
                        # Display suggestions
                        st.subheader("Suggested Topics:")
                        for i, topic in enumerate(result.get('topics', [])):
                            if st.button(f"📝 {topic['title']}", key=f"topic_{i}"):
                                st.session_state.session_data['topic'] = topic['title']
                                st.session_state.session_data['pillar'] = topic['pillar']
                                st.session_state.workflow_step = 2
                                st.rerun()
        
        elif approach == "Quick Workflow":
            st.info("🚀 Run the complete workflow in one go!")
            topic = st.text_input("Enter topic for quick workflow:", placeholder="Customer Loyalty Best Practices")
            pillar = st.selectbox("Content Pillar:", ["Customer Loyalty", "Product Features", "Industry Insights"])
            
            if st.button("🚀 Run Complete Workflow") and topic:
                with st.spinner("🤖 CrewAI agents working on complete workflow... This may take 2-3 minutes"):
                    result = call_api("/workflow/run", "POST", {
                        "topic": topic,
                        "pillar": pillar,
                        "method": "SERP",
                        "structure_type": "How-to Guide"
                    })
                    
                    if result:
                        st.session_state.session_data['workflow_result'] = result
                        st.session_state.current_session_id = result.get('session_id')
                        st.session_state.workflow_step = 8
                        st.success("✅ Complete workflow finished!")
                        st.rerun()
    
    with col2:
        st.subheader("ℹ️ About This Tool")
        st.info("""
        This tool uses CrewAI agents to:
        
        🔍 **Research** your topic thoroughly
        🏆 **Analyze** competitor content
        🔑 **Optimize** for SEO keywords
        📝 **Generate** high-quality blog posts
        
        Each agent specializes in a specific task to ensure comprehensive, professional content creation.
        """)

def step_2_research_method():
    """Step 2: Research Method Selection"""
    st.header("2️⃣ Research Method")
    
    topic = st.session_state.session_data.get('topic', 'No topic selected')
    st.info(f"📝 Current Topic: **{topic}**")
    
    method = st.radio(
        "Choose research method:",
        ["SERP Analysis", "Internal Knowledge Base", "Upload Reference Files"]
    )
    
    if method == "SERP Analysis":
        st.write("🔍 Will analyze top search results for your topic")
        
    elif method == "Internal Knowledge Base":
        st.write("📚 Will use internal knowledge base (brand docs, guidelines, etc.)")
        
    elif method == "Upload Reference Files":
        uploaded_files = st.file_uploader(
            "Upload reference documents",
            accept_multiple_files=True,
            type=['pdf', 'txt', 'md', 'docx']
        )
        
        if uploaded_files:
            st.success(f"📁 {len(uploaded_files)} files ready for upload")
    
    if st.button("Start Research"):
        with st.spinner("🤖 Knowledge Retriever agent conducting research..."):
            result = call_api("/research/run", "POST", {
                "topic": topic,
                "method": method.split()[0]  # "SERP", "Internal", "Upload"
            })
            
            if result:
                st.session_state.session_data['research'] = result
                st.session_state.workflow_step = 3
                st.success("✅ Research completed!")
                
                # Display research findings
                findings = result.get('findings', {})
                if findings.get('insights'):
                    st.subheader("🔍 Research Insights:")
                    for insight in findings['insights']:
                        st.write(f"**{insight['source']}**: {insight['snippet']}")
                
                st.rerun()

def step_3_competitor_analysis():
    """Step 3: Competitor Analysis"""
    st.header("3️⃣ Competitor Analysis")
    
    topic = st.session_state.session_data.get('topic', 'No topic selected')
    st.info(f"📝 Analyzing competitors for: **{topic}**")
    
    # Option to provide specific URLs or let system find them
    analysis_type = st.radio(
        "Competitor analysis method:",
        ["Auto-find competitors", "Provide specific URLs"]
    )
    
    competitor_urls = []
    if analysis_type == "Provide specific URLs":
        st.subheader("Enter competitor URLs:")
        for i in range(3):
            url = st.text_input(f"Competitor URL {i+1}:", key=f"url_{i}")
            if url:
                competitor_urls.append(url)
    
    if st.button("Analyze Competitors"):
        with st.spinner("🤖 Competitor Analyst agent analyzing content..."):
            result = call_api("/competitors/analyze", "POST", {
                "urls": competitor_urls if competitor_urls else [f"https://example.com/competitor-{i}" for i in range(3)],
                "topic": topic
            })
            
            if result:
                st.session_state.session_data['competitor_analysis'] = result
                st.session_state.workflow_step = 4
                st.success("✅ Competitor analysis completed!")
                
                # Display analysis results
                analysis = result.get('analysis', {})
                if analysis:
                    st.subheader("🏆 Analysis Results:")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Brand vs Competitors:**")
                        comparison = analysis.get('comparison_grid', {}).get('brand_vs_competitors', {})
                        for key, value in comparison.items():
                            st.write(f"- **{key.title()}**: {value}")
                    
                    with col2:
                        st.write("**Recommendations:**")
                        recommendations = analysis.get('recommendations', [])
                        for rec in recommendations:
                            st.write(f"- {rec}")
                
                st.rerun()

def step_4_keyword_strategy():
    """Step 4: Keyword Strategy Development"""
    st.header("4️⃣ Keyword Strategy")
    
    topic = st.session_state.session_data.get('topic', 'No topic selected')
    research_data = st.session_state.session_data.get('research', {})
    
    st.info(f"📝 Developing keywords for: **{topic}**")
    
    if st.button("Generate Keyword Strategy"):
        with st.spinner("🤖 SEO Strategist agent developing keyword strategy..."):
            result = call_api("/seo/keywords", "POST", {
                "topic": topic,
                "findings": json.dumps(research_data)
            })
            
            if result:
                st.session_state.session_data['keywords'] = result
                st.session_state.workflow_step = 5
                st.success("✅ Keyword strategy developed!")
                
                # Display keyword strategy
                strategy = result.get('strategy', {})
                st.subheader("🔑 Keyword Strategy:")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Primary Keyword:** {strategy.get('primary', 'N/A')}")
                    st.write("**Secondary Keywords:**")
                    for kw in strategy.get('secondary', []):
                        st.write(f"- {kw}")
                
                with col2:
                    st.write("**Rationale:**")
                    st.write(strategy.get('rationale', 'Keyword strategy optimized for search volume and relevance'))
                
                st.rerun()

def step_5_title_generation():
    """Step 5: Title Generation"""
    st.header("5️⃣ Title Generation")
    
    topic = st.session_state.session_data.get('topic', 'No topic selected')
    keywords = st.session_state.session_data.get('keywords', {}).get('strategy', {})
    
    st.info(f"📝 Generating titles for: **{topic}**")
    
    if st.button("Generate Title Options"):
        with st.spinner("🤖 Content Structurer agent creating title options..."):
            result = call_api("/titles/generate", "POST", {
                "topic": topic,
                "primary_keyword": keywords.get('primary', topic),
                "secondary_keywords": keywords.get('secondary', [])
            })
            
            if result:
                st.session_state.session_data['titles'] = result
                st.session_state.workflow_step = 6
                st.success("✅ Title options generated!")
                
                # Display title options
                titles = result.get('titles', {})
                st.subheader("📝 Title Options:")
                
                selected_title = st.radio(
                    "Choose your preferred title:",
                    titles.get('options', []),
                    index=0
                )
                
                st.session_state.session_data['selected_title'] = selected_title
                
                st.write(f"**Recommended:** {titles.get('recommended', 'N/A')}")
                st.write(f"**Rationale:** {titles.get('rationale', 'SEO-optimized for target keywords')}")
                
                st.rerun()

def step_6_content_structure():
    """Step 6: Content Structure Selection"""
    st.header("6️⃣ Content Structure")
    
    topic = st.session_state.session_data.get('topic', 'No topic selected')
    st.info(f"📝 Selecting structure for: **{topic}**")
    
    structure_type = st.selectbox(
        "Choose blog structure:",
        ["How-to Guide", "Listicle", "Thought-leadership", "Deep-dive"]
    )
    
    # Show structure preview
    structure_descriptions = {
        "How-to Guide": "Sequential, actionable steps with clear instructions",
        "Listicle": "Numbered or bulleted points with detailed explanations",
        "Thought-leadership": "Opinion-based content with unique perspectives",
        "Deep-dive": "Comprehensive analysis with detailed research"
    }
    
    st.write(f"**Structure Description:** {structure_descriptions[structure_type]}")
    
    if st.button("Select Structure"):
        with st.spinner("🤖 Creating structure template..."):
            result = call_api("/structure/select", "POST", {
                "structure_type": structure_type
            })
            
            if result:
                st.session_state.session_data['structure'] = result
                st.session_state.workflow_step = 7
                st.success("✅ Structure template created!")
                
                # Display structure template
                template = result.get('template', {})
                st.subheader("📋 Structure Template:")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Sections:**")
                    for section in template.get('sections', []):
                        st.write(f"- {section}")
                
                with col2:
                    st.write(f"**Format:** {template.get('format', 'N/A')}")
                    st.write(f"**CTA Placement:** {template.get('cta_placement', 'N/A')}")
                
                st.rerun()

def step_7_outline_creation():
    """Step 7: Outline Creation"""
    st.header("7️⃣ Outline Creation")
    
    topic = st.session_state.session_data.get('topic', 'No topic selected')
    structure = st.session_state.session_data.get('structure', {})
    keywords = st.session_state.session_data.get('keywords', {}).get('strategy', {})
    
    st.info(f"📝 Creating detailed outline for: **{topic}**")
    
    if st.button("Create Detailed Outline"):
        with st.spinner("🤖 Content Structurer agent creating outline..."):
            result = call_api("/outline/create", "POST", {
                "topic": topic,
                "structure": json.dumps(structure),
                "keywords": keywords
            })
            
            if result:
                st.session_state.session_data['outline'] = result
                st.session_state.workflow_step = 8
                st.success("✅ Detailed outline created!")
                
                # Display outline
                outline = result.get('outline', {})
                st.subheader("📋 Content Outline:")
                
                for section in outline.get('sections', []):
                    with st.expander(f"📝 {section['heading']} ({section['word_count']} words)"):
                        st.write(f"**Intent:** {section['intent']}")
                        st.write("**Key Points:**")
                        for point in section['key_points']:
                            st.write(f"- {point}")
                
                st.write(f"**Total Estimated Words:** {outline.get('total_estimated_words', 'N/A')}")
                
                st.rerun()

def step_8_blog_generation():
    """Step 8: Final Blog Generation"""
    st.header("8️⃣ Blog Generation")
    
    # Check if we have workflow result from quick workflow
    if 'workflow_result' in st.session_state.session_data:
        display_workflow_result()
        return
    
    topic = st.session_state.session_data.get('topic', 'No topic selected')
    outline = st.session_state.session_data.get('outline', {})
    keywords = st.session_state.session_data.get('keywords', {}).get('strategy', {})
    research = st.session_state.session_data.get('research', {})
    
    st.info(f"📝 Generating final blog for: **{topic}**")
    
    # Display summary before generation
    with st.expander("📋 Generation Summary"):
        st.write(f"**Topic:** {topic}")
        st.write(f"**Primary Keyword:** {keywords.get('primary', 'N/A')}")
        st.write(f"**Structure:** {outline.get('structure', 'N/A')}")
        st.write(f"**Estimated Length:** {outline.get('outline', {}).get('total_estimated_words', 'N/A')} words")
    
    if st.button("🚀 Generate Complete Blog Post"):
        with st.spinner("🤖 Blog Writer agent crafting your blog post... This may take 1-2 minutes"):
            result = call_api("/blog/generate", "POST", {
                "topic": topic,
                "outline": json.dumps(outline),
                "keywords": keywords,
                "research_findings": json.dumps(research)
            })
            
            if result:
                st.session_state.session_data['final_blog'] = result
                st.success("✅ Blog post generated successfully!")
                
                display_blog_result(result)

def display_workflow_result():
    """Display results from quick workflow"""
    workflow_result = st.session_state.session_data['workflow_result']
    
    st.success("🎉 Complete Workflow Finished!")
    
    # Display workflow summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Workflow Summary")
        st.write(f"**Topic:** {workflow_result.get('topic', 'N/A')}")
        st.write(f"**Pillar:** {workflow_result.get('pillar', 'N/A')}")
        st.write(f"**Status:** {workflow_result.get('workflow_status', 'N/A')}")
        st.write(f"**Generated:** {workflow_result.get('generated_at', 'N/A')}")
    
    with col2:
        st.subheader("🔑 SEO Strategy")
        keywords = workflow_result.get('results', {}).get('keyword_strategy', {})
        st.write(f"**Primary:** {keywords.get('primary', 'N/A')}")
        st.write("**Secondary:**")
        for kw in keywords.get('secondary', []):
            st.write(f"- {kw}")
    
    # Display generated blog
    blog_data = workflow_result.get('results', {}).get('final_blog', {})
    if blog_data:
        st.subheader("📝 Generated Blog Post")
        
        # Blog metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Word Count", blog_data.get('word_count', 'N/A'))
        with col2:
            st.metric("Reading Time", blog_data.get('reading_time', 'N/A'))
        with col3:
            st.metric("Agent Status", "✅ Complete")
        
        # Blog content
        st.markdown("### Preview:")
        st.markdown(blog_data.get('content', 'No content generated'))
        
        # Export options
        display_export_options(workflow_result.get('session_id'))

def display_blog_result(result):
    """Display generated blog result"""
    blog_data = result.get('blog', {})
    
    st.subheader("📝 Generated Blog Post")
    
    # Metadata
    metadata = blog_data.get('metadata', {})
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Word Count", metadata.get('word_count', 'N/A'))
    with col2:
        st.metric("Reading Time", metadata.get('reading_time', 'N/A'))
    with col3:
        st.metric("Keywords", len(metadata.get('keywords', {}).get('secondary', [])) + 1)
    
    # Content preview
    content = blog_data.get('content', 'No content generated')
    st.markdown("### Content Preview:")
    st.markdown(content)
    
    # Citations
    citations = blog_data.get('citations', [])
    if citations:
        st.subheader("📚 Sources & Citations:")
        for citation in citations:
            st.write(f"- **{citation['source']}**: {citation['url']}")
    
    # Export options
    display_export_options(result.get('session_id'))

def display_export_options(session_id):
    """Display export options"""
    if not session_id:
        return
    
    st.subheader("📤 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export as Markdown"):
            result = call_api(f"/export/{session_id}?format=markdown")
            if result:
                st.download_button(
                    "⬇️ Download Markdown",
                    result.get('content', ''),
                    file_name="blog_post.md",
                    mime="text/markdown"
                )
    
    with col2:
        if st.button("🌐 Export as HTML"):
            result = call_api(f"/export/{session_id}?format=html")
            if result:
                st.download_button(
                    "⬇️ Download HTML",
                    result.get('content', ''),
                    file_name="blog_post.html",
                    mime="text/html"
                )
    
    with col3:
        if st.button("📊 Export as JSON"):
            result = call_api(f"/export/{session_id}?format=json")
            if result:
                st.download_button(
                    "⬇️ Download JSON",
                    json.dumps(result.get('content', {}), indent=2),
                    file_name="blog_data.json",
                    mime="application/json"
                )

# Navigation buttons
def show_navigation():
    """Show navigation buttons"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.session_state.workflow_step > 1:
            if st.button("⬅️ Previous Step"):
                st.session_state.workflow_step -= 1
                st.rerun()
    
    with col2:
        # Reset workflow button
        if st.button("🔄 Start New Workflow"):
            st.session_state.workflow_step = 1
            st.session_state.session_data = {}
            st.session_state.current_session_id = None
            st.rerun()
    
    with col3:
        # Skip to quick workflow
        if st.session_state.workflow_step < 8:
            if st.button("⚡ Quick Workflow"):
                st.session_state.workflow_step = 1
                st.rerun()

# Footer with API status
def show_footer():
    """Show footer with API connection status"""
    st.markdown("---")
    
    # Check API health
    try:
        health_response = requests.get(f"http://localhost:8000/health", timeout=2)
        if health_response.status_code == 200:
            st.success("✅ Connected to FastAPI Backend")
        else:
            st.error("❌ FastAPI Backend Issues")
    except:
        st.error("❌ FastAPI Backend Not Running - Start with: `python fastapi_app.py`")
    
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Powered by CrewAI Agents 🤖 | LangChain 🦜 | Gemini AI 🚀</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    
    # Show navigation at bottom
    st.markdown("---")
    show_navigation()
    show_footer()