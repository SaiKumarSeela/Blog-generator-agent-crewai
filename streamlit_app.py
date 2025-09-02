import streamlit as st
import requests
import json
import time
from typing import Dict, List, Any
import uuid
import pandas as pd
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8085"  # Change this to your FastAPI server URL

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'workflow_data' not in st.session_state:
    st.session_state.workflow_data = {}
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1

class APIClient:
    """Client for FastAPI backend"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
    
    def make_request(self, method: str, endpoint: str, data: Dict = None, files=None):
        """Make HTTP request to API"""
        url = f"{self.base_url}{endpoint}"
        try:
            if method.upper() == "GET":
                response = requests.get(url)
            elif method.upper() == "POST":
                if files:
                    response = requests.post(url, files=files)
                else:
                    response = requests.post(url, json=data)
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {str(e)}")
            return None
    
    def health_check(self):
        return self.make_request("GET", "/health")
    
    def generate_topics(self, pillar: str):
        return self.make_request("POST", "/topic/generate", {"pillar": pillar})
    
    def run_research(self, topic: str, mode: str, urls: List[str] = None, uploads: List[str] = None):
        data = {
            "topic": topic,
            "mode": mode,
            "urls": urls or [],
            "uploads": uploads or []
        }
        return self.make_request("POST", "/research/run", data)
    
    def analyze_competitors(self, urls: List[str], topic: str = None):
        data = {"urls": urls, "topic": topic}
        return self.make_request("POST", "/competitors/analyse", data)
    
    def generate_keywords(self, topic: str, findings: str):
        data = {"topic": topic, "findings": findings}
        return self.make_request("POST", "/seo/keywords", data)
    
    def generate_titles(self, topic: str, primary: str, secondary: List[str]):
        data = {"topic": topic, "primary": primary, "secondary": secondary}
        return self.make_request("POST", "/titles/generate", data)
    
    def select_structure(self, structure_type: str):
        return self.make_request("POST", "/structure/select", {"type": structure_type})
    
    def create_outline(self, topic: str, structure: str, keywords: Dict):
        data = {"topic": topic, "structure": structure, "keywords": keywords}
        return self.make_request("POST", "/outline/create", data)
    
    def generate_blog(self, outline: str, keywords: Dict, brand_voice: str):
        data = {"outline": outline, "keywords": keywords, "brand_voice": brand_voice}
        return self.make_request("POST", "/blog/generate", data)
    
    def run_workflow(self, topic: str, pillar: str, mode: str = "SERP", 
                    structure: str = "How-to Guide", brand_voice: str = "professional"):
        data = {
            "topic": topic,
            "pillar": pillar,
            "mode": mode,
            "structure": structure,
            "brand_voice": brand_voice
        }
        return self.make_request("POST", "/workflow/run", data)
    
    def export_results(self, session_id: str, format: str = "json"):
        return self.make_request("GET", f"/export/{session_id}?format={format}")
    
    def upload_file(self, file):
        files = {"file": file}
        url = f"{self.base_url}/upload"
        try:
            response = requests.post(url, files=files)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Upload Error: {str(e)}")
            return None

# Initialize API client
api_client = APIClient(API_BASE_URL)

def main():
    st.set_page_config(
        page_title="Blog Generator AI Agent",
        page_icon="📝",
        layout="wide"
    )
    
    st.title("🤖 Blog Generator AI Agent")
    st.markdown("---")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    
    workflow_option = st.sidebar.selectbox(
        "Choose Workflow",
        ["Step-by-Step Generation", "Complete Workflow", "Export Results"]
    )
    
    # Check API health
    with st.sidebar:
        if st.button("🔍 Check API Health"):
            health = api_client.health_check()
            if health:
                st.success("✅ API is healthy")
            else:
                st.error("❌ API is not responding")
    
    if workflow_option == "Step-by-Step Generation":
        step_by_step_workflow()
    elif workflow_option == "Complete Workflow":
        complete_workflow()
    elif workflow_option == "Export Results":
        export_results()

def step_by_step_workflow():
    """Step-by-step blog generation workflow"""
    
    st.header("Step-by-Step Blog Generation")
    
    # Progress bar
    progress = st.progress(0)
    step_names = [
        "Topic Selection",
        "Research",
        "Competitor Analysis", 
        "SEO Keywords",
        "Title Generation",
        "Content Structure",
        "Outline Creation",
        "Blog Generation"
    ]
    
    current_step = st.session_state.get('current_step', 1)
    progress.progress(current_step / len(step_names))
    
    st.subheader(f"Step {current_step}: {step_names[current_step-1]}")
    
    if current_step == 1:
        step_1_topic_selection()
    elif current_step == 2:
        step_2_research()
    elif current_step == 3:
        step_3_competitor_analysis()
    elif current_step == 4:
        step_4_seo_keywords()
    elif current_step == 5:
        step_5_title_generation()
    elif current_step == 6:
        step_6_content_structure()
    elif current_step == 7:
        step_7_outline_creation()
    elif current_step == 8:
        step_8_blog_generation()

def step_1_topic_selection():
    """Step 1: Topic Selection"""
    st.markdown("### 📝 Enter a Topic or Generate Suggestions")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Manual Topic Entry")
        manual_topic = st.text_input("Enter your topic:", placeholder="e.g., Digital Marketing Strategies")
        
        if st.button("Use Manual Topic") and manual_topic:
            st.session_state.workflow_data['topic'] = manual_topic
            st.session_state.workflow_data['pillar'] = manual_topic
            st.success(f"Selected topic: {manual_topic}")
            if st.button("Continue to Research →"):
                st.session_state.current_step = 2
                st.rerun()
    
    with col2:
        st.subheader("Generate Topic Suggestions")
        pillar = st.text_input("Enter content pillar:", placeholder="e.g., Marketing, Technology, Health")
        
        if st.button("🚀 Generate Topics") and pillar:
            with st.spinner("Generating topic suggestions..."):
                result = api_client.generate_topics(pillar)
                
                if result and 'topics' in result:
                    st.session_state.session_id = result.get('session_id')
                    st.session_state.workflow_data['generated_topics'] = result['topics']
                    
                    st.success("Topics generated successfully!")
                    
                    # Display topics for selection
                    for i, topic in enumerate(result['topics']):
                        if st.button(f"Select: {topic.get('title', 'Untitled')}", key=f"topic_{i}"):
                            st.session_state.workflow_data['topic'] = topic.get('title', '')
                            st.session_state.workflow_data['pillar'] = topic.get('pillar', pillar)
                            st.session_state.workflow_data['topic_description'] = topic.get('description', '')
                            st.success(f"Selected: {topic.get('title', '')}")
                            
                            if st.button("Continue to Research →", key="continue_research"):
                                st.session_state.current_step = 2
                                st.rerun()

def step_2_research():
    """Step 2: Research"""
    if 'topic' not in st.session_state.workflow_data:
        st.error("Please complete Step 1: Topic Selection first")
        if st.button("← Back to Topic Selection"):
            st.session_state.current_step = 1
            st.rerun()
        return
    
    st.markdown("### 🔍 Choose Research Method")
    
    topic = st.session_state.workflow_data['topic']
    st.write(f"**Topic:** {topic}")
    
    research_mode = st.selectbox(
        "Select Research Mode:",
        ["SERP", "RAG", "reference"],
        help="SERP: Search engine results, RAG: Internal knowledge base, reference: Upload articles"
    )
    
    # Additional inputs based on mode
    urls = []
    uploads = []
    
    if research_mode == "reference":
        st.subheader("Upload Reference Files")
        uploaded_files = st.file_uploader(
            "Upload reference documents",
            type=['pdf', 'txt', 'md'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for file in uploaded_files:
                upload_result = api_client.upload_file(file)
                if upload_result:
                    uploads.append(upload_result['file_path'])
                    st.success(f"Uploaded: {file.name}")
    
    elif research_mode == "SERP":
        st.subheader("Additional URLs (Optional)")
        url_input = st.text_area("Enter URLs (one per line):")
        if url_input:
            urls = [url.strip() for url in url_input.split('\n') if url.strip()]
    
    if st.button("🔍 Start Research"):
        with st.spinner("Conducting research..."):
            result = api_client.run_research(topic, research_mode, urls, uploads)
            
            if result and 'findings' in result:
                st.session_state.workflow_data['research_findings'] = result['findings']
                st.session_state.workflow_data['research_mode'] = research_mode
                
                st.success("Research completed!")
                
                # Display findings
                st.subheader("Research Findings")
                for finding in result['findings']:
                    with st.expander(f"Finding from {finding.get('source', 'Unknown')}"):
                        st.write(finding.get('snippet', ''))
                        st.write(f"**Tags:** {', '.join(finding.get('tags', []))}")
                        st.write(f"**Confidence:** {finding.get('confidence', 0)}")
                
                if st.button("Continue to Competitor Analysis →"):
                    st.session_state.current_step = 3
                    st.rerun()

def step_3_competitor_analysis():
    """Step 3: Competitor Analysis"""
    if 'topic' not in st.session_state.workflow_data:
        st.error("Please complete previous steps first")
        return
    
    st.markdown("### 🏆 Competitor Analysis")
    
    topic = st.session_state.workflow_data['topic']
    st.write(f"**Topic:** {topic}")
    
    # Input competitor URLs
    st.subheader("Competitor URLs")
    competitor_urls_input = st.text_area(
        "Enter competitor URLs (one per line):",
        placeholder="https://competitor1.com/blog/article\nhttps://competitor2.com/guide"
    )
    
    competitor_urls = []
    if competitor_urls_input:
        competitor_urls = [url.strip() for url in competitor_urls_input.split('\n') if url.strip()]
    
    # Option to auto-find competitors
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🔍 Auto-Find Competitors") and topic:
            st.info("This would search for competitor content automatically (not implemented in demo)")
            # In real implementation, this would use SERP analysis to find competitors
            
    with col2:
        if st.button("📊 Analyze Competitors") and competitor_urls:
            with st.spinner("Analyzing competitors..."):
                result = api_client.analyze_competitors(competitor_urls, topic)
                
                if result:
                    st.session_state.workflow_data['competitor_analysis'] = result
                    
                    st.success("Competitor analysis completed!")
                    
                    # Display comparison grid
                    if 'comparison_grid' in result:
                        st.subheader("Comparison Grid")
                        grid = result['comparison_grid']
                        
                        df = pd.DataFrame([
                            ["Brand Analysis", grid.get('brand_analysis', '')],
                            ["Competitor Analysis", grid.get('competitor_analysis', '')],
                            ["Tone Comparison", grid.get('tone_comparison', '')],
                            ["Structure Comparison", grid.get('structure_comparison', '')],
                            ["Keyword Usage", grid.get('keyword_usage', '')]
                        ], columns=['Aspect', 'Analysis'])
                        
                        st.dataframe(df, use_container_width=True)
                    
                    # Display recommendations
                    if 'recommendations' in result:
                        st.subheader("Recommendations")
                        for rec in result['recommendations']:
                            st.write(f"• {rec}")
                    
                    if st.button("Continue to SEO Keywords →"):
                        st.session_state.current_step = 4
                        st.rerun()

def step_4_seo_keywords():
    """Step 4: SEO Keywords"""
    if 'research_findings' not in st.session_state.workflow_data:
        st.error("Please complete research step first")
        return
    
    st.markdown("### 🎯 SEO Keywords Generation")
    
    topic = st.session_state.workflow_data['topic']
    findings = st.session_state.workflow_data['research_findings']
    
    st.write(f"**Topic:** {topic}")
    
    # Convert findings to string for API
    findings_text = ""
    if isinstance(findings, list):
        findings_text = " ".join([f.get('snippet', '') for f in findings])
    elif isinstance(findings, str):
        findings_text = findings
    
    if st.button("🎯 Generate Keywords"):
        with st.spinner("Generating SEO keywords..."):
            result = api_client.generate_keywords(topic, findings_text)
            
            if result:
                st.session_state.workflow_data['keywords'] = {
                    'primary': result.get('primary', ''),
                    'secondary': result.get('secondary', []),
                    'rationale': result.get('rationale', '')
                }
                
                st.success("Keywords generated!")
                
                # Display keywords
                st.subheader("Generated Keywords")
                st.write(f"**Primary Keyword:** {result.get('primary', '')}")
                st.write(f"**Secondary Keywords:** {', '.join(result.get('secondary', []))}")
                
                with st.expander("Rationale"):
                    st.write(result.get('rationale', ''))
                
                # Allow editing
                st.subheader("Edit Keywords (Optional)")
                primary = st.text_input("Primary Keyword:", value=result.get('primary', ''))
                secondary_input = st.text_area(
                    "Secondary Keywords (one per line):", 
                    value='\n'.join(result.get('secondary', []))
                )
                
                if st.button("Update Keywords"):
                    secondary = [s.strip() for s in secondary_input.split('\n') if s.strip()]
                    st.session_state.workflow_data['keywords'] = {
                        'primary': primary,
                        'secondary': secondary,
                        'rationale': result.get('rationale', '')
                    }
                    st.success("Keywords updated!")
                
                if st.button("Continue to Title Generation →"):
                    st.session_state.current_step = 5
                    st.rerun()

def step_5_title_generation():
    """Step 5: Title Generation"""
    if 'keywords' not in st.session_state.workflow_data:
        st.error("Please complete keyword generation first")
        return
    
    st.markdown("### 📝 Title Generation")
    
    topic = st.session_state.workflow_data['topic']
    keywords = st.session_state.workflow_data['keywords']
    
    st.write(f"**Topic:** {topic}")
    st.write(f"**Primary Keyword:** {keywords.get('primary', '')}")
    
    if st.button("📝 Generate Titles"):
        with st.spinner("Generating title options..."):
            result = api_client.generate_titles(
                topic, 
                keywords.get('primary', ''), 
                keywords.get('secondary', [])
            )
            
            if result:
                st.session_state.workflow_data['titles'] = result.get('titles', [])
                st.session_state.workflow_data['chosen_title'] = result.get('chosen_title', '')
                
                st.success("Titles generated!")
                
                # Display title options
                st.subheader("Title Options")
                selected_title = st.radio(
                    "Choose a title:",
                    result.get('titles', []),
                    index=0 if result.get('titles') else None
                )
                
                # Option for custom title
                st.subheader("Custom Title (Optional)")
                custom_title = st.text_input("Enter custom title:")
                
                final_title = custom_title if custom_title else selected_title
                
                if st.button("Confirm Title Selection") and final_title:
                    st.session_state.workflow_data['chosen_title'] = final_title
                    st.success(f"Selected title: {final_title}")
                    
                    if st.button("Continue to Structure Selection →"):
                        st.session_state.current_step = 6
                        st.rerun()

def step_6_content_structure():
    """Step 6: Content Structure Selection"""
    st.markdown("### 🏗️ Content Structure Selection")
    
    structure_types = [
        "How-to Guide",
        "Listicle", 
        "Thought leadership",
        "Deep-dive explainer"
    ]
    
    selected_structure = st.selectbox("Choose content structure:", structure_types)
    
    # Show structure description
    descriptions = {
        "How-to Guide": "Step-by-step instructional content with clear actionable steps",
        "Listicle": "List-based content with numbered or bulleted items",
        "Thought leadership": "Opinion-based content establishing expertise and authority",
        "Deep-dive explainer": "Comprehensive analysis with detailed explanations"
    }
    
    st.info(descriptions.get(selected_structure, ''))
    
    if st.button("🏗️ Get Structure Template"):
        with st.spinner("Getting structure template..."):
            result = api_client.select_structure(selected_structure)
            
            if result and 'layout_template' in result:
                st.session_state.workflow_data['structure'] = selected_structure
                st.session_state.workflow_data['layout_template'] = result['layout_template']
                
                st.success("Structure template loaded!")
                
                # Display template
                template = result['layout_template']
                st.subheader("Layout Template")
                st.write(f"**Structure:** {template.get('structure', '')}")
                
                st.write("**Sections:**")
                for section in template.get('sections', []):
                    st.write(f"• {section}")
                
                st.write("**Elements:**")
                for element in template.get('elements', []):
                    st.write(f"• {element}")
                
                if st.button("Continue to Outline Creation →"):
                    st.session_state.current_step = 7
                    st.rerun()

def step_7_outline_creation():
    """Step 7: Outline Creation"""
    if 'structure' not in st.session_state.workflow_data or 'keywords' not in st.session_state.workflow_data:
        st.error("Please complete previous steps first")
        return
    
    st.markdown("### 📋 Outline Creation")
    
    topic = st.session_state.workflow_data['topic']
    structure = st.session_state.workflow_data['structure']
    keywords = st.session_state.workflow_data['keywords']
    
    st.write(f"**Topic:** {topic}")
    st.write(f"**Structure:** {structure}")
    
    if st.button("📋 Create Outline"):
        with st.spinner("Creating detailed outline..."):
            result = api_client.create_outline(topic, structure, keywords)
            
            if result and 'outline' in result:
                st.session_state.workflow_data['outline'] = result['outline']
                
                st.success("Outline created!")
                
                # Display outline
                outline = result['outline']
                st.subheader("Content Outline")
                
                st.write(f"**Title:** {outline.get('title', '')}")
                st.write(f"**Structure Type:** {outline.get('structure_type', '')}")
                st.write(f"**Estimated Words:** {outline.get('total_estimated_words', '')}")
                
                st.subheader("Sections")
                for i, section in enumerate(outline.get('sections', []), 1):
                    with st.expander(f"Section {i}: {section.get('heading', '')}"):
                        st.write(f"**Intent:** {section.get('intent', '')}")
                        st.write(f"**Tone:** {section.get('tone', '')}")
                        st.write(f"**Keywords:** {', '.join(section.get('keywords', []))}")
                        st.write(f"**Estimated Words:** {section.get('estimated_words', '')}")
                
                # Allow outline editing
                st.subheader("Edit Outline (Optional)")
                if st.checkbox("Enable outline editing"):
                    st.text_area("Edit outline JSON:", value=json.dumps(outline, indent=2), height=300)
                    if st.button("Update Outline"):
                        st.info("Outline editing functionality would be implemented here")
                
                if st.button("Continue to Blog Generation →"):
                    st.session_state.current_step = 8
                    st.rerun()

def step_8_blog_generation():
    """Step 8: Blog Generation"""
    if 'outline' not in st.session_state.workflow_data or 'keywords' not in st.session_state.workflow_data:
        st.error("Please complete previous steps first")
        return
    
    st.markdown("### ✍️ Blog Generation")
    
    outline = st.session_state.workflow_data['outline']
    keywords = st.session_state.workflow_data['keywords']
    
    # Brand voice selection
    brand_voice = st.selectbox(
        "Select brand voice:",
        ["professional, helpful, concise", "casual, friendly, conversational", "authoritative, expert, formal"],
        index=0
    )
    
    st.subheader("Generation Settings")
    col1, col2 = st.columns(2)
    with col1:
        target_words = st.number_input("Target word count:", min_value=500, max_value=5000, value=2000)
    with col2:
        include_faqs = st.checkbox("Include FAQs", value=True)
    
    if st.button("✍️ Generate Blog"):
        with st.spinner("Generating complete blog post... This may take a few minutes."):
            # Convert outline to string for API
            outline_str = json.dumps(outline) if isinstance(outline, dict) else str(outline)
            
            result = api_client.generate_blog(outline_str, keywords, brand_voice)
            
            if result:
                st.session_state.workflow_data['blog_content'] = result.get('blog_content', '')
                st.session_state.workflow_data['metadata'] = result.get('metadata', {})
                
                st.success("Blog generated successfully! 🎉")
                
                # Display metadata
                metadata = result.get('metadata', {})
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Word Count", metadata.get('word_count', 0))
                with col2:
                    st.metric("Reading Time", metadata.get('reading_time', '0 min'))
                with col3:
                    st.metric("Generated At", metadata.get('generated_at', ''))
                
                # Display blog content
                st.subheader("Generated Blog Post")
                st.markdown(result.get('blog_content', ''))
                
                # Export options
                st.subheader("Export Options")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📄 Copy to Clipboard"):
                        st.code(result.get('blog_content', ''))
                
                with col2:
                    if st.button("📥 Download Markdown"):
                        st.download_button(
                            "Download MD",
                            result.get('blog_content', ''),
                            file_name=f"blog_post_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown"
                        )
                
                with col3:
                    if st.button("🔄 Start New Blog"):
                        st.session_state.current_step = 1
                        st.session_state.workflow_data = {}
                        st.rerun()

def complete_workflow():
    """Complete workflow in one go"""
    st.header("🚀 Complete Workflow")
    st.markdown("Generate a complete blog post with all steps automated")
    
    with st.form("workflow_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            topic = st.text_input("Blog Topic *", placeholder="e.g., Digital Marketing Strategies")
            pillar = st.text_input("Content Pillar *", placeholder="e.g., Marketing")
            research_mode = st.selectbox("Research Mode", ["SERP", "RAG", "reference"])
        
        with col2:
            structure = st.selectbox("Content Structure", [
                "How-to Guide", "Listicle", "Thought leadership", "Deep-dive explainer"
            ])
            brand_voice = st.selectbox("Brand Voice", [
                "professional, helpful, concise",
                "casual, friendly, conversational", 
                "authoritative, expert, formal"
            ])
        
        submitted = st.form_submit_button("🚀 Generate Complete Blog")
        
        if submitted and topic and pillar:
            with st.spinner("Running complete workflow... This may take several minutes."):
                result = api_client.run_workflow(topic, pillar, research_mode, structure, brand_voice)
                
                if result:
                    st.success("Complete workflow executed! 🎉")
                    st.json(result)
                    
                    # Store session ID for export
                    if 'session_id' in result:
                        st.session_state.session_id = result['session_id']
                        st.info(f"Session ID: {result['session_id']} (saved for export)")

def export_results():
    """Export workflow results"""
    st.header("📤 Export Results")
    
    session_id = st.text_input("Session ID", value=st.session_state.get('session_id', ''))
    
    if not session_id:
        st.warning("Please enter a session ID or complete a workflow first")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📄 Export as JSON"):
            with st.spinner("Exporting results..."):
                result = api_client.export_results(session_id, "json")
                if result:
                    st.json(result)
                    st.download_button(
                        "Download JSON",
                        json.dumps(result, indent=2),
                        file_name=f"blog_results_{session_id}.json",
                        mime="application/json"
                    )
    
    with col2:
        if st.button("📝 Export as Markdown"):
            with st.spinner("Exporting results..."):
                # This would trigger file download from API
                st.info("Markdown export would download file directly from API")

# Navigation helpers
def reset_workflow():
    """Reset the workflow"""
    st.session_state.current_step = 1
    st.session_state.workflow_data = {}
    st.session_state.session_id = None

# Add reset button in sidebar
with st.sidebar:
    st.markdown("---")
    if st.button("🔄 Reset Workflow"):
        reset_workflow()
        st.rerun()
    
    # Show current workflow data
    if st.session_state.workflow_data:
        st.subheader("Current Progress")
        for key, value in st.session_state.workflow_data.items():
            if isinstance(value, str) and len(value) < 100:
                st.write(f"**{key}:** {value}")

if __name__ == "__main__":
    main()