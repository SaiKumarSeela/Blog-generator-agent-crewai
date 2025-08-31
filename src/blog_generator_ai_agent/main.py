#!/usr/bin/env python
import sys
import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from blog_generator_ai_agent.crew import BlogGeneratorCrew

def run():
    """
    Run the Blog Generator Crew with default inputs.
    This function is called when using 'crewai run' command.
    """
    
    # Default inputs showcasing enhanced capabilities
    inputs = {
        'pillar': 'Customer Loyalty and Rewards',
        'topic': 'How to Build an Effective Customer Loyalty Program',
        'research_method': 'SERP',  # Options: SERP, RAG, reference
        'structure_type': 'How-to Guide',
        'primary_keyword': 'customer loyalty program',
        'keywords': {
            'primary': 'customer loyalty program',
            'secondary': ['loyalty rewards', 'customer retention', 'reward program benefits', 'customer loyalty strategy']
        },
        'target_audience': 'Business owners and marketing managers',
        'content_length': 'comprehensive',  # short, medium, comprehensive
        'seo_focus': True,
        'competitor_analysis': True
    }
    
    # Initialize and run the crew
    blog_crew = BlogGeneratorCrew()
    
    print("🚀 Starting Enhanced Blog Generator Crew...")
    print(f"📝 Topic: {inputs['topic']}")
    print(f"🎯 Primary Keyword: {inputs['primary_keyword']}")
    print(f"🔍 Research Method: {inputs['research_method']}")
    print(f"📊 Structure Type: {inputs['structure_type']}")
    print("-" * 60)
    
    try:
        result = blog_crew.run_workflow(inputs)
        
        print("\n✅ Blog Generation Completed!")
        print("📄 Check 'generated_blog.md' for the output")
        print(f"📊 Result Type: {type(result)}")
        
        # Print summary of results
        if hasattr(result, 'raw'):
            print("\n📋 Generation Summary:")
            print(f"   - Tasks completed: {len(result.tasks_output) if hasattr(result, 'tasks_output') else 'N/A'}")
            print(f"   - Tokens used: {getattr(result, 'token_usage', 'N/A')}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error running crew: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def run_with_custom_inputs(topic: str, pillar: str = "Content Marketing", 
                          research_method: str = "SERP", structure_type: str = "How-to Guide",
                          **kwargs):
    """
    Run the crew with custom inputs
    
    Args:
        topic: Main topic for the blog
        pillar: Content pillar/category
        research_method: 'SERP', 'RAG', or 'reference'
        structure_type: Type of content structure
        **kwargs: Additional optional parameters
    """
    
    # Extract primary keyword from topic
    primary_keyword = kwargs.get('primary_keyword', topic.lower().replace(' ', ' '))
    
    inputs = {
        'pillar': pillar,
        'topic': topic,
        'research_method': research_method,
        'structure_type': structure_type,
        'primary_keyword': primary_keyword,
        'keywords': {
            'primary': primary_keyword,
            'secondary': kwargs.get('secondary_keywords', [
                f"how to {topic.lower()}", 
                f"{topic.lower()} guide",
                f"best {topic.lower()}"
            ])
        },
        'target_audience': kwargs.get('target_audience', 'General audience'),
        'content_length': kwargs.get('content_length', 'comprehensive'),
        'seo_focus': kwargs.get('seo_focus', True),
        'competitor_analysis': kwargs.get('competitor_analysis', True)
    }
    
    print(f"🎯 Running with custom topic: {topic}")
    print(f"🔍 Research method: {research_method}")
    print(f"📊 Content structure: {structure_type}")
    
    blog_crew = BlogGeneratorCrew()
    return blog_crew.run_workflow(inputs)

def run_research_mode_demo():
    """Demonstrate the enhanced research capabilities"""
    
    blog_crew = BlogGeneratorCrew()
    topic = "content marketing strategies"
    
    print("🔬 Research Mode Demonstration")
    print("=" * 50)
    
    # Demo SERP Analysis
    print("\n1. 📊 SERP Analysis Mode:")
    try:
        serp_results = blog_crew.run_research_mode("serp", topic, num_results=5)
        serp_data = json.loads(serp_results)
        print(f"   Found {serp_data.get('total_found', 0)} SERP results")
        if serp_data.get('findings'):
            print(f"   Top result: {serp_data['findings'][0].get('title', 'N/A')}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Demo RAG Retrieval (if knowledge base has content)
    print("\n2. 🧠 RAG Knowledge Retrieval:")
    try:
        rag_results = blog_crew.run_research_mode("rag", topic, top_k=3)
        rag_data = json.loads(rag_results)
        print(f"   Found {rag_data.get('total_found', 0)} knowledge base results")
        if rag_data.get('findings'):
            print(f"   Top finding confidence: {rag_data['findings'][0].get('confidence', 'N/A')}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Demo Competitor Analysis
    print("\n3. 🏆 Competitor Analysis:")
    try:
        comp_results = blog_crew.analyze_competitor_content(topic, num_competitors=3)
        comp_data = json.loads(comp_results)
        print(f"   Analyzed {comp_data.get('total_analyzed', 0)} competitor articles")
        if comp_data.get('competitors'):
            avg_words = comp_data.get('comparison_grid', {}).get('average_metrics', {}).get('word_count', 'N/A')
            print(f"   Average competitor word count: {avg_words}")
    except Exception as e:
        print(f"   Error: {e}")

def add_knowledge_base_content(blog_crew: BlogGeneratorCrew, 
                              pdf_paths: Optional[List[str]] = None,
                              markdown_paths: Optional[List[str]] = None,
                              urls: Optional[List[str]] = None):
    """Add content to the knowledge base"""
    
    print("📚 Adding content to knowledge base...")
    
    results = blog_crew.add_knowledge_documents(
        pdf_paths=pdf_paths,
        markdown_paths=markdown_paths,
        urls=urls
    )
    
    for content_type, result in results.items():
        if result:
            print(f"   {content_type}: Added {result.get('added', 0)}, Failed {result.get('failed', 0)}")
            if result.get('errors'):
                print(f"     Errors: {result['errors'][:2]}...")  # Show first 2 errors

def interactive_mode():
    """Interactive mode for blog generation"""
    
    print("🚀 Interactive Blog Generator")
    print("=" * 40)
    
    # Get user inputs
    topic = input("📝 Enter your blog topic: ").strip()
    if not topic:
        print("❌ Topic is required!")
        return
    
    pillar = input("📂 Content pillar (optional): ").strip() or "General"
    
    print("\n🔍 Research Methods:")
    print("  1. SERP - Search engine analysis")
    print("  2. RAG - Knowledge base retrieval") 
    print("  3. reference - Process reference articles")
    
    research_choice = input("Choose research method (1-3): ").strip()
    research_map = {'1': 'SERP', '2': 'RAG', '3': 'reference'}
    research_method = research_map.get(research_choice, 'SERP')
    
    print("\n📊 Content Structures:")
    print("  1. How-to Guide")
    print("  2. Listicle")
    print("  3. Case Study")
    print("  4. Comparison")
    
    structure_choice = input("Choose structure (1-4): ").strip()
    structure_map = {
        '1': 'How-to Guide', 
        '2': 'Listicle', 
        '3': 'Case Study', 
        '4': 'Comparison'
    }
    structure_type = structure_map.get(structure_choice, 'How-to Guide')
    
    # Ask about knowledge base
    kb_choice = input("\n📚 Add content to knowledge base? (y/n): ").lower()
    
    blog_crew = BlogGeneratorCrew()
    
    if kb_choice == 'y':
        urls_input = input("🌐 URLs (comma-separated): ").strip()
        if urls_input:
            urls = [url.strip() for url in urls_input.split(',')]
            add_knowledge_base_content(blog_crew, urls=urls)
    
    # Run the workflow
    print(f"\n🚀 Generating blog: '{topic}'")
    result = run_with_custom_inputs(
        topic=topic,
        pillar=pillar,
        research_method=research_method,
        structure_type=structure_type
    )
    
    if result:
        print("\n✅ Blog generated successfully!")
        print("📄 Check 'generated_blog.md' for your content")
    else:
        print("\n❌ Blog generation failed. Check the logs above.")

def main():
    """Main entry point with command line argument handling"""
    
    if len(sys.argv) == 1:
        # No arguments - run default
        run()
    elif sys.argv[1] == '--interactive' or sys.argv[1] == '-i':
        # Interactive mode
        interactive_mode()
    elif sys.argv[1] == '--research-demo' or sys.argv[1] == '-r':
        # Research mode demonstration
        run_research_mode_demo()
    elif sys.argv[1] == '--help' or sys.argv[1] == '-h':
        # Help
        print("🚀 Blog Generator AI Agent")
        print("=" * 40)
        print("Usage:")
        print("  python main.py                    # Run with default inputs")
        print("  python main.py --interactive      # Interactive mode")
        print("  python main.py --research-demo    # Demo research capabilities")
        print("  python main.py 'Custom Topic'     # Run with custom topic")
        print("\nResearch Methods:")
        print("  - SERP: Search engine result analysis")
        print("  - RAG: Knowledge base retrieval")
        print("  - reference: Process reference articles")
        print("\nEnvironment Variables Required:")
        print("  - GEMINI_API_KEY: Google Gemini API key")
        print("  - SERP_API_KEY: SERP API key (for web search)")
        print("  - FIRECRAWL_API_KEY: FireCrawl API key (optional)")
    else:
        # Custom topic from command line
        custom_topic = " ".join(sys.argv[1:])
        print(f"🎯 Running with custom topic: {custom_topic}")
        run_with_custom_inputs(custom_topic)

if __name__ == "__main__":
    main()