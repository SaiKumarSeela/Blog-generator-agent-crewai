
# RAG Related
EMBEDDING_MODEL_RAG:str = 'sentence-transformers/all-MiniLM-L6-v2'
CHUNK_SIZE:int = 800
CHUNK_OVERLAP: int = 100

LLM_MODEL="gemini/gemini-1.5-flash"

# API Related
API_HOST="0.0.0.0"
API_PORT=8085
STREAMLIT_PORT=8501
API_BASE_URL = f"http://localhost:{API_PORT}"


LAYOUT_TEMPLATES = {
    "How-to Guide": {
        "H1": "How to [Achieve Goal/Use Tool]",
        "H2": [
            "Introduction (Problem Statement & Importance)",
            "Step 1: [Action Step]",
            "Step 2: [Action Step]",
            "Step 3: [Action Step]",
            "Tips & Best Practices",
            "Conclusion (Summary + CTA)"
        ],
        "CTA": "Encourage user to try product/service for faster results",
        "FAQs": ["What is X?", "How long does Y take?", "Can I do Z without prior experience?"]
    },
    "Listicle": {
        "H1": "Top [Number] Ways to [Achieve Goal]",
        "H2": [
            "Introduction (Why this matters)",
            "Item 1: [Title + Explanation]",
            "Item 2: [Title + Explanation]",
            "Item 3: [Title + Explanation]",
            "Wrap-up (Key Takeaways)"
        ],
        "CTA": "Suggest exploring more via product/service",
        "FAQs": ["Which method works best?", "Can these steps be combined?", "Is this beginner-friendly?"]
    },
    "Thought Leadership": {
        "H1": "[Big Idea/Trend] and Its Impact on [Industry]",
        "H2": [
            "Introduction (Author’s POV)",
            "Current Landscape",
            "Emerging Trends",
            "Opportunities & Risks",
            "Personal Insights",
            "Conclusion (Vision + CTA)"
        ],
        "CTA": "Position brand as authority; invite reader engagement",
        "FAQs": ["Why is this trend important?", "What’s next in this space?", "How can businesses prepare?"]
    },
    "Deep-dive Explainer": {
        "H1": "Everything You Need to Know About [Topic]",
        "H2": [
            "Introduction (Context & Relevance)",
            "Background/History",
            "Key Concepts Explained",
            "Detailed Breakdown (H3 subsections allowed)",
            "Case Studies / Examples",
            "Summary & Next Steps"
        ],
        "CTA": "Encourage further exploration with product/service",
        "FAQs": ["What is the core concept?", "How does this apply to real-world use?", "Where can I learn more?"]
    }
}
