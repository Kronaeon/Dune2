"""Creative Content Generation Example using RAG Toolkit.

This example demonstrates how to use RAG for creative tasks like
story generation, content adaptation, and style transfer.
"""

import sys
from pathlib import Path
from typing import List, Dict

sys.path.append(str(Path(__file__).parent.parent))

from rag_toolkit import (
    FileRetriever,
    DirectoryRetriever,
    FilteredRetriever,
    StyleAugmenter,
    TemplateAugmenter,
    ChainAugmenter,
    LlamaCppGenerator,
    APIGenerator,
    RAGPipeline,
    JSONParser,
    TemplateParser,
    ValidationParser,
    setup_logging,
    Document
)


def story_generation_example():
    """Generate a story based on retrieved context."""
    print("=== Story Generation Example ===\n")
    
    # Retrieve background information
    retriever = FileRetriever("data/story_context.txt")
    
    # Story generation template
    story_template = """Based on the following background information, create an engaging short story:

Background:
${context}

Requirements:
- Include all major elements from the background
- Create compelling characters with clear motivations
- Build tension and resolution
- Use vivid, descriptive language
- Aim for 500-800 words

Title: [Your Title Here]

Story:"""
    
    augmenter = TemplateAugmenter(template=story_template)
    generator = LlamaCppGenerator(
        model_path="/path/to/your/model.gguf",
        n_gpu_layers=35,
        max_tokens=1500,
        temperature=0.8  # Higher temperature for creativity
    )
    
    pipeline = RAGPipeline(
        retriever=retriever,
        augmenter=augmenter,
        generator=generator
    )
    
    result = pipeline.run()
    
    print("Generated Story:")
    print(result.output)


def style_transfer_example():
    """Transform content into different writing styles."""
    print("=== Style Transfer Example ===\n")
    
    # Original content
    retriever = FileRetriever("data/technical_article.txt")
    
    # Define different style templates
    styles = {
        "casual": """Transform this technical content into a casual, conversational blog post:

Original:
${context}

Make it:
- Friendly and approachable
- Use everyday language
- Add personal touches and examples
- Keep the core information but make it engaging

Casual Version:""",
        
        "academic": """Transform this content into a formal academic paper style:

Original:
${context}

Requirements:
- Use formal academic language
- Add proper citations format [Author, Year]
- Include abstract-style summary
- Maintain objectivity
- Use passive voice where appropriate

Academic Version:""",
        
        "creative": """Transform this content into a creative narrative:

Original:
${context}

Transform into:
- A story or metaphorical explanation
- Use analogies and vivid imagery
- Make abstract concepts concrete
- Engage emotions while conveying information

Creative Version:"""
    }
    
    # Generate in each style
    for style_name, template in styles.items():
        print(f"\n--- {style_name.title()} Style ---")
        
        augmenter = TemplateAugmenter(template=template)
        generator = LlamaCppGenerator(
            model_path="/path/to/your/model.gguf",
            n_gpu_layers=35,
            max_tokens=800,
            temperature=0.7
        )
        
        pipeline = RAGPipeline(
            retriever=retriever,
            augmenter=augmenter,
            generator=generator
        )
        
        result = pipeline.run()
        print(result.output[:500] + "..." if len(result.output) > 500 else result.output)


def content_expansion_example():
    """Expand brief notes into full content."""
    print("=== Content Expansion Example ===\n")
    
    # Brief notes or outline
    notes_content = """
    Topic: Benefits of Remote Work
    
    Main Points:
    - Flexibility in schedule
    - No commute time
    - Cost savings
    - Work-life balance
    - Global talent access
    
    Challenges:
    - Communication
    - Isolation
    - Time zones
    """
    
    # Create document from notes
    notes_doc = Document(
        content=notes_content,
        metadata={"type": "outline"},
        source="notes"
    )
    
    # Expansion template
    expansion_template = """Expand these notes into a comprehensive article:

Notes:
${context}

Create a full article that:
1. Has an engaging introduction
2. Develops each point with examples and evidence
3. Addresses challenges with solutions
4. Includes transitions between sections
5. Ends with a compelling conclusion

Full Article:"""
    
    # Manual retriever that returns our notes
    class NotesRetriever(BaseRetriever):
        def retrieve(self, query=None):
            return [notes_doc]
    
    retriever = NotesRetriever()
    augmenter = TemplateAugmenter(template=expansion_template)
    generator = LlamaCppGenerator(
        model_path="/path/to/your/model.gguf",
        n_gpu_layers=35,
        max_tokens=2000,
        temperature=0.7
    )
    
    pipeline = RAGPipeline(
        retriever=retriever,
        augmenter=augmenter,
        generator=generator
    )
    
    result = pipeline.run()
    
    print("Expanded Article:")
    print(result.output)


def dialogue_generation_example():
    """Generate dialogue based on character descriptions."""
    print("=== Dialogue Generation Example ===\n")
    
    # Character descriptions
    retriever = DirectoryRetriever(
        directory="data/characters",
        pattern="*.txt"
    )
    
    # Dialogue template
    dialogue_template = """Based on these character descriptions, create a realistic dialogue:

Characters:
${context}

Scenario: The characters meet at a coffee shop to discuss a new business opportunity.

Requirements:
- Each character should speak in their unique voice
- Include personality traits in their dialogue
- Create natural conflict and resolution
- Show character development
- Format as a screenplay

DIALOGUE:"""
    
    augmenter = TemplateAugmenter(
        template=dialogue_template,
        config=AugmenterConfig(include_metadata=True)
    )
    
    generator = LlamaCppGenerator(
        model_path="/path/to/your/model.gguf",
        n_gpu_layers=35,
        max_tokens=1500,
        temperature=0.9  # High temperature for varied dialogue
    )
    
    pipeline = RAGPipeline(
        retriever=retriever,
        augmenter=augmenter,
        generator=generator
    )
    
    result = pipeline.run()
    
    print("Generated Dialogue:")
    print(result.output)


def creative_prompt_chain_example():
    """Chain multiple creative prompts for complex generation."""
    print("=== Creative Prompt Chain Example ===\n")
    
    # Initial context
    retriever = FileRetriever("data/world_building.txt")
    
    # First stage: Generate setting
    setting_template = """Based on this world-building information, create a detailed setting:

${context}

Describe:
- The physical environment
- The atmosphere and mood
- Unique features of this world
- Sensory details (sights, sounds, smells)

Setting Description:"""
    
    # Second stage: Add characters
    character_template = """Given this setting, introduce interesting characters:

${context}

Create 2-3 characters who:
- Fit naturally in this setting
- Have clear motivations
- Have interesting conflicts
- Are memorable and unique

Character Descriptions:"""
    
    # Third stage: Create plot
    plot_template = """With this setting and these characters, create a compelling plot:

${context}

Develop:
- An inciting incident
- Rising action
- Climax
- Resolution
- Theme

Plot Outline:"""
    
    # Create chain of augmenters
    setting_augmenter = TemplateAugmenter(template=setting_template)
    character_augmenter = TemplateAugmenter(template=character_template)
    plot_augmenter = TemplateAugmenter(template=plot_template)
    
    generator = LlamaCppGenerator(
        model_path="/path/to/your/model.gguf",
        n_gpu_layers=35,
        max_tokens=800,
        temperature=0.8
    )
    
    # Stage 1: Generate setting
    print("Stage 1: Generating setting...")
    setting_pipeline = RAGPipeline(retriever, setting_augmenter, generator)
    setting_result = setting_pipeline.run()
    
    # Stage 2: Generate characters based on setting
    print("Stage 2: Creating characters...")
    setting_doc = Document(
        content=setting_result.output,
        metadata={"stage": "setting"},
        source="generated"
    )
    
    class SingleDocRetriever(BaseRetriever):
        def __init__(self, doc):
            super().__init__()
            self.doc = doc
        def retrieve(self, query=None):
            return [self.doc]
    
    character_pipeline = RAGPipeline(
        SingleDocRetriever(setting_doc),
        character_augmenter,
        generator
    )
    character_result = character_pipeline.run()
    
    # Stage 3: Generate plot
    print("Stage 3: Developing plot...")
    combined_content = f"Setting:\n{setting_result.output}\n\nCharacters:\n{character_result.output}"
    combined_doc = Document(
        content=combined_content,
        metadata={"stage": "combined"},
        source="generated"
    )
    
    plot_pipeline = RAGPipeline(
        SingleDocRetriever(combined_doc),
        plot_augmenter,
        generator
    )
    plot_result = plot_pipeline.run()
    
    # Display results
    print("\n=== Generated Story Elements ===")
    print("\nSETTING:")
    print(setting_result.output[:400] + "...")
    print("\nCHARACTERS:")
    print(character_result.output[:400] + "...")
    print("\nPLOT:")
    print(plot_result.output)


def recipe_adaptation_example():
    """Adapt recipes based on dietary restrictions."""
    print("=== Recipe Adaptation Example ===\n")
    
    # Original recipe
    retriever = FileRetriever("data/original_recipe.txt")
    
    # Adaptation templates for different diets
    adaptations = {
        "vegan": {
            "requirements": "vegan (no animal products)",
            "substitutions": "plant-based alternatives for all animal products"
        },
        "gluten_free": {
            "requirements": "gluten-free",
            "substitutions": "gluten-free flour and alternatives"
        },
        "keto": {
            "requirements": "ketogenic (low-carb, high-fat)",
            "substitutions": "low-carb alternatives, increase healthy fats"
        }
    }
    
    adaptation_template = """Adapt this recipe for ${requirements} dietary requirements:

Original Recipe:
${context}

Requirements:
- Make it completely ${requirements}
- Use ${substitutions}
- Maintain the dish's essence and flavor profile
- Adjust cooking times/temperatures if needed
- Include nutritional notes

Adapted Recipe:

TITLE: ${requirements} Version

INGREDIENTS:

INSTRUCTIONS:

NUTRITIONAL NOTES:
"""
    
    # Generate adaptations
    for diet_name, diet_info in adaptations.items():
        print(f"\n--- {diet_name.replace('_', ' ').title()} Adaptation ---")
        
        augmenter = TemplateAugmenter(template=adaptation_template)
        generator = LlamaCppGenerator(
            model_path="/path/to/your/model.gguf",
            n_gpu_layers=35,
            max_tokens=800,
            temperature=0.6
        )
        
        pipeline = RAGPipeline(
            retriever=retriever,
            augmenter=augmenter,
            generator=generator
        )
        
        result = pipeline.run(
            requirements=diet_info["requirements"],
            substitutions=diet_info["substitutions"]
        )
        
        # Parse structured output
        parser = TemplateParser(
            template="TITLE: {{title}}\n\nINGREDIENTS:\n{{ingredients}}\n\nINSTRUCTIONS:\n{{instructions}}"
        )
        
        parsed = parser.parse(result.output)
        if parsed.is_valid:
            print(f"Title: {parsed.data.get('title', 'N/A')}")
            print("First few ingredients:", parsed.data.get('ingredients', '')[:200] + "...")
        else:
            print(result.output[:400] + "...")


if __name__ == "__main__":
    # Set up logging
    setup_logging(level="INFO")
    
    # Create example data
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Create example story context
    story_context = data_dir / "story_context.txt"
    if not story_context.exists():
        story_context.write_text("""
Setting: A remote lunar research station in 2157
Main Character: Dr. Sarah Chen, a xenobiologist who discovered unusual crystalline formations
Conflict: The crystals appear to be growing and exhibiting signs of consciousness
Stakes: The safety of the station and potentially Earth itself
Atmosphere: Isolation, wonder, growing unease
Key Elements: 
- The station has been cut off from Earth communications for 3 days
- Only 5 crew members remain on the station
- The crystals emit a faint humming sound that changes frequency
- Previous attempts to study similar formations ended mysteriously
""")
    
    # Create world building example
    world_building = data_dir / "world_building.txt"
    if not world_building.exists():
        world_building.write_text("""
World: Neo-Venice, 2089
- A floating city built on the ruins of the original Venice after the Great Flood
- Powered by tidal generators and solar collectors
- Connected by a network of canal-streets and mag-lev gondolas
- Architecture blends classic Venetian style with bioengineered coral structures
- Population: 2.3 million permanent residents, 5 million tourists annually
- Governed by an AI Council and human representatives
- Known for its bioluminescent gardens and underwater districts
- Home to the Global Climate Refugee Integration Center
""")
    
    print("Note: Please update model_path in the examples before running.\n")
    
    # Run examples (uncomment to run)
    try:
        # story_generation_example()
        # style_transfer_example()
        # content_expansion_example()
        # dialogue_generation_example()
        # creative_prompt_chain_example()
        # recipe_adaptation_example()
        
        print("To run examples, uncomment the desired example function calls above.")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to set up your model path or API credentials.")