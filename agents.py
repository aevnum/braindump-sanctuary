"""
agents.py (Day 2)

Contains the agentic logic for the Brain Dump Sanctuary.
This file provides the classes that app.py imports.

- QuestionAgent: Socratic question generation
- PerspectiveAgent: Multi-angle analysis
- SearchAgent: Web search + synthesis (MOCKED for Day 2)

This file uses simple, direct LLM calls for the Day 2 demo.
It is designed to be plug-and-play with app.py.

Future Work: These classes can be refactored into nodes
in a more complex LangGraph orchestration graph.
"""

import os
import json

# We'll use OpenAI as the LLM. You can swap this with Anthropic
# if you prefer, but you'll need to adjust the API call.
from openai import OpenAI

# --- API Key Configuration ---
# On Kaggle, add your "OPENAI_API_KEY" to the "Secrets" addon.
# os.environ.get() is the standard way to access it.
try:
    # Try to get from Kaggle secrets
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    API_KEY = secrets.get_secret("OPENAI_API_KEY")
except ImportError:
    # Fallback for local dev or other environments
    API_KEY = os.environ.get("OPENAI_API_KEY")
except Exception:
    API_KEY = None

if not API_KEY:
    print("WARNING: OPENAI_API_KEY not found. Please set it in your Kaggle secrets or environment.")
    # Set a placeholder to avoid crashing, but calls will fail.
    API_KEY = "YOUR_API_KEY_HERE" 

# Initialize the client globally or within each class
# Global is fine for this demo
client = OpenAI(api_key=API_KEY)


# ============== 1. Socratic Question Agent ==============

class QuestionAgent:
    """
    Generates Socratic questions to expand on a vague idea.
    """
    def __init__(self, model="gpt-4o-mini"):
        self.model = model
        self.system_prompt = """
You are a Socratic tutor. A user has a vague brain dump idea. 
Your goal is to generate 5 insightful, open-ended questions to help them 
explore this idea, discover its core, and understand their own curiosity.

- Do not answer the questions.
- Provide ONLY the list of questions.
- Return the questions as a JSON list of strings.

Example:
User Idea: "Why do we forget things we just read?"
Your Response:
[
    "What kind of material do you find you forget most often?",
    "What is your state of mind when you are reading?",
    "Are you trying to memorize facts, or understand a concept?",
    "What is the difference between remembering a fact and understanding an idea?",
    "How does this relate to the 'forgetting curve'?"
]
"""

    def generate_questions(self, idea: str) -> list[str]:
        """Generates Socratic questions for a given idea."""
        if API_KEY == "YOUR_API_KEY_HERE":
             return ["Error: OPENAI_API_KEY is not set.", "Please add it to your Kaggle secrets."]

        try:
            response = client.chat.completions.create(
                model=self.model,
                # Use JSON mode for reliable output
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"User Idea: \"{idea}\""}
                ]
            )
            questions_json_string = response.choices[0].message.content
            questions = json.loads(questions_json_string)
            
            # The LLM might return a dict {"questions": [...]}, or just [...]
            if isinstance(questions, dict):
                # Try to find the list value
                for key, value in questions.items():
                    if isinstance(value, list):
                        return value
            elif isinstance(questions, list):
                return questions
            
            return ["Error: Could not parse questions from LLM response."]

        except Exception as e:
            print(f"Error in QuestionAgent: {e}")
            return [
                "An error occurred while generating questions.",
                "Is your API key set correctly?",
                f"Details: {e}"
            ]


# ============== 2. Multi-Perspective Agent ==============

class PerspectiveAgent:
    """
    Analyzes a controversial topic from multiple angles.
    """
    def __init__(self, model="gpt-4o"): # Use a more powerful model
        self.model = model
        self.system_prompt = """
You are a multi-perspective analyst. A user has a topic,
which may be controversial or nuanced.
Your goal is to provide three distinct, concise viewpoints:
1.  **Skeptical View:** The critical or cautious perspective.
2.  **Optimistic View:** The positive or enthusiastic perspective.
3.  **Nuanced View:** A balanced, synthetic view that includes trade-offs or a "third way".

You MUST return a JSON object with exactly three keys:
"skeptical", "optimistic", and "nuanced".
Each value should be a short paragraph (2-4 sentences).
"""

    def analyze(self, topic: str) -> dict[str, str]:
        """Analyzes a topic from multiple perspectives."""
        if API_KEY == "YOUR_API_KEY_HERE":
            return {
                "skeptical": "Error: OPENAI_API_KEY not set.",
                "optimistic": "Please add your API key to Kaggle secrets.",
                "nuanced": "The agent cannot run without an API key."
            }

        try:
            response = client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Topic: \"{topic}\""}
                ]
            )
            perspectives_json_string = response.choices[0].message.content
            perspectives = json.loads(perspectives_json_string)
            
            # Ensure the keys are always present to prevent errors in app.py
            perspectives.setdefault('skeptical', 'No skeptical view generated.')
            perspectives.setdefault('optimistic', 'No optimistic view generated.')
            perspectives.setdefault('nuanced', 'No nuanced view generated.')
            
            return perspectives
            
        except Exception as e:
            print(f"Error in PerspectiveAgent: {e}")
            return {
                "skeptical": f"An error occurred: {e}",
                "optimistic": "Please check your API key and model access.",
                "nuanced": "The PerspectiveAgent failed to run."
            }


# ============== 3. Web Search Agent (MOCKED) ==============

class SearchAgent:
    """
    MOCK AGENT: Simulates web search and synthesis.
    This fulfills the "SearchAgent" requirement for Day 2
    without needing a live search API (like Tavily or SerpAPI).
    """
    def __init__(self):
        # In a real implementation, this would initialize
        # a Tavily or SerpAPI client.
        # e.g., self.tavily = TavilyClient(api_key=...)
        print("Initialized MOCK SearchAgent.")
        pass
    
    def deep_dive(self, topic: str) -> dict:
        """
        MOCK FUNCTION: Simulates a web search and synthesis.
        
        Returns a hard-coded summary and sources.
        """
        print(f"MOCK SEARCH: Deep dive for '{topic}'")
        
        # Simulate different responses for different topics
        if "dream" in topic.lower():
            return {
                "summary": (
                    "Dreams are a complex neurological phenomenon, primarily occurring during "
                    "REM sleep. Research suggests they are crucial for memory consolidation, "
                    "emotional regulation, and problem-solving. The 'fading' "
                    "is attributed to the brain's different neurochemical state during sleep, "
                    "which is not optimized for encoding new memories."
                ),
                "sources": [
                    {"title": "The Science of Dreaming - Scientific American", "url": "https://www.scientificamerican.com/article/the-science-of-dreaming/"},
                    {"title": "Why We Dream - Psychology Today", "url": "https://www.psychologytoday.com/us/basics/dreaming"}
                ]
            }
        elif "llm" in topic.lower():
            return {
                "summary": (
                    "The debate on LLM 'understanding' is central to AI research. "
                    "One view holds they are 'stochastic parrots,' brilliantly matching "
                    "statistical patterns without true comprehension. "
                    "The opposing view suggests that at their scale, these models "
                    "develop emergent, internal world models, representing a new "
                    "form of understanding."
                ),
                "sources": [
                    {"title": "On the Dangers of Stochastic Parrots - FAccT '21", "url": "https://dl.acm.org/doi/10.1145/3442188.3445922"},
                    {"title": "Sparks of AGI: Early experiments with GPT-4", "url": "https://arxiv.org/abs/2303.12712"}
                ]
            }
        else:
            # Generic fallback
            return {
                "summary": (
                    f"This is a mock summary about '{topic}'. This agent successfully simulated a "
                    "web search. In a real application, this text would be "
                    "dynamically generated by an LLM based on live search results "
                    "from a tool like Tavily or SerpAPI."
                ),
                "sources": [
                    {"title": "Mock Source 1 - Wikipedia", "url": "https://en.wikipedia.org/wiki/Main_Page"},
                    {"title": "Mock Source 2 - Example.com", "url": "https://example.com"}
                ]
            }

# --- How to refactor to LangGraph (Future Work) ---
"""
To implement your full "LangGraph" vision, you would:

1.  Define a State:
    class BrainDumpState(TypedDict):
        topic: str
        socratic_questions: list[str]
        perspectives: dict
        deep_dive: dict

2.  Create Nodes:
    - Each agent's method (e.g., `generate_questions`) becomes a node.
    - def question_node(state: BrainDumpState):
    -   questions = QuestionAgent().generate_questions(state['topic'])
    -   return {"socratic_questions": questions}
    - ... (similar nodes for perspective_node and search_node)

3.  Build the Graph:
    - workflow = StateGraph(BrainDumpState)
    - workflow.add_node("socratic", question_node)
    - workflow.add_node("perspectives", perspective_node)
    - workflow.add_node("search", search_node)
    - workflow.set_entry_point("socratic") # or a router
    - workflow.add_edge("socratic", "perspectives")
    - workflow.add_edge("perspectives", "search")
    - workflow.add_edge("search", END)

4.  Compile & Run:
    - app = workflow.compile()
    - # In app.py, you'd call:
    - # results = app.invoke({"topic": "Why do we dream?"})
    - # And then display results['socratic_questions'], etc.
    
This simple class-based approach is used for the Day 2 demo
as it directly matches your existing app.py implementation.
"""