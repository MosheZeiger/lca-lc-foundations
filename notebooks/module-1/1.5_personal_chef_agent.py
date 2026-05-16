"""
Kitchen Ingredient Extractor & Recipe Finder
=============================================
Extracts ingredients from a kitchen counter image using a vision-capable agent,
then searches the web for matching recipes using Tavily.
"""

import base64
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel
from tavily import TavilyClient

load_dotenv()

DEFAULT_IMAGE_PATH = Path("notebooks/module-1/IMG_2932.png")

SYSTEM_PROMPT_INGREDIENT_EXTRACTOR = (
    "You are an assistant that extracts ingredients from images of a kitchen counter. "
    "You will receive an image and you will return a list of ingredients you recognize, "
    "divided into categories such as 'vegetables', 'meats', 'spices', etc. "
    "You will only return the list of ingredients and categories without additional text."
)

SYSTEM_PROMPT_RECIPE_CREATOR = (
    "You are a personal chef. The user will give you a list of ingredients "
    "they have left over in their house.\n\n"
    "Using the web search tool, search the web for recipes that can be made "
    "with the ingredients they have.\n\n"
    "Return recipe suggestions and eventually the recipe instructions "
    "to the user, if requested."
)

MODEL_NAME = "gpt-5-nano"


class IngredientExtractorOutput(BaseModel):
    """Structured output for the ingredient extractor agent."""
    categories: Dict[str, list[str]]


tavily_client = TavilyClient()

@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for information."""
    return tavily_client.search(query)
    

def load_image_as_base64(image_path: Path) -> str:
    """Read a PNG file and return its content as a base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
    

def build_multimodal_message(text: str, image_b64: str, mime_type: str = "image/png") -> HumanMessage:
    """Create a multimodal HumanMessage containing text and an image."""
    return HumanMessage(content=[
        {"type": "text", "text": text},
        {"type": "image", "base64": image_b64, "mime_type": mime_type}
    ])

def create_ingredient_extractor_agent():
    """Create an agent that extracts ingredients from kitchen or fridge images."""
    return create_agent(
        model=MODEL_NAME,
        tools=[],
        system_prompt=SYSTEM_PROMPT_INGREDIENT_EXTRACTOR,
        # checkpointer=InMemorySaver(),
        response_format=IngredientExtractorOutput
    )

class RecipeCreatorOutput(BaseModel):
    """Structured output for the recipe creator agent."""
    recipes: list[Dict[str, Any]]  # Adjust the structure based on expected recipe data

def create_recipe_creator_agent():
    """Create an agent that searches the web for recipes based on ingredients."""
    return create_agent(
        model=MODEL_NAME,
        tools=[web_search],
        system_prompt=SYSTEM_PROMPT_RECIPE_CREATOR,
        # checkpointer=InMemorySaver(),
        response_format=RecipeCreatorOutput
    )


ingredient_agent = create_ingredient_extractor_agent()

def extract_ingredients_from_image(image_path: Path) -> Dict[str, list[str]]:
    """Use the ingredient extractor agent to extract ingredients from an image."""
    img_b64 = load_image_as_base64(image_path)
    agent = ingredient_agent
    config = {"configurable": {"thread_id": "1"}}

    message = build_multimodal_message(
        text=("Extract the list of ingredients you recognize from the attached image and divide them into categories. You will only return the list of ingredients and categories without additional text."),
        image_b64=img_b64
    )

    response = agent.invoke({"messages": [message]}, config)
    return response["messages"][-1].content

recipe_agent = create_recipe_creator_agent()

def find_recipes(ingredient_list: str) -> str:
    """Run the recipe creator agent to find recipes matching the given ingredients."""
    agent = recipe_agent
    config = {"configurable": {"thread_id": "2"}}

    message = HumanMessage(
        content=f"I have the following ingredients in my kitchen: {ingredient_list}"
    )

    response = agent.invoke({"messages": [message]}, config)
    return response["messages"][-1].content

def main() -> None:
    """Run the full pipeline: extract ingredients → find recipes."""
    ingredient_list = extract_ingredients_from_image(DEFAULT_IMAGE_PATH)
    print("\n\n------ Extracted Ingredients ------\n\n")
    print(ingredient_list)
    print("\n\n------ Recipe Suggestions ------\n\n")
    recipes = find_recipes(ingredient_list) 
    print(recipes)

if __name__ == "__main__":
    main()