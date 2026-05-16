from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

from tavily import TavilyClient

import base64
from ipywidgets import FileUpload
from IPython.display import display

import json
from typing import Dict, Any
from pathlib import Path
from pprint import pprint
from dotenv import load_dotenv

load_dotenv()




# Input an image

def png_to_base64_str(png_path: Path) -> str:
    with open(png_path, "rb") as f:
        img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    return img_b64

img_path = Path("notebooks/module-1/IMG_2932.png")
img_b64 = png_to_base64_str(img_path)



system_prompt_ingredient_extractor="You are an assistant that extracts ingredients from images of a kitchen counter. You will receive an image and you will return a list of ingredients you recognize, divided into categories such as 'vegetables', 'meats', 'spices', etc. You will only return the list of ingredients and categories without additional text."

class IngredientExtractorOutput(BaseModel):
    categories: Dict[str, list[str]]


agent_ingredient_extractor = create_agent(
    model="gpt-5-nano",
    tools=[],
    system_prompt=system_prompt_ingredient_extractor,
    checkpointer=InMemorySaver(),
    response_format=IngredientExtractorOutput
)

config_ingredient_extractor = {"configurable": {"thread_id": "2"}}

multimodal_question_ingredient_extractor = HumanMessage(content=[
    {"type": "text", "text": "Extract the list of ingredients you recognize from the attached image and divide them into categories. You will only return the list of ingredients and categories without additional text."},
    {"type": "image", "base64": img_b64, "mime_type": "image/png"}
])

ingredient_response = agent_ingredient_extractor.invoke(
    {"messages": [multimodal_question_ingredient_extractor]},
    config_ingredient_extractor
)

ingredient_list = ingredient_response["messages"][-1].content

# pprint(response_multimodal)
print(ingredient_response["messages"][-1].content)



### Tavily Client to search on the web

tavily_client = TavilyClient()

@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for information"""
    return tavily_client.search(query)

system_prompt_web_search = """

You are a personal chef. The user will give you a list of ingredients they have left over in their house.

Using the web search tool, search the web for recipes that can be made with the ingredients they have.

Return recipe suggestions and eventually the recipe instructions to the user, if requested.

"""

agent_recipe_creator = create_agent(
    model="gpt-5-nano",
    tools=[web_search],
    system_prompt=system_prompt_web_search,
    checkpointer=InMemorySaver()
)

config_recipe_creator = {"configurable": {"thread_id": "1"}}

response = agent_recipe_creator.invoke(
    {"messages": [HumanMessage(content=f"I have the following ingredients in my kitchen: {ingredient_list}")]},
    config_recipe_creator
)
# pprint(response)
print(response["messages"][-1].content)