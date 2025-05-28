import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import openai
from dotenv import find_dotenv, load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.errors import GraphRecursionError

from geobenchx.dataclasses import Task, Solution, Step
from geobenchx.prompts import RULES_PROMPT, SYSTEM_PROMPT
from geobenchx.constants import (
    MODEL_GEMINI,
    MODEL_GEMINI_ADV,    
    MODEL_GPT_4o,
    MODEL_GPT_41,
    MODEL_GPT_mini,
    MODEL_O3,
    MODEL_O4,
    MODEL_CLAUDE,
    MODEL_CLAUDE_mini,
    MODEL_CLAUDE_ADV3,
    MODEL_CLAUDE_ADV4
    
)
from geobenchx.tools import (
    State,
    get_unique_values_tool, 
    load_data_tool, 
    load_geodata_tool, 
    make_choropleth_map_tool, 
    make_bivariate_map_tool,
    merge_dataframes_tool, 
    filter_categorical_tool, 
    filter_numerical_tool,
    select_features_by_spatial_relationship_tool,
    filter_points_by_raster_values_tool,
    create_buffer_tool,
    get_raster_path_tool,
    get_raster_description_tool,
    get_values_from_raster_with_geometries_tool,
    analyze_raster_overlap_tool,
    calculate_line_lengths_tool,
    calculate_columns_tool,
    scale_column_by_value_tool,
    make_heatmap_tool,
    visualize_geographies_tool,
    get_centroids_tool,
    generate_contours_display_tool,
    reject_task_tool,
    calculate_column_statistics_tool
)
from geobenchx.utils import get_solution_code

_ = load_dotenv(find_dotenv())

openai.api_key  = os.getenv('OPENAI_API_KEY')

google_api_key = os.getenv("GOOGLE_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

tools = [
    load_data_tool, 
    load_geodata_tool,
    make_choropleth_map_tool,
    make_bivariate_map_tool,
    merge_dataframes_tool,
    get_unique_values_tool,
    filter_categorical_tool,
    filter_numerical_tool,
    select_features_by_spatial_relationship_tool,
    filter_points_by_raster_values_tool,
    create_buffer_tool,
    get_raster_path_tool,
    get_raster_description_tool,
    get_values_from_raster_with_geometries_tool,
    analyze_raster_overlap_tool,
    calculate_line_lengths_tool,
    calculate_columns_tool,
    scale_column_by_value_tool,
    make_heatmap_tool,
    visualize_geographies_tool,
    get_centroids_tool,
    generate_contours_display_tool,
    reject_task_tool,
    calculate_column_statistics_tool
]

def execute_task(task_text: str, temperature: float = 0, model: str = MODEL_GPT_4o, max_steps: int = 25, capture_history=False):

    # Initialize conversation history if capturing
    conversation_history = [] if capture_history else None

    solution_steps = []
    # token_total = []
    input_tokens = []
    output_tokens = []

    if model in [MODEL_CLAUDE, MODEL_CLAUDE_mini, MODEL_CLAUDE_ADV3, MODEL_CLAUDE_ADV4]:
        llm = ChatAnthropic(model=model, temperature=temperature)
    elif model in [MODEL_GPT_4o, MODEL_GPT_41, MODEL_GPT_mini]:
        llm = ChatOpenAI(model=model, temperature=temperature)
    elif model in [MODEL_O3, MODEL_O4]:
        llm = ChatOpenAI(model=model, temperature=None)
    elif model in [MODEL_GEMINI, MODEL_GEMINI_ADV]:
        llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)
    else:
        raise ValueError("Model is outside the predetermined list")

    graph = create_react_agent(llm, tools=tools, state_schema=State, state_modifier=SYSTEM_PROMPT + RULES_PROMPT) 

    inputs = {
        "messages": [("user", task_text)],
        "data_store": {},
        "image_store": [],
        "html_store":[],
        "visualize": True
        }

    config = {
        "max_concurrency": 1,
        "recursion_limit": max_steps
        }
    
    try:
        for s in graph.stream(inputs, stream_mode="values", config=config):

            message = s["messages"][-1]
            if hasattr(message, 'usage_metadata') and message.usage_metadata:
                # token_total.append(message.usage_metadata['total_tokens'])
                input_tokens.append(message.usage_metadata['input_tokens'])
                output_tokens.append(message.usage_metadata['output_tokens'])                

            if isinstance(message, tuple):
                print(message)
            else:
                message.pretty_print() 

            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:     
                    step = Step(function_name = tool_call['name'], arguments = tool_call['args']) # to keep compatibility with get_solution_code() and the previous agent
                    #save step to solution 
                    solution_steps.append(step)

            if capture_history:
                conversation_history.append({"type": message.type, "content": message.content})

            # Extract any images from state and add to conversation history
            if capture_history and "image_store" in s and s["image_store"]:
                for img_data in s["image_store"]:
                    conversation_history.append({
                        'type': 'image',
                        'content': img_data["base64"],
                        'description': img_data.get("description", "Visualization")
                    })
                s["image_store"].clear() 

            # Extract any html from state and add to conversation history
            if capture_history and "html_store" in s and s["html_store"]:
                for html_item in s["html_store"]:
                    conversation_history.append({
                        'type': 'interactive_map',
                        'content': html_item["html"],
                        'description': html_item.get("description", "Interactive Map")
                    })
                s["html_store"].clear()                 

    except GraphRecursionError as e:
        print(f"Maximum recursion depth reached: {e}")                    

    solution = Solution(steps = solution_steps)
    
    return solution, input_tokens, output_tokens, conversation_history

