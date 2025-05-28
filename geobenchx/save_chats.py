import os
import re
import html
import gc
import json
from pathlib import Path
from geobenchx.utils import get_solution_code

def format_content_for_display(content):
    """Format complex content structures for better readability in HTML"""

    if isinstance(content, dict):
        # Handle dictionary content types
        if 'text' in content:
            # If there's a text field, primarily display that
            return content['text']
        elif 'type' in content and content['type'] == 'tool_use':
            # Format tool use in a more readable way
            tool_info = f"Tool: {content.get('name', 'Unknown')}"
            if 'input' in content and content['input']:
                tool_info += f"\nInputs: {json.dumps(content['input'], indent=2)}"
            return tool_info
        else:
            # For other dictionary types, format as pretty JSON
            return json.dumps(content, indent=2)
    elif isinstance(content, list):
        # For lists, convert each item and join with newlines
        return '\n'.join(str(format_content_for_display(item)) for item in content)
    elif content is None:
        return ""
    else:
        # For simple types, just convert to string
        return str(content)
        

# Check if content is a list and convert it to a string if needed
def safe_escape(content):
    if isinstance(content, list):
        # Join list elements with newlines
        return html.escape('\n'.join(str(item) for item in content))
    elif content is None:
        return ""
    else:
        # Convert to string and escape
        return html.escape(str(content))

def save_conversation_to_html(task, conversation_history, run_folder):
    """
    Save the conversation history for a task to an HTML file.
    
    Args:
        task: The task object
        conversation_history: List of message exchanges and images
        run_folder: Path to the run folder
        
    Returns:
        str: Message indicating where the file was saved
    """
    try:
        html_parts = []

        html_parts.append(f"""<!DOCTYPE html>
    <html>
    <head>
        <title>Task {task.task_ID} Conversation</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .message {{ margin-bottom: 20px; padding: 10px; border-radius: 5px; }}
            .user {{ background-color: #f0f0f0; padding: 15px;}}
            .assistant {{ background-color: #e6f7ff; padding: 15px;}}
            .tool-call {{ background-color: #fff3e0; padding: 15px; overflow-x: auto; }}
            .tool-result {{ background-color: #e8f5e9; padding: 15px; overflow-x: auto; }}
            .image {{background-color: #f9f9f9; border: 1px solid #ddd; padding: 15px; border-radius: 5px; margin-bottom: 20px;}}
            .interactive-map {{ background-color: #f0f8ff; padding: 15px; }}
            .map-container {{ margin-top: 10px; }}
            img {{ max-width: 100%; }}
            pre {{ white-space: pre-wrap; }}
            h1, h2 {{ color: #333; }}
            .solution {{ background-color: #e8f5e9; padding: 15px; border-radius: 5px; margin-top: 20px; }}
            .solution pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 3px; }}
            ul.message {{margin-bottom: 20px; padding: 15px; border-radius: 5px;}}
        </style>
    </head>
    <body>
        <h1>Task ID: {task.task_ID}</h1>
        <div class="message">
        <strong>Task Description:</strong>
            <pre>{html.escape(task.task_text)}</pre>
        </div>
        
        <h2>Conversation:</h2>
    """)
        # Add each message in the conversation history
        for entry in conversation_history:
            if entry['type'] == 'human':
                content = format_content_for_display(entry['content'])
                html_parts.append(f"""<div class="message user">
        <strong>User Message:</strong>
        <pre>{content}</pre>
    </div>
    """)
                del content
            elif entry['type'] == 'ai':
                content = format_content_for_display(entry['content'])
                html_parts.append(f"""<div class="message assistant">
        <strong>AI Message:</strong>
        <pre>{content}</pre>
    </div>
    """)
                del content
            elif entry['type'] == 'tool':
                content = format_content_for_display(entry['content'])
                html_parts.append(f"""<div class="message tool-call">
        <strong>Tool Response:</strong>
        <pre>{content}</pre>
    </div>
    """)
                del content
            elif entry['type'] == 'tool_use':
                content = format_content_for_display(entry['content'])
                html_parts.append(f"""<div class="message tool-result">
        <strong>Tool Call:</strong>
        <pre>{content}</pre>
    </div>
    """)
                del content
            elif entry['type'] == 'image':
                # Get description if available
                description = entry.get('description', 'Generated Image')          
                # Display base64 encoded image with metadata
                content = entry['content']
                html_parts.append(f"""<div class="message image">
            <strong>{safe_escape(description)}:</strong><br>
            <img src="data:image/png;base64,{content}" alt="{safe_escape(description)}">
    </div>
    """)
                del description, content      
            elif entry['type'] == 'interactive_map':
                description = entry.get('description', 'Interactive Map')
                
                # Safe handling of different content structures
                try:
                    if isinstance(entry['content'], dict):
                        # Content is a dictionary - try to get HTML
                        if 'html' in entry['content']:
                            map_html = entry['content']['html']
                        else:
                            # Dictionary but no 'html' key - convert to string
                            map_html = f"<pre>{safe_escape(str(entry['content']))}</pre>"
                    elif isinstance(entry['content'], str):
                        # Content is already a string - check if it's HTML
                        if '<' in entry['content'] and '>' in entry['content']:
                            # Looks like HTML
                            map_html = entry['content']
                        else:
                            # Plain text - escape it
                            map_html = f"<pre>{safe_escape(entry['content'])}</pre>"
                    else:
                        # Unknown content type - convert to string
                        map_html = f"<pre>{safe_escape(str(entry['content']))}</pre>"
                        
                except (KeyError, TypeError) as e:
                    # Fallback if anything goes wrong
                    map_html = f"<pre>Error displaying interactive map: {str(e)}</pre>"
                
                html_parts.append(f"""<div class="message interactive-map">
                    <strong>{safe_escape(description)}:</strong><br>
                    <div class="map-container">
                        {map_html}
                    </div>
            </div>
            """)
                del description, map_html            

        # Add the final solution
        if task.generated_solution:
            solution_code = get_solution_code(task.generated_solution)
            html_parts.append(f"""<h2>Final Solution:</h2>
    <div class="solution">
        <pre>{safe_escape(solution_code)}</pre>
    </div>

    <h3>Token Usage:</h3>
    <ul class="message">
        <li>Input tokens: {task.generated_solution_input_tokens}</li>
        <li>Output tokens: {task.generated_solution_output_tokens}</li>
    </ul>
    """)
            del solution_code
        
        html_parts.append("""</body>
    </html>
    """)
        
        # Write the HTML to a file
        task_file = run_folder / f"task_{task.task_ID}.html"
        with open(task_file, "w", encoding="utf-8") as f:
            f.write(''.join(html_parts))

        # Final cleanup
        del html_parts
        gc.collect()    
        
        return f"Saved conversation to {task_file}"
    
    except Exception as e:
    # Emergency cleanup
        if 'html_parts' in locals():
            del html_parts
        gc.collect()
        raise e