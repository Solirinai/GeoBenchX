import os, sys
import streamlit as st
import json
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError
from typing import List, Dict, Optional
from geobenchx.utils import get_solution_code
from geobenchx.dataclasses import TaskSet, Task, Solution, Step
import io
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
from geobenchx.agent import execute_task
import traceback 
from geobenchx.constants import DATA_FOLDER, MODEL_GPT, MODEL_CLAUDE, MODEL_GEMINI, MODEL_GEMINI_ADV, ScoreValues, RESULTS_FOLDER

TASK_EXECUTE_MODEL = MODEL_CLAUDE
TASK_EXECUTE_TEMPERATURE = 0

# TODO: 
# - add button View that shows all fields of a Task
# - remove table when generating solution
# - make JSON editor more friendly (add line numbers, can do TABs, etc)


def execute_task_view(task: Task, index: int, file_name: str, tasks: TaskSet, folder: str):
    st.subheader(f"Executing Task {task.task_ID}")
    
    # Show task text
    st.write("Task Text:", task.task_text)

    if 'task_executed' not in st.session_state:
        st.session_state.task_executed = False

    # Create an expander for output
    with st.expander("Execution Output", expanded=True):
        if not st.session_state.task_executed:        
            with st.spinner('Executing task...'):
                # Capture stdout
                output_placeholder = st.empty()
                # Clear any existing plots
                plt.close('all')

                class StreamingBuffer(io.StringIO):
                    def __init__(self):
                        super().__init__()
                        self._output = ""
                    
                    def write(self, text):
                        self._output += text
                        # Update the Streamlit display
                        output_placeholder.text(self._output)
                        return super().write(text)
                
                output = StreamingBuffer()

                with redirect_stdout(output):
                    try:
                        # Execute the task
                        generated_solution, _, _, _ = execute_task(task_text = task.task_text, temperature=TASK_EXECUTE_TEMPERATURE, model=TASK_EXECUTE_MODEL, capture_history=False, max_steps=40)
                        st.session_state.generated_solution = generated_solution
                        
                        # Get all figures that were created
                        figures = [plt.figure(n) for n in plt.get_fignums()]
                        if figures:
                            st.write("#### Generated Plots:")
                            for fig in figures:
                                st.pyplot(fig, use_container_width=False)
                            plt.close('all')  # Clear all figures
                        if "generated_solution" in st.session_state:
                            st.write("#### Generated solution:")
                            st.code(st.session_state.generated_solution.model_dump_json(indent=4), language='json')
                            
                    except Exception as e:
                        st.code(f"Error during execution: {traceback.format_exc()}", language='python')
                st.session_state.task_executed = True
    
    # Save/Cancel buttons
    col1, col2 = st.columns([1, 4])
    if col1.button("Save"):
        if 'generated_solution' in st.session_state:
            task.reference_solutions = [st.session_state.generated_solution]
            delattr(st.session_state, 'generated_solution')
        tasks[index] = task
        tasks.save_to_file(file_name, folder)
        st.session_state.executing_task_index = None
        st.session_state.task_executed = False
        st.rerun()
    
    if col2.button("Cancel"):
        if 'generated_solution' in st.session_state:
            delattr(st.session_state, 'generated_solution')        
        st.session_state.executing_task_index = None
        st.session_state.task_executed = False
        st.rerun()

def edit_task(task: Task, index: int, file_name: str, tasks: TaskSet, folder: str):
    st.subheader(f"Editing Task {task.task_ID}")
    
    # Edit task text
    new_task_text = st.text_area(
        "Task Text",
        value=task.task_text,
        height=100
    )
    # Edit task text
    new_reference_solution_description = st.text_area(
        "Reference Solution Description",
        value=task.reference_solution_description,
        height=200
    )
    
    # Editable text area for solution
    json_data = json.dumps([solution.model_dump(exclude_unset=True) for solution in task.reference_solutions], indent=4)
    new_reference_solutions_json = st.text_area(
        "Edit Reference Solutions JSON",
        value=json_data,
        height=700
    )
    
    col1, col2 = st.columns([1, 4])
    if col1.button("Save Changes"):
        try:
            # Update task text
            task.task_text = new_task_text
            # Update task text
            task.reference_solution_description = new_reference_solution_description
            
            # Update solutions
            new_reference_solutions_list = json.loads(new_reference_solutions_json)
            task.reference_solutions = [Solution.model_validate(solution) for solution in new_reference_solutions_list]
            
            # Save changes
            tasks[index] = task
            tasks.save_to_file(file_name, folder)
            st.success("Saved successfully")
            st.session_state.editing_task_index = None
            st.rerun()
        except Exception as e:
            st.error(f"Invalid JSON: {str(e)}")
    
    if col2.button("Cancel"):
        st.session_state.editing_task_index = None
        st.rerun()

def evaluate_task(task: Task, index: int, file_name: str, tasks: TaskSet, folder: str):
    st.subheader(f"Evaluating Task {task.task_ID}")
    st.write("#### Task Text:", task.task_text)
    st.write("#### Reference solutions:")

    for sol_idx, solution in enumerate(task.reference_solutions):
        if sol_idx > 0:
            st.write("---")  # Separator between solutions
        st.write(f"Solution {sol_idx + 1}:", key=f"ref_sol_header_{sol_idx}")
        st.code(get_solution_code(solution), language="python")

    st.write("#### Generated solution:")
    if task.generated_solution:
        st.code(get_solution_code(task.generated_solution), language="python")


    # Edit task text
    match_reasoning_Human = st.text_area(
        "Reasoning for match score",
        value=task.match_reasoning_Human,
        height=100
    )
    
    default_index = ScoreValues.names().index(task.match_score_Human.name) if task.match_score_Human is not None else 0
    match_score_name_Human = st.selectbox("Match score:", ScoreValues.names(), index=default_index)
    match_score_Human = ScoreValues[match_score_name_Human]
    
    col1, col2 = st.columns([1, 4])
    if col1.button("Save Changes"):
        # Update score
        task.match_score_Human = match_score_Human
        # Update reasoning
        task.match_reasoning_Human = match_reasoning_Human
        
        # Save changes
        tasks[index] = task
        tasks.save_to_file(file_name, folder)
        st.success("Saved successfully")
        st.session_state.evaluating_task_index = None
        st.rerun()
    if col2.button("Cancel"):
        st.session_state.evaluating_task_index = None
        st.rerun()

def get_json_files(directory: str) -> List[str]:
    return [f for f in os.listdir(directory) if f.endswith('.json')]

def show_tasks_table(file_name: str, folder: str):
    tasks = TaskSet.read_from_file(file_name, folder)
    
    if 'editing_task_index' not in st.session_state:
        st.session_state.editing_task_index = None
    if 'executing_task_index' not in st.session_state:
        st.session_state.executing_task_index = None
    if 'evaluating_task_index' not in st.session_state:
        st.session_state.evaluating_task_index = None
   
    if st.session_state.editing_task_index is not None:
        edit_task(tasks[st.session_state.editing_task_index], 
                 st.session_state.editing_task_index,
                 file_name, 
                 tasks,
                 folder)
    elif st.session_state.executing_task_index is not None:
        execute_task_view(tasks[st.session_state.executing_task_index],
                        st.session_state.executing_task_index,
                        file_name,
                        tasks,
                        folder)        
    elif st.session_state.evaluating_task_index is not None:
        evaluate_task(tasks[st.session_state.evaluating_task_index],
                      st.session_state.evaluating_task_index,
                      file_name,
                      tasks,
                      folder)
    else:
        # Wider columns, especially for Solution, and fixed width for buttons
        widths = [1, 1, 3, 10, 1, 1, 1]
        cols = st.columns(widths)
        cols[0].write("#### Task ID")
        cols[1].write("#### Task labels")

        cols[2].write("#### Task Text")
        cols[3].write("#### Reference Solutions")
        cols[4].write("#### Edit")
        cols[5].write("#### Generate")
        cols[6].write("#### Evaluate")
        st.divider()
        for i, task in enumerate(tasks):
            cols = st.columns(widths)
            cols[0].write(task.task_ID)
            cols[1].write('  \n'.join([l for l in task.task_labels]))
            cols[2].write(task.task_text)
            
            with cols[3]:
                # Iterate through solutions and display Python code for each
                for sol_idx, solution in enumerate(task.reference_solutions):
                    if sol_idx > 0:
                        st.write("---")  # Separator between solutions
                    st.write(f"Solution {sol_idx + 1}:", key=f"sol_header_{i}_{sol_idx}")
                    st.code(get_solution_code(solution), language="python")

            
            # Center the buttons in their columns using containers
            with cols[4].container():
                if st.button("Edit", key=f"edit_{i}", use_container_width=True):
                    st.session_state.editing_task_index = i
                    st.rerun()
            
            with cols[5].container():
                if st.button("Generate", key=f"gen_{i}", use_container_width=True):
                    st.session_state.executing_task_index = i
                    st.rerun()

            with cols[6].container():
                if st.button("Evaluate", key=f"evaluate_{i}", use_container_width=True):
                    st.session_state.evaluating_task_index = i
                    st.rerun()

            st.divider()

def run_editor(folder: str):
    st.set_page_config(layout="wide")  # Makes the app use full screen width
    st.title("Task Solution Editor")
   
    json_files = get_json_files(folder)
    selected_file = st.selectbox("Select JSON file", json_files)
    
    if selected_file:
        show_tasks_table(selected_file, folder)

if __name__ == "__main__":
    run_editor(DATA_FOLDER)
    # run_editor(RESULTS_FOLDER)