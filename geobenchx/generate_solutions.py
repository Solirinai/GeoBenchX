import time
from tqdm import tqdm
from typing import List
# from geobenchx.agent import execute_task
from geobenchx.agent import execute_task
from geobenchx.utils import get_solution_code
from geobenchx.constants import RESULTS_FOLDER
from geobenchx.dataclasses import TaskSet


wait_time = 61

def generate_solutions(tasks: TaskSet, model: str, temperature: float, output_filename: str = None, max_steps: int =25, skip_solved = True) -> tuple[TaskSet, int]:
    """
   Generates solution attempts for each unsolved task using an LLM.

    Parameters:
       tasks (TaskSet): Set of tasks to solve
       model (str): Name of LLM model to use
       temperature (float): Sampling temperature for generation
       output_filename (str, optional): File to save results. Defaults to None.
       max_steps (int, optional): Maximum number of steps allowed. Defaults to 25,
       skip_solved:

    Returns:
       tuple: Contains:
           - tasks (TaskSet): Original TaskSet with generated solutions added
           - total_tokens (int): Total token usage across all runs

   Skips tasks that already have a generated solution. Saves intermediate results to 
   the specified file if output_filename is provided.
    """
    tasks.metadata['model'] = model
    tasks.metadata['temperature'] = temperature
    total_input = 0
    total_output = 0
    for task in tqdm(tasks):
        print(f"Task ID: {task.task_ID}")
        print(f"Task text: {task.task_text}")
        if task.generated_solution is not None and skip_solved:
            print("Skipping task, it is alredy solved.")
            continue
        success = False
        try_count = 0
        while(not success):
            try:
                solution, input_tokens, output_tokens = execute_task(task.task_text, temperature = temperature, model=model, max_steps=max_steps)
                print('='*30)
                print(get_solution_code(solution))
                print(f"Tokens used: input tokens {sum(input_tokens)}, output_tokens {sum(output_tokens)}")
                print('='*30)
                total_input += sum(input_tokens)
                total_output += sum(output_tokens)
                task.generated_solution = solution
                task.generated_solution_input_tokens = sum(input_tokens)
                task.generated_solution_output_tokens = sum(output_tokens)

                if output_filename:
                    tasks.save_to_file(output_filename, folder=RESULTS_FOLDER)
                success = True
            except Exception as e:
                try_count += 1
                print(repr(e))

    print(f"TOTAL tokens used: total input tokens {total_input}, total output_tokens {total_output}")
    tasks.metadata['total_input_tokens_for_generation'] = total_input
    tasks.metadata['total_output_tokens_for_generation'] = total_output

    if output_filename:
        tasks.save_to_file(output_filename, folder=RESULTS_FOLDER)

    return tasks, total_input, total_output
