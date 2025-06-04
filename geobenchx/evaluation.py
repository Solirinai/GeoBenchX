import os
import json
import re
from typing import List, Dict, Any
from statistics import mean, median
from scipy.stats import binomtest
from tqdm import tqdm

import openai
from dotenv import find_dotenv, load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

_ = load_dotenv(find_dotenv())

openai.api_key  = os.getenv('OPENAI_API_KEY')
google_api_key = os.getenv("GOOGLE_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

from geobenchx.constants import ROLE_USER
from geobenchx.tools import(
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
    calculate_column_statistics_tool)
from geobenchx.utils import get_solution_code, compute_confusion_stats
from geobenchx.dataclasses import Task, Solution, Step, TaskSet, select_tasks_with_labels
from geobenchx.constants import (
    MODEL_CLAUDE,
    MODEL_CLAUDE_ADV4,
    MODEL_CLAUDE_ADV3,
    MODEL_GPT_41,
    MODEL_GPT_4o,
    MODEL_GPT_mini,
    MODEL_GEMINI,
    MODEL_GEMINI_ADV,
    MODEL_GEMINI_ADV2,
    RESULTS_FOLDER,
    ScoreValues
)


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

EXAMPLE_0 = """
<TASK>What is the total length of railways within areas that received more than 3 feet of snow this season in the USA?</TASK>
<REFERENCE SOLUTIONS>
<REFERENCE SOLUTION>
get_raster_path(rasterdataset='Accumulated snow cover season 2024-2025, USA, inches') 
generate_contours_display(raster_path='data/sfav2_CONUS_2024093012_to_2025052012_processed.tif', output_filename='snow_contours.shp', contour_interval='36', column_title='snow_depth', nodataval='-99999', output_geodataframe_name='snow_contours', min_value='36') 
load_geodata(geodataset='USA counties borders', output_geodataframe_name='us_counties') 
select_features_by_spatial_relationship(features_geodataframe_name='us_counties', reference_geodataframe_name='snow_contours', spatial_predicates='['intersects']', output_geodataframe_name='snowy_counties') 
load_geodata(geodataset='Railway Network of North America', output_geodataframe_name='railways') 
select_features_by_spatial_relationship(features_geodataframe_name='railways', reference_geodataframe_name='snowy_counties', spatial_predicates='['within', 'intersects', 'crosses']', output_geodataframe_name='snowy_railways') 
visualize_geographies(geodataframe_names='['snowy_counties', 'snowy_railways']', layer_styles='[{'color': 'blue', 'alpha': 0.3, 'label': 'Counties with >3ft snow'}, {'color': 'red', 'alpha': 0.6, 'label': 'Railways'}]', title='Railways in Areas with >3ft Snow Cover (2024-2025 Season)') 
calculate_line_lengths(geodataframe_name='snowy_railways', output_variable_name='railway_lengths')</REFERENCE SOLUTION>
<REFERENCE SOLUTION>
get_raster_path(rasterdataset='Accumulated snow cover season 2024-2025, USA, inches') 
generate_contours_display(raster_path='data/sfav2_CONUS_2024093012_to_2025052012_processed.tif', output_filename='snow_contours.shp', contour_interval='36', column_title='snow_depth', nodataval='-99999', output_geodataframe_name='snow_contours', min_value='36') 
load_geodata(geodataset='Railway Network of North America', output_geodataframe_name='railways') 
select_features_by_spatial_relationship(features_geodataframe_name='railways', reference_geodataframe_name='snow_contours', spatial_predicates='['touches', 'intersects', 'crosses', 'overlaps']', output_geodataframe_name='snowy_railways') 
calculate_line_lengths(geodataframe_name='snowy_railways', output_variable_name='railway_lengths')
</REFERENCE SOLUTION>
</REFERENCE SOLUTIONS>
<CANDIDATE SOLUTION>
load_geodata(geodataset='Railway Network of North America', output_geodataframe_name='railways') 
get_raster_path(rasterdataset='Accumulated snow cover season 2023-2024, USA, inches') 
generate_contours_display(raster_path='data/sfav2_CONUS_2023093012_to_2024093012_processed.tif', output_filename='snow_contours.shp', contour_interval='36', min_value='36', column_title='snow_depth', nodataval='-99', output_geodataframe_name='snow_contours', plot_result='False') 
select_features_by_spatial_relationship(features_geodataframe_name='railways', reference_geodataframe_name='snow_contours', spatial_predicates='['intersects']', output_geodataframe_name='snow_railways') 
calculate_line_lengths(geodataframe_name='snow_railways', output_variable_name='railway_lengths') 
visualize_geographies(geodataframe_names='['snow_contours', 'snow_railways']', layer_styles='[{'color': 'blue', 'alpha': 0.3, 'label': 'Snow > 3 feet'}, {'color': 'red', 'linewidth': 1, 'label': 'Affected Railways'}]', title='Railways in Areas with >3 Feet of Snow Accumulation (2023-2024 Season)')
</CANDIDATE SOLUTION>
<MATCH REASONING>Candidate solution is matching the second reference solution. Since the 'reference_solution_description' says that 'visualize_geographies' step is optional, this difference does not affect the match score. However, 'nodataval' value is identified incorrectly. This is important difference, but since it does not affect the solution result, overall, the candidate solution is partially matching the reference solution. Also, 'spatial_predicates' in the candidate solution include only one, 'intersects'. However, this way, the candidate solution might miss some features comparing to the reference solution. This is a partial match.</MATCH REASONING>
<MATCH SCORE>1</MATCH SCORE>
"""

EXAMPLE_1 = """
<TASK>Make a heatmap showing population concentration in earthquake-affected zones</TASK>
<REFERENCE SOLUTIONS>
<REFERENCE SOLUTION>
load_geodata(geodataset='Earthquakes occurences and magnitude March 15- February 14 2025', output_geodataframe_name='earthquakes') 
get_raster_path(rasterdataset='USA population 2020, people, resolution 1 km') 
get_values_from_raster_with_geometries(raster_path='data/usa_ppp_2020_1km_Aggregated_UNadj.tif', geodataframe_name='earthquakes', output_variable_name='earthquake_population') 
make_heatmap(geodataframe_name='earthquakes', value_column='earthquake_population', radius='30', map_style='carto-positron', title='Heatmap of population in earthquakes locations (March 2024 - February 2025)')
</REFERENCE SOLUTION>
<REFERENCE SOLUTION>
reject_task()
</REFERENCE SOLUTION>
</REFERENCE SOLUTIONS>
<CANDIDATE SOLUTION>
load_geodata(geodataset='Earthquakes occurences and magnitude March 15- February 14 2025', output_geodataframe_name='earthquakes') 
filter_numerical(dataframe_name='earthquakes', conditions='mag >= 4.0', output_dataframe_name='significant_earthquakes') 
get_raster_path(rasterdataset='USA population 2020, people, resolution 1 km') 
make_heatmap(geodataframe_name='significant_earthquakes', value_column='mag', map_style='carto-darkmatter', radius='30')
</CANDIDATE SOLUTION>
<MATCH REASONING>
The candidate solution does not match to any of the reference solutions. It does load correct datasets, but misses important step 'get_values_from_raster_with_geometries' to get population in earthquakes locations and produces heatmap of earthquakes instead of population. Also it contains extra step "filter_numerical" to select major earthquakes though the task does not require this.
</MATCH REASONING>
<MATCH SCORE>0</MATCH SCORE>
"""

EXAMPLE_2 = """
<TASK>How many mineral extraction facilities in Algeria are located within areas with population density over 1000 people per square km?</TASK>
<REFERENCE SOLUTIONS>
<REFERENCE SOLUTION>
load_geodata(geodataset='Mineral extraction facilities in Africa', output_geodataframe_name='mines') 
filter_categorical(dataframe_name='mines', filters={'Country': 'Algeria'}, output_dataframe_name='algeria_mines') 
get_raster_path(rasterdataset='Algeria population density per 1 km 2020, 1 km resolution') 
filter_points_by_raster_values(raster_path='data/dza_pd_2020_1km_UNadj.tif', points_geodataframe_name='algeria_mines', value_column='population_density', output_geodataframe_name='high_density_mines', filter_type='greater', threshold1='1000')
</REFERENCE SOLUTION>
</REFERENCE SOLUTIONS>
<CANDIDATE SOLUTION>
get_raster_path(rasterdataset='Algeria population density per 1 km 2020, 1 km resolution') 
load_geodata(geodataset='Mineral extraction facilities in Africa', output_geodataframe_name='minerals') 
filter_categorical(dataframe_name='minerals', filters={'Country': 'Algeria'}, output_dataframe_name='algeria_minerals') 
filter_points_by_raster_values(raster_path='data/GeoData/dza_pd_2020_1km_UNadj.tif', points_geodataframe_name='algeria_minerals', value_column='pop_density', output_geodataframe_name='high_density_minerals', filter_type='greater', threshold1='1000')
</CANDIDATE SOLUTION>
<MATCH REASONING>Both reference and candidate solution use the same datasets, tools. Order of steps is different, however, this is not an essential difference. Moreover, the last step is the same and the important arguments 'filter_type' and 'threshold1' are the same, meaning the final result will be same.</MATCH REASONING>
<MATCH SCORE>2</MATCH SCORE>
"""

# EXAMPLE_2 = """
# <TASK> </TASK>
# <REFERENCE SOLUTIONS>
# <REFERENCE SOLUTION>

# </REFERENCE SOLUTION>
# <REFERENCE SOLUTION>

# </REFERENCE SOLUTION>
# </REFERENCE SOLUTIONS>
# <CANDIDATE SOLUTION>

# </CANDIDATE SOLUTION>
# <MATCH REASONING></MATCH REASONING>
# <MATCH SCORE></MATCH SCORE>
# """


examples_list = [
    EXAMPLE_0, EXAMPLE_1, EXAMPLE_2
]

EVALUATOR_PERSONA = """
You are working in quality control in a GIS technology office. 
You are checking coding candidate solutions for geospatial tasks for matching the correct solutions that we call reference solutions.
You are attentive to detail, knowledgeable about cartography principles. 
You provide concise reasoning before making decisions on whether the candidate solution matches any of the reference solutions provided for this task.
"""

TOOLS_DESCRIPTION = ''
for tool in tools:
    tool_args_str = '\n'.join([f"{name}({v.get('type', '')}): {v['description']}" for name, v in tool.args.items() if name != 'state'])
    tool_args_str = tool_args_str.replace("{", "{{").replace("}", "}}")
    tool_description = tool.description.replace("{", "{{").replace("}", "}}")
    TOOLS_DESCRIPTION = TOOLS_DESCRIPTION + f"""<TOOL>Tool name: {tool.name}\n Tool arguments:\n {tool_args_str} \n Tool description: {tool_description}</TOOL>\n\n""" 

EXAMPLES = ''
for example in examples_list:
    EXAMPLES = EXAMPLES + f"""<EXAMPLE>{example.replace("{", "{{").replace("}", "}}")}</EXAMPLE>\n\n""" 
    
EVALUATION_TAXONOMY = """
<TAXONOMY>
While evaluating how close the candidate solution matches to the reference solution, you are using matching score.
Matching score = 0 - Candidate solution does not match any of the reference solutions provided for this task, there are essential discrepancies like different non-similar data are used, inaproppriate tools are used or different results are produced
Matching score = 1 - Candidate solution partially matches at least one of the reference solutions provided for this task, there are non-critical discrepancies like color scheme used for mapping.
Matching score = 2 - Candidate solution fully matches one of the reference solutions provided for this task, there are only non-essential discrepancies like wording of the map's legend.
</TAXONOMY>
"""

INSTRUCTIONS = """
<INSTRUCTIONS>
While checking if the candidate solution matches to one of the reference solutions provided for this task, you use the following considerations:
- Solutions are snippets of Python code calling various tools.
- Reference solution can be empty meaning that no tool calls are needed to answer the task.
- Reference solution can consist of single 'reject_task' tool call meaning that the task is not possible to solive with the given tools and datasets.
- For some tasks multiple reference solutions can be specified in REFERENCE SOLUTIONS block.
- If task requires to make contour lines from a raster, unless the task calls for a specific contour interval, the difference between contour interval in reference and candidate solutions does not matter.
- If a task require to make a heatmap, the differences in map styles and radius value between reference and candidate solutions do not matter.
- Pay attention to comments in reference solutions, because they specify when multiple variations of correct tool arguments are acceptable.
- FULL MATCH:
    - If the candidate solution uses the same tools and the same datasets to produce the same results as reference solution, the candidate solution fully matches the reference solution.
    - If the candidate solution stops with 'reject_task' tool call and one of the reference solutions is 'reject_task', the candidate solution fully matches the reference solution.
    - If the reference solution is not empty, then if candidate solution includes some incorrect steps but does include all the correct steps and produced correct results, the candidate solution is considered as matching the reference solution.
    - If the reference solution is empty, then the candidate solution should be empty to be considered a match.
    - If correct data are passed between the tools, it does not matter if the names given to the dataframes, geodataframes and variables are different between the candidate and reference solution.
- NO MATCH:
    - If the candidate solution and the reference solution use different set of tools, different set of datasets, the candidate solution does not match the reference solution.
    - If candidate solution and reference solution use data from different years and difference is 2 years and more, the solutions do not match.
    - It does matter which columns are selected from dataframes and geodataframes. If different columns are selected in the reference and candidate solution, the solutions do not match.
    - If categorical filter is used in both reference and candidate solutions and the filters in both do not match, the candidate solution does not match to the reference solution.
    - If the reference solution is empty, this indicates the task should be solved without using provided tools. In this case, if a candidate solution contains any tool call (even a 'reject_task' tool call), it doesn't match the reference solution.
    - If reference solution is the 'reject_task' tool call and the candidate solution is empty, the candidate solution does not match the reference solution.
- PARTIAL MATCH:
    - Color scheme and colormap selected matter. If the reference solution and candidate solution use different color schemes or color maps, it is a partial match.
    - If nodata values used as argument in some functions is different between reference and generated solution, it is a partial match.
    - If lists of spatial predicates used for spatial selection are different between reference and generated solution, it is a partial match.
    - If candidate solution and reference solution use data from different years and difference is 1 year, it is a partial match.
- Candidate solution matches the reference solution only if all observations indicate a full match. If any observation shows no match, the candidate solution is not matching the reference solution. If no observation shows a no match but at least one shows a partial match, the candidate solution is classified as partially matching he reference solution. 
</INSTRUCTIONS>
"""

RESULT_CHECKING_PROMPT = f""" 
{EVALUATOR_PERSONA}
Solutions use the following TOOLS:
{TOOLS_DESCRIPTION}
After comparing the candidate and reference solutions, you produce the score using the following TAXONOMY:
{EVALUATION_TAXONOMY}
You compare the candidate and reference solutions taking into account the INSTRUCTIONS:
{INSTRUCTIONS}

Below are some EXAMPLES of such comparisons:
{EXAMPLES}

Now analyse the new candidate solution for the task and produce new reasoning and the score:
<TASK>{{task_text}}</TASK>
<REFERENCE SOLUTIONS>
{{reference_solutions}}
</REFERENCE SOLUTIONS>
<CANDIDATE SOLUTION>
{{candidate_solution}}
</CANDIDATE SOLUTION>

First, think step by step. After you finish analysis, please, record your reasoning within tags <MATCH REASONING></MATCH REASONING> and record your match score within tags <MATCH SCORE></MATCH SCORE>.
"""

def score_task_solution(task: Task, model: str = MODEL_CLAUDE, temperature: float = 0, verbose = False) -> tuple[Task, int, int]:
   
    """
    Evaluates a candidate solution against a reference solution using ... for comparison.
    
    Parameters:
        task (Task): object should contain:
            task_text (str): Original task description
            reference_solutions (List[Solution]): Reference solution to compare against
            generated_solution (Solution): Solution to be evaluated
        model (str): name of the LLM used for evaluation, defaults to antropic model
        temperature (float): temperature of the evaluation, defaults to 0 
        verbose (bool, optional): Whether to print detailed evaluation. Defaults to False
    
    Returns:
        tuple: Contains:
            - reasoning_match (str): Explanation of match evaluation
            - score_match (int): Numerical score (0-100) indicating solution quality
            - input_tokens (int): Number of tokens in input prompt
            - output_tokens (int): Number of tokens in ...'s response

    Uses ... to compare solutions and extract evaluation metrics from structured response tags.
    """
    if model in [MODEL_CLAUDE, MODEL_CLAUDE_ADV3,MODEL_CLAUDE_ADV4]:
        llm = ChatAnthropic(model=model, temperature=temperature)
    elif model in [MODEL_GPT_41, MODEL_GPT_mini, MODEL_GPT_4o]:
        llm = ChatOpenAI(model=model, temperature=temperature)
    elif model in [MODEL_GEMINI, MODEL_GEMINI_ADV, MODEL_GEMINI_ADV2]:
        llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)
    else:
        raise ValueError("Model is outside the predetermined list")

    tools = []

    if task.generated_solution is not None:
        generated_solution_str = get_solution_code(task.generated_solution, add_num = False)
    else:
        raise ValueError('generated_solution is None')


    reference_solutions_str = ""
    if task.reference_solution_description is not None:
        reference_solutions_str = f"Here are general comments about the reference solutions: {task.reference_solution_description}"

    for reference_solution in task.reference_solutions:
        reference_solution_str = get_solution_code(reference_solution, add_num = False)
        reference_solutions_str += f"<REFERENCE SOLUTION>\n{reference_solution_str}\n</REFERENCE SOLUTION>"

    PROMPT = RESULT_CHECKING_PROMPT.format(task_text = task.task_text, reference_solutions = reference_solutions_str, candidate_solution = generated_solution_str)

    if verbose:
        print(f"The prompt to be sent to evaluator LLM: \n{PROMPT}")

    graph = create_react_agent(llm, tools, state_modifier=RESULT_CHECKING_PROMPT)

    inputs = {
    "messages": [("user", PROMPT)],
    }
    config = {
        "max_concurrency": 1,
        # "recursion_limit": max_steps,
        }

    for s in graph.stream(inputs, stream_mode="values", config=config):

        message = s["messages"][-1]
        if hasattr(message, 'usage_metadata') and message.usage_metadata:
            input_tokens = message.usage_metadata['input_tokens']
            output_tokens = message.usage_metadata['output_tokens']        

        if verbose:
            print(f"Evaluator LLM response: \n{message}")
        
        verdict = str(message)
        reasoning_match = re.search(r'<MATCH REASONING>(.*?)</MATCH REASONING>', verdict, re.DOTALL)
        if reasoning_match:
            reasoning_match =  reasoning_match.group(1).strip()
        score_match = re.search(r'<MATCH SCORE>(.*?)</MATCH SCORE>', verdict, re.DOTALL)
        if score_match:
            score_match = score_match.group(1).strip()
            score_match = int(score_match)

        task.match_score_LLM = ScoreValues(score_match)
        task.match_reasoning_LLM = reasoning_match

    return task, input_tokens, output_tokens

def generate_eval_stats_evaluator(tasks_filename: str, folder: str = RESULTS_FOLDER):
    evaluated_tasks = TaskSet.read_from_file(tasks_filename, folder)

    scores = [(task.match_score_Human.value, task.match_score_LLM.value) for task in evaluated_tasks if task.match_score_LLM is not None] 
    scores_human, scores_LLM = zip(*scores)

    counts, accuracy, accuracy_ci = compute_confusion_stats(scores_human, scores_LLM)
    print(counts)
    print(accuracy) 
    print(accuracy_ci)
        

def generate_eval_stats(tasks_w_solutions: TaskSet, alpha: float = 0.05):
    """
    Generates statistical metrics for solution evaluation results.

    Parameters:
        tasks_filename (str): Name of file containing evaluated tasks to compute stats on
        folder (str, optional): Directory containing tasks file. Defaults to RESULTS_FOLDER

    Returns:
        dict: Contains:
            - match_frequencies (dict): Score frequencies as proportions
            - mean_diff_in_solution_length (float): Mean relative difference in solution lengths
            - median_diff_in_solution_length (float): Median relative difference in solution lengths 
            - confidence_intervals (dict): {score: (lower_bound, proportion, upper_bound)} at 1-alpha confidence

    Computes score distributions, solution length differences, and confidence intervals using binomial test.
    """

    scores = [task.match_score_LLM for task in tasks_w_solutions]
    counts = {x: scores.count(x) for x in set(scores)}
    frequencies = {x: round(scores.count(x)/len(scores), 2) for x in set(scores)}
    len_diffs_list = []
    for task in tasks_w_solutions:
        if task.reference_solutions: 
            len_cand_solution = len(task.generated_solution)
            len_correct_solution = sum(len(ref_solution) for ref_solution in task.reference_solutions) / len(task.reference_solutions) #average len of reference solution            
            if len_correct_solution > 0:                
                len_diff = (len_cand_solution - len_correct_solution)/len_correct_solution
            else:
                len_diff = len_cand_solution
            len_diffs_list.append(len_diff)
    len_diffs_mean = mean(len_diffs_list)  
    len_diffs_median = median(len_diffs_list) 

    total = len(tasks_w_solutions)
    confidence_level = 1 - alpha
    intervals = {}

    for value, count in counts.items():
        # count successes out of total
        res = binomtest(count, total)
        ci = res.proportion_ci(confidence_level=confidence_level)
        p_hat = count / total

        intervals[value] = (ci.low, p_hat, ci.high)
        intervals[value] = [round(el, 2) for el in intervals[value]]  
    
    model = tasks_w_solutions.metadata.get('model', None)
    temperature = tasks_w_solutions.metadata.get('temperature', None)
    eval_model = tasks_w_solutions.metadata.get('evaluator_model', None)
    eval_temp = tasks_w_solutions.metadata.get('evaluator_temperature', None)
    
    res_dict = {
            'model': model,
            'temperature': temperature,
            'evaluator_model': eval_model,
            'evaluator_temperature': eval_temp,
            'match_frequencies': frequencies,
            'mean_diff_in_solution_length': len_diffs_mean,
            'median_diff_in_solution_length':len_diffs_median,
            'confidence_intervals': intervals
            }
    
    return res_dict

def score_solutions_set(tasks_filename: str, folder: str = RESULTS_FOLDER, model: str = MODEL_CLAUDE, temperature: float = 0, skip_scored = True):
    """
    Scores a set of generated solutions from taskfile against reference solutions and saves results back to the taskfile.

    Parameters:
        tasks_filename (str): Name of file containing tasks to evaluate
        folder (str, optional): Directory containing tasks file. Defaults to RESULTS_FOLDER
        model: name of LLM used for evaluation, defaults to antropic model used
        temperature: temperature of the generation, defaults to 0

    Returns:
        Nothing - the results of scoring are saved back to the file

    Processes each task:
    - Loads tasks and metadata
    - Scores generated vs reference solutions 
    - Tracks token usage
    - Saves updated results after each evaluation
    - Reports total token usage
    """
    
    tasks = TaskSet.read_from_file(tasks_filename, folder)
    tasks.metadata['evaluator_model'] = model
    tasks.metadata['evaluator_temperature'] = temperature
    total_input, total_output = 0, 0
    for task in tqdm(tasks):
        print(f"Task ID: {task.task_ID}")
        print(f"Task text: {task.task_text}")    
        if task.match_score_LLM is not None and skip_scored:
            print("Skipping task, it is alredy evaluated.")
            continue
        try:
            task, input_tokens, output_tokens = score_task_solution(task, model, temperature) 
            print(f"Matching score: {task.match_score_LLM}")
            print(f"input tokens: {input_tokens}, output tokens: {output_tokens}")
            total_input += input_tokens
            total_output += output_tokens
            tasks.metadata['total_input_tokens_for_evaluation'] = total_input
            tasks.metadata['total_output_tokens_for_evaluation'] = total_output 
            tasks.save_to_file(tasks_filename, folder)
        except Exception as e:
            print(repr(e))
    print(f"Total input tokens: {total_input}, total output tokens: {total_output}")  
   

def get_eval_stats_by_subsets(tasks_file_name: str, folder: str = RESULTS_FOLDER, labels: List[str] = None, functions_names: List[str] = None, alpha: float = 0.05):
    """
    Generate evaluation statistics for tasks filtered by labels or function names.
    
    Args:
        tasks_file_name: Name of the task file to analyze
        folder: Directory containing the tasks file, defaults to RESULTS_FOLDER
        labels: List of labels to filter tasks by
        functions_names: List of function names to filter tasks by
        alpha: Significance level for statistical tests, defaults to 0.05
        
    Returns:
        List of dictionaries containing evaluation statistics for each subset
    """

    tasks_w_solutions = TaskSet.read_from_file(tasks_file_name, folder)

    results = []

    if labels is not None:
        for label in labels:
            tasks_with_label = select_tasks_with_labels(tasks_w_solutions, [label])
            res = generate_eval_stats(tasks_with_label, alpha)
            result_dict = {
                'task_label': label,
                'tasks_with_label':len(tasks_with_label),
                'evaluation_results': res
            }
            results.append(result_dict)
    else:
        print('No labels were provided for generating separate statistics.')

    if functions_names is not None:
        for function_name in functions_names:

            tasks_with_function = [task for task in tasks_w_solutions.tasks
                                    if any(step.function_name == function_name for solution in task.reference_solutions 
                                    for step in solution.steps)]
            taskset_with_function = TaskSet(metadata=tasks_w_solutions.metadata, tasks=tasks_with_function)
            res = generate_eval_stats(taskset_with_function, alpha)
            result_dict = {
                'function_evaluated': function_name,
                'tasks_using_function': len(tasks_with_function),
                'evaluation_results': res
            }
            results.append(result_dict)

            tasks_without_function = [task for task in tasks_w_solutions.tasks
                                     if not any(step.function_name == function_name for solution in task.reference_solutions 
                                                for step in solution.steps)]
            taskset_without_function = TaskSet(metadata=tasks_w_solutions.metadata, tasks=tasks_without_function)
            res_2 = generate_eval_stats(taskset_without_function, alpha)
            result_dict_2 = {
                'function_evaluated': f"NOT_{function_name}",
                'tasks_WITHOUT_function': len(tasks_without_function),
                'evaluation_results': res_2               
            }
            results.append(result_dict_2)
    else:
        print('No functions were provided for generating separate statistics')
            
    return results

def get_eval_stats_by_pure_solvability(tasks_file_name: str, folder: str = RESULTS_FOLDER, alpha: float = 0.05):
    """
    Generate evaluation statistics separately for:
    1. Tasks that have only 1 reference solution and this solution is "reject_task"
    2. All other tasks (which may include tasks with multiple reference solutions, 
        one of which could be "reject_task")

    Args:
        tasks_file_name: Name of the task file to analyze
        folder: Directory containing the tasks file, defaults to RESULTS_FOLDER
        alpha: Significance level for statistical tests, defaults to 0.05
        
    Returns:
        List of dictionaries containing evaluation statistics for each subset
    """

    tasks_w_solutions = TaskSet.read_from_file(tasks_file_name, folder)

    results = [] 

    unsolvable_tasks = [task for task in tasks_w_solutions.tasks
                            if len(task.reference_solutions)==1 and 
                            any(step.function_name == 'reject_task' for step in task.reference_solutions[0].steps)
                            ]
    unsolvable_taskset = TaskSet(metadata=tasks_w_solutions.metadata, tasks=unsolvable_tasks)
    res = generate_eval_stats(unsolvable_taskset, alpha)
    result_dict = {
        'subset_type': 'purely_unsolvable',
        'description': 'Tasks with only 1 reference solution that is reject_task',
        'task_count': len(unsolvable_tasks),
        'evaluation_results': res
    }
    results.append(result_dict)

    solvable_tasks = [task for task in tasks_w_solutions.tasks
                            if len(task.reference_solutions)>1 or 
                            not any(step.function_name == 'reject_task' for step in task.reference_solutions[0].steps)
                            ]
    solvable_taskset = TaskSet(metadata=tasks_w_solutions.metadata, tasks=solvable_tasks)
    res_2 = generate_eval_stats(solvable_taskset, alpha)
    result_dict_2 = {
        'subset_type': 'solvable',
        'description': 'All other tasks (may include multiple solutions, some with reject_task)',
        'task_count': len(solvable_tasks),
        'evaluation_results': res_2          
    }
    results.append(result_dict_2)
            
    return results
