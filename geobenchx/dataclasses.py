import json
import os
from collections import Counter
from pprint import pformat, pprint
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from geobenchx.constants import (
    DATA_FOLDER,
    NO_LABEL,
    RESULTS_FOLDER,
    ScoreValues,
    TaskLabels,
)
from geobenchx.utils import generate_timestamp_id, get_solution_code


class Step(BaseModel):
    function_name: str
    arguments: Dict
    comment: Optional[str] = None


class Solution(BaseModel):
    steps: List[Step] = Field(default_factory=list)

    def __len__(self):
        """
        Optional convenience method to allow:
            len(task_set)
        """
        return len(self.steps)

class Task(BaseModel):
    task_ID: str = Field(default_factory=generate_timestamp_id)
    task_text: str
    
    task_labels: Optional[List[TaskLabels]] = Field(default_factory=list)

    reference_solution_description: Optional[str] = None
    reference_solutions: Optional[List[Solution]] = Field(default_factory=list)

    generated_solution: Optional[Solution] = None

    generated_solution_input_tokens: Optional[int] = None
    generated_solution_output_tokens: Optional[int] = None

    match_reasoning_LLM: Optional[str] = None
    match_score_LLM: Optional[ScoreValues] = None

    match_reasoning_Human: Optional[str] = None
    match_score_Human: Optional[ScoreValues] = None
    
    class Config:
        extra = "forbid"  # This allows extra fields


    def __str__(self):
        repr_dict = {}
        for k,v in self:
            if isinstance(v, ScoreValues): 
                strval = v.name
            elif isinstance(v, Solution):
                strval = get_solution_code(v)
            elif isinstance(v, list) and v and isinstance(v[0], Solution):
                strval = "\n---\n".join([get_solution_code(s) for s in v])
            elif isinstance(v, list) and v and isinstance(v[0], TaskLabels):
                strval = '\n'.join([l.name for l in v])
            elif isinstance(v, list) and not v:
                strval = ''
            else:
                strval = str(v)

            repr_dict[k] = strval
        
        s = pd.Series(repr_dict, name = f"Task details").to_markdown()

        return s
    
    __repr__ = __str__




class TaskSet(BaseModel):
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tasks: List[Task] = Field(default_factory=list)

    def __iter__(self):
        """
        Make TaskSet iterable over its tasks, so:
            for task in task_set:
                ...
        """
        return iter(self.tasks)
    
    def __len__(self):
        """
        Optional convenience method to allow:
            len(task_set)
        """
        return len(self.tasks)
    
    def __getitem__(self, index: Union[int, slice]) -> Union[Task, List[Task]]:
        """
        Return the item(s) at the given index or slice.
        - If `index` is an `int`, returns a single Task.
        - If `index` is a `slice`, returns a list of Tasks.
        """
        return self.tasks[index]
    
    def __setitem__(
        self,
        index: Union[int, slice],
        value: Union[Task, List[Task]]
    ) -> None:
        """
        Assign to the item(s) at the given index or slice.
        - If `index` is an `int`, `value` must be a single Task.
        - If `index` is a `slice`, `value` should be a list of Tasks.
        """
        # If you want to enforce that the value is valid Task(s), you can:
        #  - check type/value here, or
        #  - rely on the tasks: List[Task] definition to ensure Pydantic data integrity.
        self.tasks[index] = value

    def append(self, task: Task) -> None:
        """
        Append a single Task to the TaskSet.
        Example usage: taskset.append(Task(...))
        """
        # If you need validation beyond what's built into Task, add it here.
        self.tasks.append(task)

    def extend(self, tasks: List[Task]) -> None:
        """
        Extend the TaskSet with multiple Tasks.
        Example usage: taskset.extend([Task(...), Task(...)])
        """
        self.tasks.extend(tasks)

    @classmethod
    def read_from_file(cls, filename: str, folder:str = DATA_FOLDER) -> "TaskSet":
        """
        Reads tasks and metadata from a JSON file.

        Args:
            filename: Name of JSON file containing tasks
            folder: Directory path containing the file

        Returns:
            TaskSet object
        """    

        file_path = os.path.join(folder, filename)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return cls(**data)


    def save_to_file(self, filename: str, folder: str):
        """
        Saves tasks and metadata to a JSON file.

        Args:
            filename: Output JSON filename 
            folder: Directory path for output file

        """
        # Create the folder if it doesn't exist
        os.makedirs(folder, exist_ok=True)
        
        # Full path to the output file
        file_path = os.path.join(folder, filename)
        
        # Convert the taskset to a dictionary
        data = self.model_dump()
        
        # Write to file with pretty formatting
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def get_labels_counts(self) -> Dict[str, int]:
        """
        Returns a dictionary mapping each unique label to the number of
        tasks that contain it.
        """
        label_counter = Counter()
        for task in self.tasks:
            # If a task has no labels, treat it as having the "NO_LABEL" label.
            if not task.task_labels:
                label_counter.update([NO_LABEL])
            else:
                label_counter.update(task.task_labels)
        return dict(label_counter)

    def sample_stratified(self, num_samples: int) -> "TaskSet":
        """
        Returns a new TaskSet with `num_samples` tasks selected in a manner
        that roughly balances label coverage, using NumPy's weighted sampling
        without replacement.

        If num_samples >= len(self.tasks), returns all tasks.
        """
        # Edge case: if num_samples >= total tasks, just return all
        if num_samples >= len(self.tasks):
            return TaskSet(metadata=self.metadata, tasks=self.tasks)

        # Retrieve label frequencies using get_labels_counts
        label_freq = self.get_labels_counts()

        # Compute a weight for each task:
        #     weight(task) = sum(1 / label_freq[label] for label in task_labels)
        # If a task has no labels, assign weight=0.
        weights = []
        for t in self.tasks:
            if t.task_labels:
                w = sum((1.0 / label_freq[lbl]) for lbl in t.task_labels if lbl in label_freq)
            else:
                # If the task has no labels, treat it as having NO_LABEL
                w = 1.0 / label_freq[NO_LABEL]
            weights.append(w)

        weights_array = np.array(weights, dtype=float)
        total_weight = weights_array.sum()

        # Normalize weights to probabilities
        probs = weights_array / total_weight
        # Weighted sampling without replacement using NumPy
        chosen_indices = np.random.choice(
            len(self.tasks),
            size=num_samples,
            replace=False,
            p=probs
        )

        chosen_tasks = [self.tasks[i] for i in chosen_indices]

        return TaskSet(metadata=self.metadata, tasks=chosen_tasks)
    
def select_tasks_with_labels(self, labels: List[str]) -> "TaskSet":
    """
    If `labels` is non-empty, return only tasks that have at least
    one label from `labels`.
    If `labels` is empty, return only tasks whose `task_labels`
    list is empty.
    """
    # Convert list to a set for faster membership checks
    labels_set = set(labels)

    if not labels:  # empty list
        # Select tasks that have no labels at all
        filtered_tasks = [task for task in self.tasks if not task.task_labels]
    else:
        # Select tasks that have at least one label in labels_set
        filtered_tasks = [
            task for task in self.tasks
            if any(label in labels_set for label in task.task_labels)
        ]

    return TaskSet(metadata=self.metadata, tasks=filtered_tasks)


def check_data_validity(master_data_filename = "all_sets_tasks_solutions.json"):

    #load master list
    print(f"Loading master datalist from file {master_data_filename} in folder {DATA_FOLDER}")
    tasks = TaskSet.read_from_file(master_data_filename, DATA_FOLDER)

    print("Checking duplicates in master list...")
    ids = [task.task_ID for task in tasks]
    dup_ids = {task_id: cnt for task_id, cnt in Counter(ids).items() if cnt>1}
    if dup_ids:
        print(f"Found duplicated IDs: {dup_ids}")
    texts = [task.task_text for task in tasks]
    dup_texts = {task_text: cnt for task_text, cnt in Counter(texts).items() if cnt>1}
    if dup_texts:
        print(f"Found duplicated texts: {dup_texts}")
    print("Done.")

    print("Checking labels in master list...")
    for task in tasks:
        if not task.task_labels:
            print(f"WARNING: For task ID: {task.task_ID} from {master_data_filename} there are no LABELS!")    

    #make list of files to check in DATA_FOLDER and RESULTS_FOLDER
    #we assume that 
    # 1) all JSON files in DATAFOLDER are subsets of master list
    # 2) all JSON files in RESULTS_FOLDER are same set as master list but with generated solutions and stats added (but text, labels, ref solutions, etc, are same as in master list)
    folders = [RESULTS_FOLDER, DATA_FOLDER]
    files_to_check = {}
    for folder in folders:
        json_files = [f for f in os.listdir(folder) if f.endswith('.json')]
        if folder == DATA_FOLDER:
            json_files.remove(master_data_filename)
        files_to_check[folder] = json_files

    print(f"Will be checking these files:\n {pformat(files_to_check)}...")

    for folder in files_to_check:
        print(f"Processing folder: {folder}...")
        for fname in files_to_check[folder]:
            print(f"Processing file: {fname}...")
            try:
                tasks_to_check = TaskSet.read_from_file(fname, folder)
                for task in tasks:
                    ts = [t for t in tasks_to_check if t.task_text==task.task_text]
                    if len(ts) > 1 or (len(ts)==0 and folder == RESULTS_FOLDER):
                        print(f"ERROR: For task ID: {task.task_ID} from {master_data_filename} found {len(ts)} tasks with same TEXT in file {folder}\\{fname} !")
                    ts = [t for t in tasks_to_check if t.task_ID==task.task_ID]
                    if len(ts) > 1 or (len(ts)==0 and folder == RESULTS_FOLDER):
                        print(f"ERROR: For task ID: {task.task_ID} from {master_data_filename} found {len(ts)} tasks with same ID in file {folder}\\{fname} !")
                    elif len(ts):
                        task_to_check = ts[0]
                        if task_to_check.task_text != task.task_text:
                            print(f"ERROR: For task ID: {task.task_ID} from {master_data_filename} task with same ID in file {folder}\\{fname} contains different TEXT!")
                        if task_to_check.task_labels != task.task_labels:
                            print(f"ERROR: For task ID: {task.task_ID} from {master_data_filename} task with same ID in file {folder}\\{fname} contains different LABELS!")


            except Exception as e:
                print(f"ERROR: happened processing file {fname} in folder {folder}:\n {'-'*20} \n {e}")
                continue

    print('Done.')