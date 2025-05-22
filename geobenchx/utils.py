import random
from datetime import datetime
from io import StringIO


from geobenchx.constants import ScoreValues


def generate_timestamp_id(prefix = "TASK"):
    now = datetime.now()
    datestr = now.strftime("%y%m%d")
    timestr = now.strftime("%H%M%S")
    random_digits = str(random.randint(100000, 999999))
        
    return f"{prefix}_{datestr}_{timestr}_{random_digits}"

def get_dataframe_info(df):    
    """ Gets information from df.info() as a string"""    
    buffer = StringIO()
    df.info(buf=buffer)
    # # Get only string (object) columns
    # string_columns = df.select_dtypes(include=['object']).columns         
    # # Print unique values for each string column
    # for col in string_columns:
    #     print(f"\nUnique values in {col}:", file=buffer)
    #     print(str(df[col].unique()), file=buffer)

    # Calculate percentages of nonj-empty cells
    non_empty_stats = (df.count() / len(df) * 100).round(2)
    non_empty_info = "\n\nNon-empty values (%):\n"
    for col, pct in non_empty_stats.items():
        non_empty_info += f"{col}: {pct}%\n"

    info_string = buffer.getvalue()
    buffer.close() 
    return info_string, non_empty_info


def get_solution_code(solution: "Solution", add_num = True):
    """Print out solutions in readable Python-like format """    
    buffer = StringIO()
    solution = [step.model_dump() for step in solution.steps]
    for i, step in enumerate(solution):
        func_name = step['function_name']
        args_dict = step['arguments']
        comment = step.get('comment')
        args_str = ', '.join(f"{k}='{v}'" for k, v in args_dict.items())
        if add_num:
            print(f"{i+1}.", file=buffer, end = " ")
        print(f"{func_name}({args_str})", file=buffer, end= " ")
        if comment:
            print(f"#{comment}", file=buffer, end = " ")
        print("\n", file=buffer, end = "")    
    
    return buffer.getvalue()

def compute_confusion_stats(scores_human, scores_LLM, labels=ScoreValues.values(), alpha=0.05):
    """
    Compute confusion matrix counts, overall accuracy, and confidence bounds.
    
    Parameters:
    - scores_human: list of human scores
    - scores_LLM: list of LLM scores
    - labels: score categories (default: [0, 1, 2])
    - alpha: significance level (default: 0.05 for 95% CI)
    
    Returns:
    - cm_counts: confusion matrix with counts
    - accuracy: proportion of correct answers (diagonal sum / total)
    - accuracy_ci: tuple of (lower_bound, upper_bound) for accuracy
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import confusion_matrix
    from statsmodels.stats.proportion import proportion_confint
    
    # Compute counts matrix
    cm_counts = confusion_matrix(scores_human, scores_LLM, labels=labels)
    
    # Calculate total records and correct answers (diagonal sum)
    total_records = len(scores_human)
    total_correct = np.sum(np.diag(cm_counts))
    
    # Calculate accuracy (diagonal frequency)
    accuracy = total_correct / total_records if total_records > 0 else 0
    
    # Compute confidence interval for accuracy
    accuracy_ci = proportion_confint(total_correct, total_records, alpha=alpha, method='wilson')
    
    return (pd.DataFrame(cm_counts, 
                         index=[f'Human_{l}' for l in labels], 
                         columns=[f'LLM_{l}' for l in labels]),
            accuracy,
            accuracy_ci)