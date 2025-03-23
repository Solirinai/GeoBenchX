from enum import Enum, IntEnum

ROLE_SYSTEM = 'system'
ROLE_TOOL = 'function'
ROLE_USER = 'user'
ROLE_ASSISTANT = 'assistant'
ROLE_MODEL = 'model'

DATA_FOLDER = 'benchmark_set'
RESULTS_FOLDER = 'results'

MODEL_GPT = 'gpt-4o-2024-08-06'
MODEL_GPT_mini = 'gpt-4o-mini-2024-07-18'
MODEL_O = 'o3-mini-2025-01-31'
MODEL_GEMINI = 'gemini-1.5-pro-002'
MODEL_GEMINI_ADV = 'gemini-2.0-flash-001'
MODEL_CLAUDE = 'claude-3-5-sonnet-20241022'
MODEL_CLAUDE_mini = 'claude-3-5-haiku-20241022'
MODEL_CLAUDE_ADV = 'claude-3-7-sonnet-20250219'

class ScoreValues(IntEnum):
    NO_MATCH = 0
    PARTIAL_MATCH = 1
    MATCH = 2

    @classmethod
    def values(cls):
        """Return a list of all enum values"""
        return [member.value for member in cls]
    
    @classmethod
    def names(cls):
        """Return a list of all enum names"""
        return [member.name for member in cls]

class TaskLabels(str, Enum):
    """Enumeration of allowed labels."""
    HEATMAPS_CONTOUR_LINES = "Heatmaps, Contour Lines"
    TASK_SET_04 = "Task Set 04"
    SPATIAL_OPERATIONS = "Spatial operations"
    TASK_SET_03 = "Task Set 03"
    PROCESS_MERGE_VISUALIZE = "Process, Merge, Visualize"
    TASK_SET_02 = "Task Set 02"
    MERGE_VISUALIZE = "Merge, Visualize"
    TASK_SET_01 = "Task Set 01"
    VAGUE = "Vague"
    HARD = "Hard"
    CONTROL = "Control question"

NO_LABEL = "<NO_LABEL>"