from pydantic import BaseModel
from typing import List, Optional

class TaskLV5(BaseModel):
    thought: str
    conclusion: str

class TaskLV4(BaseModel):
    thought: str
    subtasks: Optional[List[TaskLV5]] = None
    conclusion: str

class TaskLV3(BaseModel):
    thought: str
    # tooluse: Optional[Union[ToolUse, str]] = None
    subtasks: Optional[List[TaskLV4]] = None
    conclusion: str

class TaskLV2(BaseModel):
    thought: str
    # tooluse: Optional[Union[ToolUse, str]] = None
    subtasks: Optional[List[TaskLV3]] = None
    conclusion: str

class Task(BaseModel):
    thought: str
    # subtasks: Optional[List['Task']] = None
    subtasks: Optional[List[TaskLV2]] = None
    conclusion: str

Task.model_rebuild()

class Solution(BaseModel):
    reasoning: List[Task]
    answer: str