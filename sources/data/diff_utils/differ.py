from diff.common import Insert, Keep, Remove
from typing import List, Literal, NamedTuple, Union
# from .util import is_code, is_file_same
from diff.myers import myers_diff

DiffKeyType = Literal["add", "remove", "modify", "same"]

def fileLineDiff(before, after) -> List[NamedTuple]:
    """
    Returns:
        List[NamedTuple]: 返回改动的行差异分析结果
    """    
    result = myers_diff(before, after)
    return result
    
def getModifiedBitmap(diffResult: List[Union[Keep, Insert, Remove]]) -> List[bool]:
    lineModified: List[bool] = []
    for elem in diffResult:
        if isinstance(elem, Keep):
            lineModified.append(False)
        elif isinstance(elem, Insert):
            lineModified.append(True)
    return lineModified
