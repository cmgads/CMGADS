from typing import List, NamedTuple, Union

# These define the structure of the history, and correspond to diff output with
# lines that start with a space, a + and a - respectively.
class Keep(NamedTuple):
    line: str
class Insert(NamedTuple):
    line: str
class Remove(NamedTuple):
    line: str


# See frontier in myers_diff
class Frontier(NamedTuple):
    x: str
    history: List[Union[Keep, Insert, Remove]]


def diff_to_string(diff_result: List[NamedTuple], line_number: bool = True) -> str:
    """
    Returns:
        str: 字符串表示的差异
    """    
    ret = ''
    before_i = 1
    after_i = 1
    for elem in diff_result:
        if isinstance(elem, Keep):
            if line_number: ret += '{i: <4}  {line}\n'.format(i=before_i, line=elem.line)
            else: ret += f'  {elem.line}\n'
            before_i += 1
            after_i += 1
        elif isinstance(elem, Insert):
            if line_number: ret += '{i: <4} + {line}\n'.format(i=after_i, line=elem.line)
            else: ret += f'+ {elem.line}\n'
            after_i += 1
        else:
            if line_number: ret += '{i: <4} - {line}\n'.format(i=before_i, line=elem.line)
            else: ret += f'- {elem.line}\n'
            before_i += 1

    return ret