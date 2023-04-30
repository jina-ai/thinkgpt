import sys
from io import StringIO
from typing import List, Optional, Dict, Tuple

from langchain.schema import LLMResult, BaseOutputParser
from pydantic.fields import Field
from pydantic.main import BaseModel


class LineSeparatorOutputParser(BaseOutputParser[List]):

    def parse(self, text: str) -> List[str]:
        return text.split('\n')


class PythonREPL(BaseModel):
    """Simulates a standalone Python REPL."""

    globals: Optional[Dict] = Field(default_factory=dict, alias="_globals")
    locals: Optional[Dict] = Field(default_factory=dict, alias="_locals")

    def run(self, command: str) -> Tuple[str, Optional[str]]:
        """Run command with own globals/locals and returns anything printed."""
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        error = None
        try:
            exec(command, self.globals, self.locals)
            sys.stdout = old_stdout
            output = mystdout.getvalue()
        except Exception as e:
            sys.stdout = old_stdout
            output = mystdout.getvalue()
            # TODO: need stack trace as well
            error = str(e)
        return output, error


def get_n_tokens(input: str, model_name: str = 'gpt-3.5-turbo'):
    import tiktoken
    enc = tiktoken.encoding_for_model(model_name)
    res = enc.encode(input)
    return len(res)


def fit_context(text_elememts: List[str], max_tokens: int):
    results = []
    total_tokens = 0
    for element in text_elememts:
        total_tokens += get_n_tokens(element)
        if total_tokens <= max_tokens:
            results.append(element)
        else:
            break
    return results