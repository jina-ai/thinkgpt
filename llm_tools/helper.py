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