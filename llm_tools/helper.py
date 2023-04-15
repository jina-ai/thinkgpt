from typing import List

from langchain.schema import LLMResult, BaseOutputParser

class LineSeparatorOutputParser(BaseOutputParser[List]):

    def parse(self, text: str) -> List[str]:
        return text.split('\n')
