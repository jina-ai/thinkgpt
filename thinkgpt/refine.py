import warnings
from typing import Dict, List, Any

from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI, BaseLLM
from langchain.chat_models import ChatOpenAI


REFINE_PROMPT = PromptTemplate(template="""
Based on the critics, fix the content provided to you. {instruction_hint}:
content:
{content}
---------
critics:
{critics}
---------
""", input_variables=["critics", "content", "instruction_hint"])


class RefineChain(LLMChain):
    """Prompts the LLM to request to remember memory as needed"""
    def __init__(self, **kwargs):
        super().__init__(prompt=REFINE_PROMPT, **kwargs)


    def predict(self, instruction_hint: str = '', **kwargs: Any) -> str:
        return super().predict(instruction_hint=instruction_hint, **kwargs)


class RefineMixin:
    refine_chain: RefineChain

    def refine(self, content: str, critics: List[str], instruction_hint: str = '') -> str:
        return self.refine_chain.predict(
            content=content, critics="\n".join(critics), instruction_hint=instruction_hint
        )


if __name__ == '__main__':
    chain = RefineChain(llm=ChatOpenAI(model_name='gpt-3.5-turbo'))
    print(chain.predict(
        content="""
import re
    print('hello world')
        """,
        critics="""
  File "/Users/alaeddine/Library/Application Support/JetBrains/PyCharm2022.3/scratches/scratch_166.py", line 2
    print('hello world')
IndentationError: unexpected indent
        """, instruction_hint="Fix the code snippet based on the error provided. Only provide the fixed code snippet between `` and nothing else."))
