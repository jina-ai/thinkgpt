import re
from typing import List, Optional, Type, Dict, Any, Union, Iterable

from langchain.llms import OpenAI, BaseLLM, OpenAIChat
from langchain.schema import LLMResult, BaseOutputParser, Generation
from langchain.embeddings import OpenAIEmbeddings
from docarray import DocumentArray, Document
from pydantic.config import Extra
from llm_tools.helper import PythonREPL

from llm_tools.abstract import AbstractMixin, AbstractChain
from llm_tools.condition import ConditionMixin, ConditionChain
from llm_tools.memory import MemoryMixin, ExecuteWithContextChain
from llm_tools.refine import RefineMixin, RefineChain

embeddings_model = OpenAIEmbeddings()


class ThinkGPT(OpenAIChat, MemoryMixin, AbstractMixin, RefineMixin, ConditionMixin, extra=Extra.allow):
    """Wrapper around OpenAI large language models to augment it with memory

    To use, you should have the ``openai`` python package installed, and the
    environment variable ``OPENAI_API_KEY`` set with your API key.
    """

    def __init__(self,
                 memory: DocumentArray = None,
                 execute_with_context_chain: ExecuteWithContextChain = None,
                 abstract_chain: AbstractChain = None,
                 refine_chain: RefineChain = None,
                 condition_chain: ConditionChain = None,
                 verbose=True,
                 # TODO: model name can be specified per mixin
                 **kwargs
                 ):
        super().__init__(**kwargs)
        # TODO: offer more docarray backends
        self.memory = memory or DocumentArray()
        self.openai = OpenAI(model_name=kwargs.get('model_name'))
        self.execute_with_context_chain = execute_with_context_chain or ExecuteWithContextChain.from_llm(
            self.openai, verbose=verbose)
        self.abstract_chain = abstract_chain or AbstractChain.from_llm(
            self.openai, verbose=verbose)
        self.refine_chain = refine_chain or RefineChain.from_llm(
            self.openai, verbose=verbose)
        self.condition_chain = condition_chain or ConditionChain.from_llm(
            self.openai, verbose=verbose)
        self.mem_cnt = 0
        # TODO: actually not really needed here


    def generate(
            self, prompts: List[str], stop: Optional[List[str]] = None, remember: Union[int, List[str]] = 0
    ) -> LLMResult:
        # only works for single prompt
        if len(prompts) > 1:
            raise Exception('only works for a single prompt')
        prompt = prompts[0]
        if isinstance(remember, int) and remember > 0:
            remembered_elements = self.remember(prompt, limit=5)
            result = self.execute_with_context_chain.predict(prompt=prompt, context='\n'.join(remembered_elements) if remembered_elements else 'Nothing')
        elif isinstance(remember, list):
            result = self.execute_with_context_chain.predict(prompt=prompt, context='\n'.join(remember))
        else:
            return super().generate(prompts, stop=stop)

        return LLMResult(generations=[[Generation(text=result)]])

    def predict(
            self, prompt: str, stop: Optional[List[str]] = None, remember: Union[int, List[str]] = 0
    ) -> str:
        return self.generate([prompt], remember=remember).generations[0][0].text


if __name__ == '__main__':
    llm = ThinkGPT(model_name="gpt-3.5-turbo")

    task = 'Implement python code that uses llm_tools to learn about docarray v2 code and then predict with remembered information about docarray v2. Only give the code between `` and nothing else'
    code = llm.predict(task, remember=llm.remember(task, limit=10, sort_by_order=True))
    # introduce a bug to the code
    code = '  ' + code
    output, error = PythonREPL().run(code.strip('`'))
    if error:
        code = llm.refine(
            code,
            instruction_hint='Fix the provided code. Only print the fixed code between `` and nothing else.',
            critics=[error]
        )
    print(code)

    # print(
    #     llm.predict('generate python code that uses langchain to send a request to OpenAI LLM and ask it suggest 3 recipes using gpt-4 model', remember=5))

    # task = 'generate python code that uses langchain to send a request to OpenAI LLM and ask it suggest 3 recipes using gpt-4 model'
    # if not llm.condition(f'Do you know all the requirements to achieve the following task ? {task}'):
    #     remember = llm.remember(task)
    # else:
    #     remember = 0
    # print(llm.predict(task, remember=remember))
