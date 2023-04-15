import re
from typing import List, Optional, Type, Dict, Any

import numpy as np
from langchain.llms import OpenAI, BaseLLM, OpenAIChat
from langchain.schema import LLMResult, BaseOutputParser, Generation
from langchain import PromptTemplate, LLMChain
from langchain.embeddings import OpenAIEmbeddings
from docarray import DocumentArray, Document
from langchain.output_parsers import PydanticOutputParser
from langchain.vectorstores import FAISS
from pydantic.config import Extra

from llm_tools.abstract import AbstractMixin, AbstractChain
from llm_tools.condition import ConditionMixin, ConditionChain
from llm_tools.memory import RememberOrExecuteChain, RememberOutputParser, MemoryMixin
from llm_tools.refine import RefineMixin, RefineChain

embeddings_model = OpenAIEmbeddings()


class ThinkGPT(OpenAIChat, MemoryMixin, AbstractMixin, RefineMixin, ConditionMixin, extra=Extra.allow):
    """Wrapper around OpenAI large language models to augment it with memory

    To use, you should have the ``openai`` python package installed, and the
    environment variable ``OPENAI_API_KEY`` set with your API key.
    """

    def __init__(self,
                 memory: DocumentArray = None,
                 remember_or_execute_chain: RememberOrExecuteChain = None,
                 abstract_chain: AbstractChain = None,
                 refine_chain: RefineChain = None,
                 condition_chain: ConditionChain = None,
                 parser: RememberOutputParser = None,
                 verbose=True,
                 # TODO: model name can be specified per mixin
                 **kwargs
                 ):
        super().__init__(**kwargs)
        # TODO: offer more docarray backends
        self.memory = memory or DocumentArray()
        self.openai = OpenAI(model_name=kwargs.get('model_name'))
        self.remember_or_execute_chain = remember_or_execute_chain or RememberOrExecuteChain.from_llm(
            self.openai, verbose=verbose)
        self.abstract_chain = abstract_chain or AbstractChain.from_llm(
            self.openai, verbose=verbose)
        self.refine_chain = refine_chain or RefineChain.from_llm(
            self.openai, verbose=verbose)
        self.condition_chain = condition_chain or ConditionChain.from_llm(
            self.openai, verbose=verbose)
        # TODO: actually not really needed here
        self.parser = parser or RememberOutputParser()


    def generate(
            self, prompts: List[str], stop: Optional[List[str]] = None, limit: int = 5
    ) -> LLMResult:
        # only works for single prompt
        if len(prompts) > 1:
            raise Exception('only works for a single prompt')
        prompt = prompts[0]
        new_prompt = self.remember_or_execute_chain.predict(prompt=prompt, context='Nothing')
        parsed = self.parser.parse(new_prompt)
        if parsed['action'] == 'REMEMBER':
            print('need to remember:', parsed['value'])
            # TODO: actually fit as much context as possible
            # TODO: add remember API
            remembered_elements = self.remember(parsed['value'], limit=5)
            print('remembered elements', remembered_elements)
            # TODO: keep prompting to remember until a condition is met
            result = self.remember_or_execute_chain.predict(prompt=prompt, context='\n'.join(remembered_elements))
            return LLMResult(generations=[[Generation(text=result)]])

        elif parsed['action'] == 'FINISH':
            return LLMResult(generations=[[Generation(text=parsed['value'])]])

        return super().generate(prompts, stop=stop)

    def remember(self, concept: str, limit: int = 5) -> List[str]:
        docs = self.memory.find(np.asarray(embeddings_model.embed_query(concept)), limit=limit)
        return [doc.text for doc in docs]

    def predict(
            self, prompt: str, stop: Optional[List[str]] = None, limit: int = 5
    ) -> str:
        return self.generate([prompt], limit=limit).generations[0][0].text


if __name__ == '__main__':
    llm = ThinkGPT(model_name="gpt-3.5-turbo")

    # print(llm.abstract(observations=[
    #     "in tunisian, I did not eat is \"ma khditech\"",
    #     "I did not work is \"ma khdemtech\"",
    #     "I did not go is \"ma mchitech\"",
    # ], instruction_hint="output the rule in french"))

    print(llm.refine(
        content="""
    import re
        print('hello world')
            """,
        critics=[
            'File "/Users/alaeddine/Library/Application Support/JetBrains/PyCharm2022.3/scratches/scratch_166.py", line 2',
            "  print('hello world')",
            'IndentationError: unexpected indent'
        ],
        instruction_hint="Fix the code snippet based on the error provided. Only provide the fixed code snippet between `` and nothing else."))

    # llm.teach("""
    # LangChain is a python framework for developing applications powered by language models.
    # """)
    # llm.teach("""
    # Langchain offers the OpenAI class, use it when you want to send a request to OpenAI LLMs
    # """)
    # llm.teach("""
    # You can import OpenAI class from langchain like this: from langchain.llms import OpenAI
    # """)
    # llm.teach("""
    # Use the OpenAI class like so:
    # llm = OpenAI(model_name="text-ada-001", n=2, best_of=2)
    # llm("Tell me a joke")
    # 'Why did the chicken cross the road? To get to the other side.'
    # """)
    # llm.teach("""
    # Available model names in open ai: "text-ada-001", "gpt-3.5-turbo", "gpt-4"
    # """)
    #
    # llm.teach("""
    # something else that you will almost never need
    # """)
    #
    # llm.teach("""
    # DocArray allows you to represent your data, in an ML-native way. Put simply, DocArray lets you represent your data in a dataclass-like way, with ML as a first class citizen
    # """)
    #
    # llm.teach("""
    # Generate a DocARray v2 schema:
    # from docarray import BaseDoc
    # from docarray.typing import TorchTensor, ImageUrl
    #
    # # Define your data model
    # class MyDocument(BaseDoc):
    #     description: str
    #     image_url: ImageUrl  # could also be VideoUrl, AudioUrl, etc.
    #     image_tensor: TorchTensor[1704, 2272, 3]  # you can express tensor shapes!
    # """)
    #
    # print(
    #     llm.predict('generate python code that uses langchain to send a request to OpenAI LLM and ask it suggest 3 recipes using gpt-4 model'))
