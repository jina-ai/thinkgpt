import re
from typing import Dict, List, Union

import numpy as np
from langchain import PromptTemplate, LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI, BaseLLM, OpenAIChat
from docarray import Document, DocumentArray


EXECUTE_WITH_CONTEXT_PROMPT = PromptTemplate(template="""
Given a context information, reply to the provided request
Context: {context}
User request: {prompt}
""", input_variables=["prompt", "context"], )



class ExecuteWithContextChain(LLMChain):
    """Prompts the LLM to execute a request with potential context"""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        return cls(prompt=EXECUTE_WITH_CONTEXT_PROMPT, llm=llm, verbose=verbose)


class MemoryMixin:
    memory: DocumentArray
    mem_cnt: int
    embeddings_model: OpenAIEmbeddings

    def memorize(self, concept: Union[str, Document, DocumentArray, List]):
        self.mem_cnt += 1
        if isinstance(concept, str):
            doc = Document(text=concept, embedding=np.asarray(self.embeddings_model.embed_query(concept)), tags={'mem_cnt': self.mem_cnt})
            self.memory.append(doc)
        elif isinstance(concept, Document):
            assert concept.embedding is not None
            concept.tags['mem_cnt'] = self.mem_cnt
            self.memory.append(concept)
        elif isinstance(concept, (DocumentArray, list)):
            for doc in concept:
                self.memorize(doc)
        else:
            raise ValueError('wrong type, must be either str, Document, DocumentArray, List')

    def remember(self, concept: Union[str, Document] = None, limit: int = 5, sort_by_order: bool = False) -> List[str]:
        if len(self.memory) == 0:
            return []
        if concept is None:
            return [doc.text for doc in self.memory[-limit:]]
        elif isinstance(concept, str):
            query_input = Document(embedding=np.asarray(self.embeddings_model.embed_query(concept)))
        elif isinstance(concept, Document):
            assert concept.embedding is not None
            query_input = concept
        else:
            raise ValueError('wrong type, must be either str or Document')

        docs = self.memory.find(query_input, limit=limit)[0]
        # memory needs to be sorted in chronological order
        if sort_by_order:
            docs = sorted(docs, key=lambda doc: doc.tags['mem_cnt'])
        return [doc.text for doc in docs]
