import textwrap

from pydantic.config import Extra
import warnings
from typing import Dict, List, Any

from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI, BaseLLM, OpenAIChat

SUMMARIZE_PROMPT = PromptTemplate(template="""
Shorten the following memory chunk of an autonomous agent from a first person perspective, using at most {max_tokens} tokens. {instruction_hint}:
content:
{content}
---------
""", input_variables=["content", "instruction_hint", "max_tokens"])


class SummarizeChain(LLMChain, extra=Extra.allow):
    """Prompts the LLM to summarize content as needed"""
    def __init__(self,
                 summarizer_chunk_size: int = 3000,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        # TODO: offer more docarray backends
        self.summarizer_chunk_size = summarizer_chunk_size

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        return cls(prompt=SUMMARIZE_PROMPT, llm=llm, verbose=verbose)

    def predict(self, content, **kwargs: Any) -> str:
        return super().predict(
            content=content,
            **kwargs)


class SummarizeMixin:
    summarize_chain: SummarizeChain

    def summarize(self, content: str, max_tokens: int = 4096, instruction_hint: str = '') -> str:
        response = self.summarize_chain.predict(
            # TODO: should retrieve max tokens from the llm if None
            content=content, instruction_hint=instruction_hint, max_tokens=max_tokens
        )
        return response

    def chunked_summarize(self, content: str, max_tokens: int = 4096, instruction_hint: str = '') -> str:
        num_tokens = self.summarize_chain.llm.get_num_tokens(content)

        if num_tokens > max_tokens:
            avg_chars_per_token = len(content) / num_tokens
            chunk_size = int(avg_chars_per_token * self.summarize_chain.summarizer_chunk_size)
            chunks = textwrap.wrap(content, chunk_size)
            summary_size = int(max_tokens / len(chunks))
            result = ""


            for chunk in chunks:
                result += self.summarize(content=chunk, max_tokens=summary_size, instruction_hint=instruction_hint)
        else:
            return content
        return result


if __name__ == '__main__':
    chain = SummarizeChain.from_llm(OpenAI(model_name="gpt-3.5-turbo"))
    print(chain.predict(
        content="""Artificial intelligence (AI) is intelligence demonstrated by machines, 
        unlike the natural intelligence displayed by humans and animals, which involves 
        consciousness and emotionality. The distinction between the former and the latter 
        categories is often revealed by the acronym chosen. 'Strong' AI is usually labelled 
        as AGI (Artificial General Intelligence) while attempts to emulate 'natural' 
        intelligence have been called ABI (Artificial Biological Intelligence).""",
        instruction_hint="Keep the summary concise and within 50 words.",
        max_tokens=50))
