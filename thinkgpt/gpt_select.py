from typing import Dict, List, Any, Union, Optional

from langchain import PromptTemplate, LLMChain
from langchain.schema import LLMResult, BaseOutputParser, Generation
from langchain.llms import OpenAI, BaseLLM, OpenAIChat
from langchain.prompts.few_shot import FewShotPromptTemplate

SELECT_EXAMPLES = [
    {
        "question": "Which animal is known as the king of the jungle?",
        "options_text": '\n'.join(["Lion", "Elephant", "Tiger", "Giraffe"]),
        "answer": "Lion"
    },
    {
        "question": "Which planet is closest to the Sun?",
        "options_text": '\n'.join(["Mars", "Earth", "Venus", "Mercury"]),
        "answer": "Mercury"
    },
    {
        "question": "What color do you get when you mix blue and yellow?",
        "options_text": '\n'.join(["Green", "Orange", "Purple", "Brown"]),
        "answer": "Green"
    },
]

SELECT_EXAMPLE_PROMPT = PromptTemplate(template="""
Question:
{question}

Options:
{options_text}

Answer:
{answer}
---------
""", input_variables=["question", "options_text", "answer"])


def format_options(options: List[str]) -> str:
    return "\n".join(options)




class SelectOutputParser(BaseOutputParser[Optional[str]]):
    options: List[str]

    def parse(self, text: str) -> Optional[str]:
        if text.strip().lower() not in [option.lower() for option in self.options]:
            return None
        return text.strip()


class SelectChain(LLMChain):

    @classmethod
    def from_llm(cls, llm: BaseLLM, select_examples: Optional[List] = None, verbose: bool = True) -> LLMChain:
        SELECT_PROMPT = FewShotPromptTemplate(
            prefix="Choose the correct answer for the following question from the provided options.",
            examples=select_examples or SELECT_EXAMPLES,
            example_prompt=SELECT_EXAMPLE_PROMPT,
            suffix="{instruction_hint}\nQuestion:\n{question}\nOptions:\n{options_text}\nAnswer:\n",
            input_variables=["instruction_hint", "question", "options_text"]
        )
        return cls(prompt=SELECT_PROMPT, llm=llm, verbose=verbose)

    def predict(self, question: str, options: List[str], instruction_hint: str = '', **kwargs: Any) -> str:
        options_text = format_options(options)
        result = super().predict(question=question, options_text=options_text, instruction_hint=instruction_hint,
                                 **kwargs)
        return SelectOutputParser(options=options).parse(result)


class SelectMixin:
    select_chain: SelectChain

    def select(self, question: str, options: List[str], instruction_hint: str = '', select_chain: Optional[SelectChain] = None) -> str:
       chain = select_chain or self.select_chain
       return chain.predict(question=question, options=options, instruction_hint=instruction_hint)


if __name__ == '__main__':
    chain = SelectChain.from_llm(OpenAI(model_name="gpt-3.5-turbo"))
    print(chain.predict(question="Which animal is known as the king of the jungle?",
                        options=["Lion", "Elephant", "Tiger", "Giraffe"], instruction_hint=""))
