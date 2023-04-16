from typing import Dict, List, Any, Union

from langchain import PromptTemplate, LLMChain
from langchain.schema import LLMResult, BaseOutputParser, Generation
from langchain.llms import OpenAI, BaseLLM, OpenAIChat
from langchain.prompts.few_shot import FewShotPromptTemplate

CONDITION_EXAMPLES = [
    {
        "question": "Is the sky blue?",
        "answer": "True"
    },
    {
        "question": "Do humans have three arms?",
        "answer": "False"
    },
    {
        "question": "Is water wet?",
        "answer": "True"
    },
]

CONDITION_EXAMPLE_PROMPT = PromptTemplate(template="""
Question:
{question}

Answer:
{answer}
---------
""", input_variables=["question", "answer"])

CONDITION_PROMPT = FewShotPromptTemplate(
    prefix="Determine whether the following statement is true or false. Only reply by true or false and nothing else. {instruction_hint}",
    examples=CONDITION_EXAMPLES,
    example_prompt=CONDITION_EXAMPLE_PROMPT,
    suffix="Question:\n{question}\nAnswer:",
    input_variables=["instruction_hint", "question"]
)


class ConditionOutputParser(BaseOutputParser[Union[bool, str]]):

    def parse(self, text: str) -> Union[bool, str]:
        text = text.strip().lower()
        if text == "true":
            return True
        elif text == "false":
            return False
        else:
            return "Wrong format of the answer"


class ConditionChain(LLMChain):

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        return cls(prompt=CONDITION_PROMPT, llm=llm, verbose=verbose)

    def predict(self, question: str, instruction_hint: str = '', **kwargs: Any) -> bool:
        result = super().predict(question=question, instruction_hint=instruction_hint, **kwargs)
        return ConditionOutputParser().parse(result)


class ConditionMixin:
    condition_chain: ConditionChain

    def condition(self, question: str, instruction_hint: str = '') -> bool:
        return self.condition_chain.predict(question=question, instruction_hint=instruction_hint)


if __name__ == '__main__':
    chain = ConditionChain.from_llm(OpenAI(model_name="gpt-3.5-turbo"))
    print(chain.predict(question="Is 2+2 equal to 4?", instruction_hint=""))
