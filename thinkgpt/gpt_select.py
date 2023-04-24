from typing import Dict, List, Any, Union, Optional

from langchain import PromptTemplate, LLMChain
from langchain.schema import LLMResult, BaseOutputParser, Generation
from langchain.llms import OpenAI, BaseLLM
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.chat_models import ChatOpenAI

SELECT_EXAMPLES = [
    {
        "question": "Which animal is known as the king of the jungle?",
        "options_text": '\n'.join(["Lion", "Elephant", "Tiger", "Giraffe"]),
        "answer": "Lion"
    },
    {
        "question": "What color do you get when you mix blue and yellow?",
        "options_text": '\n'.join(["Green", "Orange", "Purple", "Brown"]),
        "answer": "Green"
    },
    {
        "question": "Which animal is a carnivore?",
        "options_text": '\n'.join(["Lion", "Elephant", "Tiger", "Giraffe"]),
        "answer": "Lion\nTiger"
    },
    {
        "question": "Which of these are plants?",
        "options_text": '\n'.join(["Lily", "Giraffe", "Cactus", "Rose"]),
        "answer": "Lily\nGiraffe\nCactus\nRose"
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




class SelectOutputParser(BaseOutputParser[List[str]]):
    options: List[str]

    def parse(self, text: str) -> List[str]:
        results = []
        answers = text.split('\n')
        for answer in answers:
            if answer.strip().lower() in [option.lower() for option in self.options]:
                results.append(answer.strip())
        return results


class SelectChain(LLMChain):
    def __init__(self, select_examples: Optional[List] = None, **kwargs):
        SELECT_PROMPT = FewShotPromptTemplate(
            prefix="Choose the correct answer(s) for the following question from the provided options. If there are "
                   "many answers, return each one in a separate line. {num_choices_hint}",
            examples=select_examples or SELECT_EXAMPLES,
            example_prompt=SELECT_EXAMPLE_PROMPT,
            suffix="{instruction_hint}\nQuestion:\n{question}\nOptions:\n{options_text}\nAnswer:\n",
            input_variables=["instruction_hint", "question", "options_text", "num_choices_hint"]
        )
        super().__init__(prompt=SELECT_PROMPT, **kwargs)

    def predict(self, question: str, options: List[str], instruction_hint: str = '', num_choices: int = None, **kwargs: Any) -> List[str]:
        num_choices_hint = f"Return exactly {num_choices} answer" if num_choices else ''
        options_text = format_options(options)
        result = super().predict(question=question, options_text=options_text, instruction_hint=instruction_hint, num_choices_hint=num_choices_hint,
                                 **kwargs)
        return SelectOutputParser(options=options).parse(result)


class SelectMixin:
    select_chain: SelectChain

    def select(self, question: str, options: List[str], instruction_hint: str = '', select_chain: Optional[SelectChain] = None, num_choices: int = None) -> List[str]:
       chain = select_chain or self.select_chain
       return chain.predict(question=question, options=options, instruction_hint=instruction_hint, num_choices=num_choices)


if __name__ == '__main__':
    chain = SelectChain(llm=ChatOpenAI(model_name="gpt-3.5-turbo"))
    print(chain.predict(question="Which animal is known as the king of the jungle?",
                        options=["Lion", "Elephant", "Tiger", "Giraffe"], instruction_hint=""))
