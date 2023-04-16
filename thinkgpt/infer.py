import warnings
from typing import Dict, List, Any

from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI, BaseLLM, OpenAIChat
from langchain.prompts.few_shot import FewShotPromptTemplate

from thinkgpt.helper import LineSeparatorOutputParser

INFERENCE_EXAMPLE_PROMPT = PromptTemplate(template="""
Facts:
{facts}

New Observations:
{new_observations}
---------
""", input_variables=["facts", "new_observations"])

examples = [
    {
        "facts": '\n'.join([
            "water boils at 100 degrees Celsius",
            "water freezes at 0 degrees Celsius",
        ]),
        "new_observations": '\n'.join([
            "ice can be formed by cooling water to 0 degrees Celsius or below",
            "steam can be produced by heating water to 100 degrees Celsius or above",
        ])
    },
    {
        "facts": '\n'.join([
            "plants need sunlight, water, and carbon dioxide to perform photosynthesis",
            "photosynthesis is the process by which plants produce glucose and oxygen",
        ]),
        "new_observations": '\n'.join([
            "a plant placed in a dark room will not be able to perform photosynthesis and may eventually die",
            "providing a plant with enough sunlight, water, and carbon dioxide will help it grow and produce oxygen",
        ])
    },
]

INFERENCE_PROMPT = FewShotPromptTemplate(
    prefix="Based on the following facts, infer new observations. Put each new observation in a separate line. {instruction_hint}",
    examples=examples,
    example_prompt=INFERENCE_EXAMPLE_PROMPT,
    suffix="Facts:\n{facts}\nNew Observations:",
    input_variables=["instruction_hint", "facts"]
)


class InferChain(LLMChain):
    """Prompts the LLM to generate new observations based on the given facts"""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        return cls(prompt=INFERENCE_PROMPT, llm=llm, verbose=verbose)

    def predict(self, instruction_hint: str = '', **kwargs: Any) -> str:
        return super().predict(instruction_hint=instruction_hint, **kwargs)


class InferMixin:
    infer_chain: InferChain

    def infer(self, facts: List[str], instruction_hint: str = '') -> List[str]:
        result = self.infer_chain.predict(
            facts="\n".join(facts), instruction_hint=instruction_hint
        )
        return LineSeparatorOutputParser().parse(result)


if __name__ == '__main__':
    chain = InferChain.from_llm(OpenAI(model_name="gpt-3.5-turbo"))
    # examples from the paper https://arxiv.org/abs/2304.03442
    print(chain.predict(facts="\n".join([
        "Klaus Mueller is writing a research paper",
        "Klaus Mueller enjoys reading a book on gentrification",
        "Klaus Mueller is conversing with Ayesha Khan about exercising"
    ])))
