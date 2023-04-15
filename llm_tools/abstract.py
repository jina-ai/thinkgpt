import re
import warnings
from typing import Dict, List, Any

import numpy as np
from langchain import PromptTemplate, LLMChain
from langchain.schema import LLMResult, BaseOutputParser, Generation
from langchain.llms import OpenAI, BaseLLM, OpenAIChat
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.few_shot import FewShotPromptTemplate
from docarray import Document, DocumentArray

examples = [
  {
    "observations": '\n'.join([
        "father stops at red traffic lights",
        "cars start moving when the light turns green",
        "bike yield to pedestrians at crosswalks with pedestrian signals",
        "truck stops at red traffic lights",
    ]),
    "rules": '\n'.join([
        "drivers must stop at a red traffic light and can move in green lights",
        "drivers must yield to pedestrians at designated crosswalks",
    ])
  },
  {
    "observations": '\n'.join([
        "Consider A a set of (X, Y) pairs",
        "first element is (1, 3)",
        "second element is (2, 5)",
        "third element is (3, 7)",
        "forth element is (4, 9)",
    ]),
    "rules": '\n'.join([
        "The relationship between the first element X and the second element Y in the set A can be described by a function: y = f(x) = 2x - 1",
    ])
  },
  {
    "observations": '\n'.join([
        "Fridge of mass 70 kg falls in 3 sec from height 4 meters",
        "pillow of mass 0.5 kg falls in 3 sec from height 4 meters",
        "rock of mass 1 kg falls in 1 sec from height 2 meters",
        "paper of mass 10 gram falls in 1 sec from height 2 meters",
    ]),
    "rules": '\n'.join([
        "all objects fall at the same rate in a vacuum, regardless of their mass",
    ])
  },
]

ABSTRACTION_EXAMPLE_PROMPT = PromptTemplate(template="""
Observations:
{observations}

Rules:
{rules}
---------
""", input_variables=["observations", "rules"])


ABSTRACTION_PROMPT = FewShotPromptTemplate(
    prefix="Infer a rule from the following observations. {instruction_hint}",
    # TODO: examples should be closes to the prefix/goal using example selector so they are easily applicable to specific use cases
    examples=examples,
    example_prompt=ABSTRACTION_EXAMPLE_PROMPT,
    suffix="Observations:\n{observations}\nRules:",
    input_variables=["instruction_hint", "observations"]
)


class RememberOutputParser(BaseOutputParser[Dict]):

    def parse(self, text: str) -> Dict:
        # Greedy search for 1st json candidate.
        match = re.match(r"^REMEMBER\((.*)\)$", text.strip().strip('"\'.'), re.MULTILINE | re.IGNORECASE | re.DOTALL)
        if match:
            return {'action': 'REMEMBER', 'value': match.group(1)}
        else:
            return {'action': 'FINISH', 'value': text.strip()}

class AbstractChain(LLMChain):
    """Prompts the LLM to request to remember memory as needed"""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        if not (hasattr(llm, 'model_name') and llm.model_name == 'gpt-4'):
            warnings.warn(
                "Keep in mind that LLMs except 'gpt-4' do not exhibit as good abstraction abilities as gpt-4"
            )
        return cls(prompt=ABSTRACTION_PROMPT, llm=llm, verbose=verbose)

    def predict(self, instruction_hint: str = '', **kwargs: Any) -> str:
        return super().predict(instruction_hint=instruction_hint, **kwargs)



class MemoryMixin:
    memory: DocumentArray

    def teach(self, concept: str, infer=False):
        # todo: if infer=True, augment the concept and summerize it first
        self.memory.append(Document(text=concept, embedding=np.asarray(embeddings_model.embed_query(concept))))

    def remember(self, concept: str, limit: int = 5) -> List[str]:
        docs = self.memory.find(np.asarray(embeddings_model.embed_query(concept)), limit=limit)
        return [doc.text for doc in docs]


if __name__ == '__main__':
    chain = AbstractChain.from_llm(OpenAI(model_name="gpt-4"))
    print(chain.predict(observations="\n".join([
        "in tunisian, I did not eat is \"ma khditech\"",
        "I did not work is \"ma khdemtech\"",
        "I did not go is \"ma mchitech\"",
    ]), context='Nothing'))
