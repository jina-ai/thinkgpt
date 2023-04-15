import re
from typing import Dict, List

import numpy as np
from langchain import PromptTemplate, LLMChain
from langchain.schema import LLMResult, BaseOutputParser, Generation
from langchain.llms import OpenAI, BaseLLM, OpenAIChat
from langchain.embeddings import OpenAIEmbeddings
from docarray import Document, DocumentArray
embeddings_model = OpenAIEmbeddings()


MEMORY_PROMPT = PromptTemplate(template="""
You are a GPT agent with limited information dating only up to 2021.
To answer user requests, you might need extra information provided.
Try to reply to the user request and if you think your knowledge and the context are not enough to reply to the user's request and you need more information, ask to retrieve this information from your memory by saying "REMEMBER(<information>)".
If you think your knowledge is enough, just reply directly.
The user might give you some extra information as Context. You can use this information to reply to his request. If there is no Context information, the user will say something like the following:
Context: Nothing

For example:

User request: implement a pydantic schema with the following fields: txt of type str and value of type int
Context: Nothing
System: here's an example implementation of a Pydantic schema with the fields "txt" of type str and "value" of type int:

python
Copy code
from pydantic import BaseModel

class TextValueSchema(BaseModel):
    txt: str
    value: int

-----------------

User request: implement a DocArray v2 schema with the following fields: im1 of type ImageUrl and im_tensor of type TorchTensor
Context: Nothing
System: "REMEMBER('DocArray v2 schema example with ImageURL type and TorchTensor type')"

-----------------

User request: implement a DocArray v2 schema with the following fields: im1 of type ImageUrl and im_tensor of type TorchTensor
Context: DocArray allows you to represent your data, in an ML-native way. Put simply, DocArray lets you represent your data in a dataclass-like way, with ML as a first class citizen:
from docarray import BaseDoc
from docarray.typing import TorchTensor, ImageUrl

# Define your data model
class MyDocument(BaseDoc):
    description: str
    image_url: ImageUrl  # could also be VideoUrl, AudioUrl, etc.
    image_tensor: TorchTensor[1704, 2272, 3]  # you can express tensor shapes!

System: 
from docarray import BaseDoc
from docarray.typing import TorchTensor, ImageUrl

# Define your data model
class MyDocument(BaseDoc):
    description: str
    im1: ImageUrl  # could also be VideoUrl, AudioUrl, etc.
    im_tensor: TorchTensor[1704, 2272, 3]  # you can express tensor shapes!


-----------------
Let's start.
Remember, if you don't know something, use the REMEMBER(<information>) command, don't say that you do not know.

User request: {prompt}
Context: {context}
System: 
""", input_variables=["prompt", "context"], )


class RememberOutputParser(BaseOutputParser[Dict]):

    def parse(self, text: str) -> Dict:
        # Greedy search for 1st json candidate.
        match = re.match(r"^REMEMBER\((.*)\)$", text.strip().strip('"\'.'), re.MULTILINE | re.IGNORECASE | re.DOTALL)
        if match:
            return {'action': 'REMEMBER', 'value': match.group(1)}
        else:
            return {'action': 'FINISH', 'value': text.strip()}

class RememberOrExecuteChain(LLMChain):
    """Prompts the LLM to request to remember memory as needed"""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        return cls(prompt=MEMORY_PROMPT, llm=llm, verbose=verbose)


class MemoryMixin:
    memory: DocumentArray

    def teach(self, concept: str, infer=False):
        # todo: if infer=True, augment the concept and summerize it first
        self.memory.append(Document(text=concept, embedding=np.asarray(embeddings_model.embed_query(concept))))

    def remember(self, concept: str, limit: int = 5) -> List[str]:
        docs = self.memory.find(np.asarray(embeddings_model.embed_query(concept)), limit=limit)
        return [doc.text for doc in docs]