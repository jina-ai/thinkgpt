# ThinkGPT 🧠🤖
ThinkGPT is a Python library aimed at implementing Chain of Thoughts for Large Language Models (LLMs), prompting the model to think, reason, and to create generative agents. 
The library aims to help with the following:
* solve limited context with long memory and compressed knowledge
* enhance LLMs' one-shot reasoning with higher order reasoning primitives
* add intelligent decisions to your code base


## Key Features ✨
* Thinking building blocks 🧱:
    * Memory 🧠: GPTs that can remember experience
    * Self-refinement 🔧: Improve model-generated content by addressing critics
    * Abstraction 🌐: Encourages LLMs to generalize rules from examples or observations
    * Inference 💡️: Make educated guesses based on available information
    * Natural Language Conditions 📝: Easily express choices and conditions in natural language
* Efficient and Measurable GPT context length 📐
* Extremely easy setup and pythonic API 🎯 thanks to [DocArray](https://github.com/docarray/docarray)

## Installation 💻
You can install ThinkGPT using pip:

```shell
pip install git+https://github.com/alaeddine-13/thinkgpt.git
```

## API Documentation 📚
### Basic usage:
```python
from thinkgpt.llm import ThinkGPT
llm = ThinkGPT(model_name="gpt-3.5-turbo")
# Make the llm object learn new concepts
llm.memorize(['DocArray is a library for representing, sending and storing multi-modal data.'])
llm.predict('what is DocArray ?', remember=llm.remember('DocArray definition'))
```

### Memorizing and Remembering information
```python
llm.memorize([
    'DocArray allows you to send your data, in an ML-native way.',
    'This means there is native support for Protobuf and gRPC, on top of HTTP and serialization to JSON, JSONSchema, Base64, and Bytes.',
])

print(llm.remember('Sending data with DocArray', limit=1))
```
```text
['DocArray allows you to send your data, in an ML-native way.']
```

### Predicting with context from long memory
```python
from examples.knowledge_base import knowledge
llm.memorize(knowledge)
llm.predict('Implement a DocArray schema with 2 fields: image and TorchTensor', remember=llm.remember('DocArray schemas and types'))
```

### Self-refinement

```python
print(llm.refine(
    content="""
import re
    print('hello world')
        """,
    critics=[
        'File "/Users/user/PyCharm2022.3/scratches/scratch_166.py", line 2',
        "  print('hello world')",
        'IndentationError: unexpected indent'
    ],
    instruction_hint="Fix the code snippet based on the error provided. Only provide the fixed code snippet between `` and nothing else."))

```

```text
import re
print('hello world')
```

One of the applications is self-healing code generation implemented by projects like [gptdeploy](https://github.com/jina-ai/gptdeploy) and [wolverine](https://github.com/biobootloader/wolverine)

### Induce rules from observations
Amount to higher level and more general observations from current observations:
```python
llm.abstract(observations=[
    "in tunisian, I did not eat is \"ma khditech\"",
    "I did not work is \"ma khdemtech\"",
    "I did not go is \"ma mchitech\"",
])
```

```text
['Negation in Tunisian Arabic uses "ma" + verb + "tech" where "ma" means "not" and "tech" at the end indicates the negation in the past tense.']
```

This can help you end up with compressed knowledge that fits better the limited context length of LLMs.
For instance, instead of trying to fit code examples in the LLM's context, use this to prompt it to understand high level rules and fit them in the context.

### Natural language condition
Introduce intelligent conditions to your code and let the LLM make decisions
```python
llm.condition(f'Does this represent an error message ? "IndentationError: unexpected indent"')
```
```text
True
```
### Natural language select
Alternatively, let the LLM choose among a list of options:
```python
llm.select(
    question="Which animal is the king of the jungle?",
    options=["Lion", "Elephant", "Tiger", "Giraffe"]
)
```
```text
Lion
```

## Use Cases 🚀
Find out below example demos you can do with `thinkgpt`
### Teaching ThinkGPT a new language
```python
from thinkgpt.llm import ThinkGPT

llm = ThinkGPT(model_name="gpt-3.5-turbo")

rules = llm.abstract(observations=[
    "in tunisian, I did not eat is \"ma khditech\"",
    "I did not work is \"ma khdemtech\"",
    "I did not go is \"ma mchitech\"",
], instruction_hint="output the rule in french")
llm.memorize(rules)

llm.memorize("in tunisian, I studied is \"9rit\"")

task = "translate to Tunisian: I didn't study"
llm.predict(task, remember=llm.remember(task))
```
```text
The translation of "I didn't study" to Tunisian language would be "ma 9ritech".
```

### Teaching ThinkGPT how to code with `thinkgpt` library
```python
from thinkgpt.llm import ThinkGPT
from examples.knowledge_base import knowledge

llm = ThinkGPT(model_name="gpt-3.5-turbo")

llm.memorize(knowledge)

task = 'Implement python code that uses thinkgpt to learn about docarray v2 code and then predict with remembered information about docarray v2. Only give the code between `` and nothing else'
print(llm.predict(task, remember=llm.remember(task, limit=10, sort_by_order=True)))
```

Code generated by the LLM:
```text
from thinkgpt.llm import ThinkGPT
from docarray import BaseDoc
from docarray.typing import TorchTensor, ImageUrl

llm = ThinkGPT(model_name="gpt-3.5-turbo")

# Memorize information
llm.memorize('DocArray V2 allows you to represent your data, in an ML-native way')


# Predict with the memory
memory = llm.remember('DocArray V2')
llm.predict('write python code about DocArray v2', remember=memory)
```
### Replay Agent memory and infer new observations
Refer to the following script for an example of an Agent that replays its memory and induces new observations.
This concept was introduced in [the Generative Agents: Interactive Simulacra of Human Behavior paper](https://arxiv.org/abs/2304.03442).

```shell
python -m examples.replay_expand_memory
```
```text
new thoughts:
Klaus Mueller is interested in multiple topics
Klaus Mueller may have a diverse range of interests and hobbies
```

### Replay Agent memory, criticize and refine the knowledge in memory
Refer to the following script for an example of an Agent that replays its memory, performs self-criticism and adjusts its memory knowledge based on the criticism.
```shell
python -m examples.replay_criticize_refine
```
```text
refined "the second number in Fibonacci sequence is 2" into "Observation: The second number in the Fibonacci sequence is actually 1, not 2, and the sequence starts with 0, 1."
...
```
This technique was mainly implemented in the [the Self-Refine: Iterative Refinement with Self-Feedback paper](https://arxiv.org/abs/2303.17651)


For more detailed usage and code examples check `./examples`.

