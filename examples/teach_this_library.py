from thinkgpt.llm import ThinkGPT
from examples.knowledge_base import knowledge

llm = ThinkGPT(model_name="gpt-3.5-turbo")

llm.memorize(knowledge)

task = 'Implement python code that uses thinkgpt to learn about docarray v2 code and then predict with remembered information about docarray v2. Only give the code between `` and nothing else'
print(llm.predict(task, remember=llm.remember(task, limit=10, sort_by_order=True)))