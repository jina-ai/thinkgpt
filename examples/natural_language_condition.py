from examples.knowledge_base import knowledge
from thinkgpt.llm import ThinkGPT

llm = ThinkGPT(model_name="gpt-3.5-turbo")

task = 'generate python code that uses langchain to send a request to OpenAI LLM and ask it suggest 3 recipes using gpt-4 model'
if not llm.condition(f'Do you know all the requirements to achieve the following task ? {task}'):
    print('learning knowledge')
    llm.memorize(knowledge)

print(llm.predict(task, remember=llm.remember(task, sort_by_order=True, limit=5)))
