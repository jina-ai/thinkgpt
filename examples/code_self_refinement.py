from llm_tools.llm import ThinkGPT

llm = ThinkGPT(model_name="gpt-3.5-turbo")

print(llm.refine(
    content="""
import re
    print('hello world')
        """,
    critics=[
        'File "/Users/alaeddine/Library/Application Support/JetBrains/PyCharm2022.3/scratches/scratch_166.py", line 2',
        "  print('hello world')",
        'IndentationError: unexpected indent'
    ],
    instruction_hint="Fix the code snippet based on the error provided. Only provide the fixed code snippet between `` and nothing else."))
