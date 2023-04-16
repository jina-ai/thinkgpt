from thinkgpt.llm import ThinkGPT

llm = ThinkGPT(model_name="gpt-3.5-turbo")

# Load old memory
old_memory = [
    "Klaus Mueller is writing a research paper",
    "Klaus Mueller enjoys reading a book on gentrification",
    "Klaus Mueller is conversing with Ayesha Khan about exercising"
]

# Teach the LLM the old memory
llm.memorize(old_memory)

# Induce reflections based on the old memory
new_observations = llm.infer(facts=llm.remember())
print('new thoughts:')
print('\n'.join(new_observations))

llm.memorize(new_observations)
