from llm_tools.llm import ThinkGPT

llm = ThinkGPT(model_name="gpt-3.5-turbo")

# Load old memory
old_memory = [
    "I learned about the Fibonacci sequence last week.",
    "The Fibonacci sequence starts with 0, 1 and then each subsequent number is the sum of the two preceding ones.",
    "I used the Fibonacci sequence in a coding challenge.",
]

# Teach the LLM the old memory
llm.memorize(old_memory)

# Induce reflections based on the old memory
new_observations = llm.infer(facts=llm.remember())
print(new_observations)

llm.memorize(new_observations)
