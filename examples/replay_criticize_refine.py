from llm_tools.llm import ThinkGPT

llm = ThinkGPT(model_name="gpt-3.5-turbo", verbose=False)

# Load old memory
old_memory = [
    "I learned about the Fibonacci sequence last week.",
    "The Fibonacci sequence starts with 0, 1 and then each subsequent number is the sum of the two preceding ones.",
    "I used the Fibonacci sequence in a coding challenge.",
    'the next number in the Fibonacci sequence after 1 is 1',
    'the fifth number in the Fibonacci sequence is 3 (0, 1, 1, 2, 3)',
    'the Fibonacci sequence can be used in mathematical patterns and algorithms.',
    'the second number in Fibonacci sequence is 2'
]

# Teach the LLM the old memory
llm.memorize(old_memory)

refined_memory = []
for observation in llm.remember(limit=10):
    # Reflect on the old memory and self-criticize
    critic = llm.predict(f"What critic can you say about the following observation? If there is no critic, just say nothing. Observation: {observation}")
    choice = llm.select('given the following observation and criticism, would you remove, refine or keep the observation?\n'
                        f'Observation: {observation}\n'
                        f'Critics: {critic}', options=['refine', 'remove', 'keep'])
    if choice.lower() == 'refine':
        new_observation = llm.refine(content=observation, critics=[critic], instruction_hint='Just give the new observation and nothing else')
        print(f'refined "{observation}" into "{new_observation}"')
        refined_memory.append(new_observation)
    elif choice.lower() == 'remove':
        print(f'removed "{observation}"')
        pass
    elif choice.lower() == 'keep':
        refined_memory.append(observation)

print('------------ new memory ------------')
print('\n'.join(refined_memory))


# Output
# refined "the second number in Fibonacci sequence is 2" into "The corrected observation is that the second number in the Fibonacci sequence is 1."
# ------------ new memory ------------
# I learned about the Fibonacci sequence last week.
# The Fibonacci sequence starts with 0, 1 and then each subsequent number is the sum of the two preceding ones.
# I used the Fibonacci sequence in a coding challenge.
# the next number in the Fibonacci sequence after 1 is 1
# the fifth number in the Fibonacci sequence is 3 (0, 1, 1, 2, 3)
# the Fibonacci sequence can be used in mathematical patterns and algorithms.
# The corrected observation is that the second number in the Fibonacci sequence is 1.
