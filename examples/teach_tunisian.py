from llm_tools.llm import ThinkGPT

llm = ThinkGPT(model_name="gpt-3.5-turbo")

rules = llm.abstract(observations=[
    "in tunisian, I did not eat is \"ma khditech\"",
    "I did not work is \"ma khdemtech\"",
    "I did not go is \"ma mchitech\"",
], instruction_hint="output the rule in french")
llm.memorize(rules)

llm.memorize("in tunisian, I went is \"mchit\"")

task = "translate to Tunisian: I didn't go"
print(llm.predict(task, remember=llm.remember(task)))
