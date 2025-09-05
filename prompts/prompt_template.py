from langchain_core.prompts import PromptTemplate

# Create a prompt template
prompt_template = PromptTemplate.from_template("Tell me a joke about {topic}")

# Fill in the template
prompt = prompt_template.format(topic="cats")

print(prompt)
