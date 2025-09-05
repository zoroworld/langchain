from langchain.prompts import FewShotPromptTemplate, PromptTemplate

example_template = PromptTemplate.from_template("Q: {question}\nA: {answer}")
examples = [
    {"question": "2+2", "answer": "4"},
    {"question": "3+5", "answer": "8"}
]

few_shot = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_template,
    suffix="Q: {new_question}\nA:",
    input_variables=["new_question"]
)

prompt = few_shot.format(new_question="7+9")
print(prompt)
