from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# procedure
# prompt(which take poem from tex file) send to model to make poem then parser in string

# create model
load_dotenv()

# Define the model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)


model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template='Write a summary for the following poem - \n {poem}',
    input_variables=['poem']
)

parser = StrOutputParser()

loader = TextLoader('cricket.txt', encoding='utf-8')
docs = loader.load()

# print(type(docs))
# print(len(docs))
# print(docs[0].page_content)
# print(docs[0].metadata)

chain = prompt | model | parser

result = chain.invoke({'poem':docs[0].page_content})

print(result)


