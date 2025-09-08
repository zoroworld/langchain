from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question','text']
)

parser = StrOutputParser()

url = "https://www.amazon.in/Lenovo-i7-13650HX-39-6cm-300Nits-83DV00LWIN/dp/B0D5D9SVMP/ref=asc_df_B0D5D9SVMP?mcid=577cb542f5d3387288b8059403871b46&tag=googleshopdes-21&linkCode=df0&hvadid=709855510254&hvpos=&hvnetw=g&hvrand=4356921038407694300&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=9153352&hvtargid=pla-2313502813626&gad_source=1&th=1"
loader = WebBaseLoader(url)
docs = loader.load()

chain = prompt | model | parser

result = chain.invoke({'question':'What is the prodcut that we are talking about?', 'text':docs[0].page_content})


print(result)