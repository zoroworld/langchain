from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('dl-curriculum.pdf')
docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    separators=["\n\n", "\n", " ", ""]
)

result = splitter.split_documents(docs)

# print(result)
# print(result[0])
# print(result[0].metadata['total_pages'])
print(result[1].page_content)

 # [
 #    Document(
 #        metadata={
 #            'producer': 'Skia/PDF m131 Google Docs Renderer',
 #            'creator': 'PyPDF',
 #            'creationdate': '',
 #            'title': 'Deep Learning Curriculum',
 #            'source': 'dl-curriculum.pdf',
 #            'total_pages': 23,
 #            'page': 0,
 #            'page_label': '1'
#            },
 #        page_content='
#        CampusXDeepLearningCurriculum\nA.
#        ArtificialNeuralNetworkandhowtoimprovethem\n1.
#        BiologicalInspiration\n●
#        Understandingtheneuronstructure●
#        Synapsesandsignaltransmission●
#        Howbiologicalconceptstranslatetoart'),
 # ]

