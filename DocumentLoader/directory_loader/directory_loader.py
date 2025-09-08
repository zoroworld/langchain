from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
import itertools

#Notes
# If pypdfloader is only wrapper the not hold pydf
# after pdf page[1-24] then start pdf2[25 - ]

loader = DirectoryLoader(
    path='books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

# see 1
# docs = loader.load()
# print(len(docs))  # total number of pages across both PDFs
# print(docs[23].metadata)
# print(docs[23].page_content[:200])


#see 2
# docs = loader.lazy_load()
# for i, doc in enumerate(docs, start=1):
#     print(f"Doc {i}:")
#     print("Text:", doc.page_content[:100])  # first 100 chars
#     print("Metadata:", doc.metadata)
#     print("-" * 50)


# see 2
docs = loader.lazy_load()
for document in docs:
    print(document.metadata)