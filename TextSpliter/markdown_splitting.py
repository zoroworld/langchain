from langchain.text_splitter import RecursiveCharacterTextSplitter,Language
text = """
1. Artificial Neurons
2. History of Neural Networks
    - Early models (Perceptron)
    - Backpropagation and MLPs
    - The "AI Winter" and resurgence of neural networks
    - Emergence of deep learning

3. Perceptron and Multilayer Perceptrons (MLP)
   The perceptron was an early model of a neural network...
"""
# Initialize the splitter
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=500,
    chunk_overlap=50,
)

# Perform the split
chunks = splitter.split_text(text)

print(len(chunks))
print(chunks[0])