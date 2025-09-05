from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot. Your name is {name}."),
    ("human", "Hello, how are you doing?"),
    ("ai", "I'm doing well, thanks!"),
    ("human", "{user_input}"),
    MessagesPlaceholder("chat_history")
])

prompt = template.invoke(
    {
        "name": "Bob",
        "user_input": "What is your name?",
        "chat_history": [("human", "what's 5 + 2"), ("ai", "5 + 2 is 7")]
    }
)

# Print messages one by one
for msg in prompt.messages:
    print(f"{msg.type.upper()}: {msg.content}")
