from langchain_community.tools import ShellTool

shell_tool = ShellTool()

results = shell_tool.invoke('ipconfig')

# print(results)
# print(shell_tool.run({"commands": ["echo 'Hello World!'", "time"]}))