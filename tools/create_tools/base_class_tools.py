from langchain.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field

# arg schema using pydantic
class MultiplyInput(BaseModel):
    a: int = Field(..., description="The first number to multiply")
    b: int = Field(..., description="The second number to multiply")

class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "Multiply two numbers"
    args_schema: Type[BaseModel] = MultiplyInput

    def _run(self, a: int, b: int) -> int:
        return a * b

multiply_tool = MultiplyTool()

# invoke with schema-compliant dict
result = multiply_tool.invoke({"a": 3, "b": 3})

print(result)                       # 9
print(multiply_tool.name)           # multiply
print(multiply_tool.description)    # Multiply two numbers
# print(multiply_tool.args_schema.schema())  # shows JSON schema of args

print(multiply_tool.args_schema.model_json_schema())
