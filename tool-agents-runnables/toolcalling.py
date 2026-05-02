from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage

from rich import print

llm = ChatMistralAI(model="mistral-small-2506")

# create tool
@tool
def getting_text_len(text:str) -> int:
    
    """Returns the number of character in a given text"""
    
    return len(text)

tools = {
    "getting_text_len" : getting_text_len
}


# binding tool with llm
llm_with_tool = llm.bind_tools([getting_text_len])



message = []

prompt = input("You: ")
query = HumanMessage(prompt)
message.append(query)

# step 1: llm decide tool
result = llm_with_tool.invoke(message)


# append ai message to  the messages array
message.append(result)

# step 2-4: Execute tool
if result.tool_calls:
    tool_name = result.tool_calls[0]["name"]
    tool_message = tools[tool_name].invoke(result.tool_calls[0])
    print(tool_message)
    message.append(tool_message)
# print(message)

result = llm_with_tool.invoke(message)

print(result.content)




    


