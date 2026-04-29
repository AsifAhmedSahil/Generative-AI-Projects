from dotenv import load_dotenv
load_dotenv()


from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableLambda


# components
model = ChatMistralAI(model="mistral-small-2506")
parser = StrOutputParser()

# two different prompts

short_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in 1-2 lines"
)

details_prompt = ChatPromptTemplate.from_template(
    "Exxplain {topic} in details"
)

# input 
topic  = "machine learning"

chain = RunnableParallel(
    {
        "short": RunnableLambda(lambda x:x["short"]) | short_prompt | model | parser,
        "detailed": RunnableLambda(lambda x:x["detailed"]) | details_prompt | model | parser

    }
)

result = chain.invoke({
    "short": {"topic":"machine learning"},
    "detailed": {"topic":"deep learning"}
})

print(result['short'])
print(result['detailed'])