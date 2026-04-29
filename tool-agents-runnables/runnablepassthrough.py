from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel,RunnablePassthrough

# create model and parser
model = ChatMistralAI(model="mistral-small-2506")
parser = StrOutputParser()

# create code prompt
code_prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a code generator agent."),
        ("human","{topic}")
    ]
)

# explain prompt
explain_prompt = ChatPromptTemplate([
    ("system","You are a code explainer, who explain code in simple terms."),
    ("human","explain the following code in simple word: \n{code}")
])


# Runnable Passthrough
# Runnable Passthrough is used when we want to keep the original input or some intermediate data while passing it
# through the pipeline. Normally, in a sequence, each step replaces the previous output, so earlier data gets lost. But in
# many real-world scenarios, we need to carry multiple pieces of information together. RunnablePassthrough allows us to
# forward the input as it is, without modifying it, so that it can be used along with other outputs in later steps. In simple
# terms, it helps us preserve data while still continuing the flow of the pipeline.

# 1st sequence for get the code
seq = code_prompt | model | parser

# 2nd sequence for code explainer

seq2 = RunnableParallel(
    {
        "code": RunnablePassthrough(),
        "explain": explain_prompt | model | parser
    }
)

chain =  seq | seq2

result = chain.invoke("write a palindrome code in python")

print(result["code"])
print(result["explain"])