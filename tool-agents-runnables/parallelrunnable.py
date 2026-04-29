from dotenv import load_dotenv
load_dotenv()


from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableLambda

# Runnable Sequence
# Now that we understand what runnables are, let’s see how we actually use them in real applications.
# We’re going to learn this in three simple steps.
# First, we’ll start with the most basic and most important pattern a simple sequence.
# In this, we just connect components like a pipeline. The output of one component becomes the input of the
# next. So your flow becomes something like prompt to model to parser. This is the foundation, and honestly,
# most applications start like this.


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

# Parallel Runnable
# Now till this point, we have only seen how to build a single pipeline where the input flows step by step and
# produces one output. But in real-world applications, sometimes we don’t want just one result. We might want
# multiple outputs at the same time, like a short explanation and a detailed explanation, or maybe the same answer
# in different formats. Instead of running the pipeline again and again for each case, we can use a Parallel
# Runnable.
# In this approach, we define multiple pipelines inside a dictionary, and all of them run simultaneously on the
# same input. Each pipeline produces its own output, and in the end, we get all the results together in a structured
# form. So instead of one input giving one output, now one input can give multiple outputs in a single execution,
# which makes our applications more powerful and efficient.

# Runnable Lambda

# Now sometimes in our pipeline, we don’t just want to pass data forward, we might want to slightly modify it or pick
# a specific part of it before sending it to the next step. This is where RunnableLambda comes in. It allows us to write
# a simple Python function and insert it inside our pipeline as a runnable. For example, if our input is a dictionary with
# multiple keys, and we only want to send one specific part of it to a particular pipeline, we can use RunnableLambda
# to extract that part. So instead of the entire input going everywhere, we can control exactly what each component
# receives. In simple terms, RunnableLambda lets us add custom logic inside our runnable flow, making our pipelines
# more flexible and powerful.

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