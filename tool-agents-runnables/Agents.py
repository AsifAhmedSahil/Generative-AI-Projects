from dotenv import load_dotenv
load_dotenv()

import os
import requests

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage,ToolMessage
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call

from tavily import TavilyClient

# =========================
# 🌦️ Weather Tool
# =========================

@tool
def get_weather(city:str) -> str:
    """Get current weather of a city"""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={api_key}&units=metric"

    response = requests.get(url)
    data = response.json()

    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Could not fetch weather')}"

    temp =  data["main"]["temp"]
    desc = data["weather"][0]["description"]

    return f"Weather in {city}: {desc}, {temp}°C"

# =========================
# 📰 News Tool (Tavily)
# =========================

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def get_news(city:str) -> str:
    """Get the latest news in the city"""

    response = tavily_client.search(
        query=f"Latest news in {city}",
        search_depth="basic",
        max_results=3
    )

    results = response.get("results",[])

    if not results:
        return f"no news found in {city}"
    
    news_list = []

    for r in results:
        title = r.get("title","No Title")
        url = r.get("url","")
        snippet = r.get("content","")

        news_list.append(
            f"- {title}\n  🔗 {url}\n  📝 {snippet[:100]}..."
        )

    return f"Latest news in {city}:\n\n" + "\n\n".join(news_list)


# =========================
# 🧠 LLM Setup
# =========================

llm = ChatMistralAI(model="mistral-small-2506")

# ******************************************* Work Process *********************

# create agent with custom tool , handling tool calling login using langchain and automate the workflow 1st then add human in tha loop as a middleware 
# 
# ********************************************************

# create agent using langchain and automate the flow -- there are no human in the loop concept 
agent = create_agent(
    model=llm,
    tools=[get_news,get_weather],
    system_prompt="You are a helpful city agent."
)

print("Agent Calling | type 'exit' for quit" )

while True:
    user_input = input("You : ")
    if user_input.lower() == "exit":
        break

    result = agent.invoke({
        "messages" : [{"role":"user","content":user_input}]
    })

    print("bot : ", result['messages'][-1].content )

# here is the middleware , there are human in the loop concept using middleware - using @wrap_tool_call

@wrap_tool_call
def human_approval(request,handler):
    """Ask for human approval before every tool call"""
    tool_name = request.tool_call["name"]
    confirm = input(f"Agent want to call '{tool_name}'. Approve? (yes/no)")
    if confirm.lower() != "yes":
        return ToolMessage(
            content="Tool call denied by the user...!",
            tool_call_id = request.tool_call["id"]
        )
    
    return handler(request)

agent_with_middleware = create_agent(
    model=llm,
    tools=[get_news,get_weather],
    system_prompt="You are a helpful city agent.",
    middleware=[human_approval]
)

while True:
    user_input = input("You : ")
    if user_input.lower() == "exit":
        break

    result = agent_with_middleware.invoke({
        "messages" : [{"role":"user","content":user_input}]
    })

    print("bot : ", result['messages'][-1].content )


