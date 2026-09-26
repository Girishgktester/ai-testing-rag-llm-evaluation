import os

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

import os
from pathlib import Path
from dotenv import load_dotenv

cwd = Path.cwd()
env_candidates = [
    cwd / ".env",
    cwd / "notebooks" / ".env",
    cwd.parent / "notebooks" / ".env",
]
env_file = next((p for p in env_candidates if p.exists()), None)
if env_file is None:
    raise FileNotFoundError("No .env found. Expected notebooks/.env with OPENAI_API_KEY.")
load_dotenv(env_file, override=True)

key = (os.getenv("OPENAI_API_KEY") or "").strip()
if not key or "paste_your_key" in key:
    raise ValueError(f"Set OPENAI_API_KEY in {env_file}.")

print(f"Loaded env from: {env_file}")
print("OPENAI_API_KEY configured: True")

api_key = os.getenv(key)
if not api_key:
    raise ValueError("Set OPENAI_API_KEY in your environment.")

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=api_key,
)

agent = create_agent(
    model=model,
    tools=[]
)

response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "What is the capital of India?"
        }
    ]
})

print(response["messages"][-1].content)
