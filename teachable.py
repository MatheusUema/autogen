from enum import auto
import os
from dotenv import load_dotenv
import autogen
from autogen.agentchat.contrib.teachable_agent import TeachableAgent
from autogen import UserProxyAgent

load_dotenv()
api_key = os.getenv('API_KEY')
config_list = [
    {
        'model': 'gpt-4',
        'api_key': api_key
    }
]

llm_config = {
    "timeout": 60,
    "seed" : 42, #cacheing
    "config_list": config_list,
    "temperature": 0, #how creative ai response will be
}


llm_config = {
    "config_list": config_list,
    "timeout": 60,
    "cache_seed": None,  # Use an int to seed the response cache. Use None to disable caching.
}

teach_config={
    "verbosity": 0,  # 0 for basic info, 1 to add memory operations, 2 for analyzer messages, 3 for memo lists.
    "reset_db": True,  # Set to True to start over with an empty database.
    "path_to_db_dir": "./tmp/notebook/teachable_agent_db",  # Path to the directory where the database will be stored.
    "recall_threshold": 1.5,  # Higher numbers allow more (but less relevant) memos to be recalled.
}

try:
    from termcolor import colored
except ImportError:

    def colored(x, *args, **kwargs):
        return x
    
teachable_agent = TeachableAgent(
    name="teachableagent",
    llm_config=llm_config,
    teach_config=teach_config)

user = UserProxyAgent(
    name="user",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: True if "TERMINATE" in x.get("content") else False,
    max_consecutive_auto_reply=0,
)

def _reset_agents():
    teachable_agent.reset()
    user.reset()

def teachable_chat():
    _reset_agents()
    text = "What is the Vicuna model?"
    user.initiate_chat(teachable_agent, message=text, clear_history=True)

    text = "Vicuna is a 13B-parameter language model released by Meta."
    user.initiate_chat(teachable_agent, message=text, clear_history=False)

    text = "What is the Orca model?"
    user.initiate_chat(teachable_agent, message=text, clear_history=False)

    text = "Orca is a 13B-parameter language model released by Microsoft. It outperforms Vicuna on most tasks."
    user.initiate_chat(teachable_agent, message=text, clear_history=False)

    teachable_agent.learn_from_user_feedback()

    text = "How does the Vicuna model compare to the Orca model?"
    user.initiate_chat(teachable_agent, message=text, clear_history=True)

# teachable_chat()
    
def math_chat():
    text = """Consider the identity:  
9 * 4 + 6 * 6 = 72
Can you modify exactly one integer (and not more than that!) on the left hand side of the equation so the right hand side becomes 99?
-Let's think step-by-step, write down a plan, and then write down your solution as: "The solution is: A * B + C * D".
"""
    user.initiate_chat(teachable_agent, message=text, clear_history=True)

    text = """Consider the identity:  
9 * 4 + 6 * 6 = 72
Can you modify exactly one integer (and not more than that!) on the left hand side of the equation so the right hand side becomes 99?
-Let's think step-by-step, write down a plan, and then write down your solution as: "The solution is: A * B + C * D".

Here's some advice that may help:
1. Let E denote the original number on the right.
2. Let F denote the final number on the right.
3. Calculate the difference between the two, G = F - E.
4. Examine the numbers on the left one by one until finding one that divides evenly into G, where negative integers are allowed.
5. Calculate J = G / H. This is the number of times that H divides into G.
6. Verify that J is an integer, and that H * J = G.
7. Find the number on the left which is multiplied by H, and call it K.
8. Change K to K + J.
9. Recompute the value on the left, and verify that it equals F.
Finally, write down your solution as: "The solution is: A * B + C * D".
"""
    user.initiate_chat(teachable_agent, message=text, clear_history=False)

    teachable_agent.learn_from_user_feedback()

    text = """Consider the identity:  
    9 * 4 + 6 * 6 = 72
    Can you modify exactly one integer (and not more than that!) on the left hand side of the equation so the right hand side becomes 99?
    -Let's think step-by-step, write down a plan, and then write down your solution as: "The solution is: A * B + C * D".
    """
    user.initiate_chat(teachable_agent, message=text, clear_history=True)

    text = """Consider the identity:  
8 * 3 + 7 * 9 = 87
Can you modify exactly one integer (and not more than that!) on the left hand side of the equation so the right hand side becomes 59?
-Let's think step-by-step, write down a plan, and then write down your solution as: "The solution is: A * B + C * D".
"""
    user.initiate_chat(teachable_agent, message=text, clear_history=False)

math_chat()