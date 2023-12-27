from enum import auto
import os
from dotenv import load_dotenv
import autogen

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

assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config,
)

# You can create as much assistant agents as desired, if done, also necessary to define system message to define their role
# assistant = autogen.AssistantAgent(
#     name="CTO",
#     llm_config=llm_config,
#     system_message="Chief Technical office of a tech company"
# )


#User proxy is an agent that acts in behalf of user: it can do things automatically on your behalf, or ask each depth for approval to do these things
#Can have multiple user proxy agents as well
#human input mode -> ALWAYS, TERMINATE, NEVER -> it defines how much manual input you wanna give (always = every single step, terminate = just when task is completed, never)
#max consecutive auto reply = sets the maximum number of times the agents can go back and forth with each other
#is termination message = defines message that ends task
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=5,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "./web"
    },
    llm_config=llm_config,
    system_message="""Reply TERMINATE if the task has been solved at full satisfaction.
Otherwise, reply CONTINUE, or the reason why the task is not solved yet."""
)

task = """
    Write python code to output numbers 1 to 100, and then store the code in a file
"""

user_proxy.initiate_chat(
    assistant,
    message=task 
)

task2 = """
    Change the code in the file you just created to instead output numbers 1 to 200 in intervals of 2
"""

user_proxy.initiate_chat(
    assistant,
    message=task2
)