#ReAct Prompts

FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION = """Answer the following questions as best you can. Specifically, you have access to the following APIs:

{func_str}

Use the following format:
Thought: you should always think about what to do
Action: the action to take, should be one of {func_list}
Action Input: the input to the action

Begin! Remember: (1) Follow the format, i.e,
Thought:
Action:
Action Input:
(2)The Action: MUST be one of the following:{func_list}
(3)If you believe that you have obtained enough information (which can be judge from the history observations) that can answer the task, please call:
Action: Finish
Action Input: {{"return_type": "give_answer", "final_answer": your answer string}}.
Begin!"""

FORMAT_INSTRUCTIONS_USER_FUNCTION = """{input_description}"""

FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION_ZEROSHOT = """Answer the following questions as best you can. Specifically, you have access to the following APIs:

{func_str}

Use the following format:
Thought: you should always think about what to do
Action: the action to take, should be one of {func_list}
Action Input: the input to the action

Begin! Remember: (1) Follow the format, i.e,
Thought:
Action:
Action Input:
(2)The Action: MUST be one of the following:{func_list}
(3)If you believe that you have obtained enough information (which can be judge from the history observations) that can answer the task, please call:
Action: Finish
Action Input: {{"return_type": "give_answer", "final_answer": your answer string}}.
Question: {question}

Here are the history actions and observations:
"""


## Tree prompts

DIVERSITY_PROMPT='''This is not the first time you try this task, all previous trails failed.
Before you generate my thought for this state, I will first show you your previous actions for this state, and then you must generate actions that is different from all of them. Here are some previous actions candidates:
{previous_candidate}
Remember you are now in the intermediate state of a trail, you will first analyze the now state and previous action candidates, then make actions that is different from all the previous.'''

