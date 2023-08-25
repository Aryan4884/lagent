from lagent.actions import ActionExecutor
from lagent.actions.rapid_api_search import RapidAPISearch
from lagent.agents.dfsdt import DFSDT
from lagent.llms.openai import GPTAPI
from lagent.llms.tool_llama import ToolLLaMA
from lagent.llms.huggingface import HFTransformerCasualLM
from lagent.agents.dfs.toolbench_load import generate_task_list, get_white_list
from lagent.agents.dfs.utils import replace_llama_with_condense
from termcolor import colored
import os
import json
from lagent.agents.dfs.prompts_internlm import FORMAT_INSTRUCTIONS_USER_FUNCTION, FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION
from lagent.agents.dfsdt import ToolBenchProtocol

tool_root_dir = 'data/toolenv/tools'
query_dir = 'data/instruction/inference_query_demo.json'
output_dir_path = 'work_dirs/toolllama_dfs'
# set OPEN_API_KEY in your environment or directly pass it with key=''
# model = GPTAPI(model_type='gpt-3.5-turbo', key='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')

# replace_llama_with_condense(4)
model = HFTransformerCasualLM(
    '/cpfs01/shared/public/public_hdd/llmit/ckpt/v210_transformers/st_v210rc1_transformer',
    tokenizer_path="pretrain_models/internlm_v1_1",
    meta_template=[
        dict(
            role='system',
            begin='<|System|>:',
            end='\n'),
        dict(role='user', begin='<|User|>:', end='\n'),
        dict(role='function', begin='<|System|>:', end='\n'),
        dict(
            role='assistant',
            begin='<|Bot|>:',
            end='<eoa>\n',
            generate=True)
    ]
)
protocol = ToolBenchProtocol(
    FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION=FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION,
    FORMAT_INSTRUCTIONS_USER_FUNCTION=FORMAT_INSTRUCTIONS_USER_FUNCTION,
)

api_tool = RapidAPISearch(toolbench_key='')
chatbot = DFSDT(
    llm=model,
    protocol=protocol,
    action_executor=ActionExecutor(actions=[api_tool]),
)

white_list = get_white_list(tool_root_dir)
task_list = generate_task_list(query_dir=query_dir, white_list=white_list)
for idx, task in enumerate(task_list):
    query_id, query_json, tool_des = task
    # output_file_path = os.path.join(output_dir_path,f"{query_id}_dfs.json")
    # if os.path.exists(output_file_path):
    #     continue
    query_kwargs = dict(
        process_id=idx,
        query_json=query_json,
        tool_descriptions=tool_des,
        tool_root_dir=tool_root_dir
    )
    agent_return = chatbot.chat(query_kwargs)
    with open(output_file_path,"w") as writer:
        data = chatbot.to_json(answer=True,process=True)
        data["answer_generation"]["query"] = query_json['query']
        json.dump(data, writer, indent=2)
        success = data["answer_generation"]["valid_data"] and "give_answer" in data["answer_generation"]["final_answer"]
        print(colored(f"[process({idx})]valid={success}", "green"))
