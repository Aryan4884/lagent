from lagent.actions import ActionExecutor
from lagent.actions.rapid_api_search import RapidAPISearch
from lagent.agents.dfsdt import DFSDT
from lagent.llms.openai import GPTAPI
from lagent.llms.tool_llama import ToolLLaMA
from lagent.agents.dfs.llama_model import ToolLLaMAV1
from lagent.agents.dfs.toolbench_load import generate_task_list, get_white_list
from lagent.agents.dfs.utils import replace_llama_with_condense
from termcolor import colored
import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--query_dir', type=str, default="data/test_query/G1_category_test_query.json")
parser.add_argument('--output_dir', type=str, default="work_dirs/toolllama_origin_dfs_G1_category")
parser.add_argument('--split', type=int, default=1)
parser.add_argument('--idx', type=int, default=0)
args = parser.parse_args()

tool_root_dir = 'data/toolenv/tools'
query_dir = args.query_dir
output_dir_path = args.output_dir
split_size = args.split; curr_id = args.idx

os.makedirs(output_dir_path, exist_ok=True)

# set OPEN_API_KEY in your environment or directly pass it with key=''
# model = GPTAPI(model_type='gpt-3.5-turbo', key='')

replace_llama_with_condense(4)
model = ToolLLaMAV1('pretrain_models/ToolLLaMA-7b')

# model = ToolLLaMA(
#     path='pretrain_models/ToolLLaMA-7b',
#     max_seq_len=8192,
#     meta_template=[
#         dict(
#             role='system',
#             begin='System: ',
#             end='\n'),
#         dict(role='user', begin='User: ', end='\n'),
#         dict(role='function', begin='Function: ', end='\n'),
#         dict(
#             role='assistant',
#             begin='Assistant: ',
#             end='',
#             generate=True)
#     ])


api_tool = RapidAPISearch(toolbench_key='')
chatbot = DFSDT(
    llm=model,
    action_executor=ActionExecutor(actions=[api_tool]),
)

white_list = get_white_list(tool_root_dir)
# white_list = dict()
task_list = generate_task_list(query_dir=query_dir, white_list=white_list)
num_task = len(task_list)
assert num_task % split_size == 0, "split size error"
each_split_size = num_task // split_size
print(f"processing {each_split_size * curr_id}--{each_split_size * (curr_id + 1)}")
task_list = task_list[each_split_size * curr_id: each_split_size * (curr_id + 1)]

for idx, task in enumerate(task_list):
    query_id, query_json, tool_des = task
    output_file_path = os.path.join(output_dir_path,f"{query_id}_dfs.json")
    if os.path.exists(output_file_path):
        continue
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
