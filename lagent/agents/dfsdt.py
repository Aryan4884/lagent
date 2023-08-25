import re
import warnings
from typing import Dict, List, Optional, Tuple, Union

from lagent.actions import ActionExecutor
from lagent.agents.base_agent import BaseAgent
from lagent.llms.base_api import BaseAPIModel
from lagent.llms.base_llm import BaseModel
from lagent.schema import ActionReturn, ActionStatusCode, AgentReturn
from lagent.agents.dfs.tree import my_tree, tree_node
from lagent.agents.dfs.prompts import FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION,\
         FORMAT_INSTRUCTIONS_USER_FUNCTION, DIVERSITY_PROMPT
from lagent.agents.dfs.toolbench_load import fetch_api_json, api_json_to_openai_json
from lagent.agents.dfs.utils import process_system_message, react_parser, chat_completion_request
from copy import deepcopy
import time
import json
import random

class ToolBenchProtocol:
    """A wrapper of ReWOO prompt which manages the response from LLM and
    generate desired prompts in a ReWOO format.

    Args:
        planner_prompt (str): prompt template for planner.
        worker_prompt (str): prompt template for workers/actions.
        solver_prompt (str): prompt template for solver.
        reformat_prompt (str): prompt template to regenerate
            response for LLM.
    """

    def __init__(
        self,
        FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION=FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION,
        FORMAT_INSTRUCTIONS_USER_FUNCTION=FORMAT_INSTRUCTIONS_USER_FUNCTION,
        add_tool=True,
    ) -> None:
        self.FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION = FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION
        self.FORMAT_INSTRUCTIONS_USER_FUNCTION = FORMAT_INSTRUCTIONS_USER_FUNCTION
        self.add_tool = add_tool

    def set_api_information(self, 
                            query_json,
                            tool_descriptions,
                            tool_root_dir,
                            process_id=0):
        self.tool_root_dir = tool_root_dir
        self.max_observation_length = 1024
        # self.observ_compress_method = "truncate"
        # self.retriever = retriever
        self.process_id = process_id

        self.tool_names = []
        self.cate_names = []

        self.input_description = query_json["query"]
        self.functions = []
        self.api_name_reflect = {}

        # if self.retriever is not None:
        #     query_json = self.retrieve_rapidapi_tools(self.input_description, args.retrieved_api_nums, args.tool_root_dir)
        #     data_dict = self.fetch_api_json(query_json)
        #     tool_descriptions = self.build_tool_description(data_dict)
        # else:
        data_dict = fetch_api_json(self.tool_root_dir, query_json)
        self.func_list = []; self.functions_str = []
        for k,api_json in enumerate(data_dict["api_list"]):
            standard_tool_name = tool_descriptions[k][0]
            openai_function_json,cate_name, pure_api_name = api_json_to_openai_json(api_json,standard_tool_name)
            self.func_list.append(openai_function_json['name'])
            new_function = '{name}: {description}. {name} ALWAYS have this parameters:{parameter}'.format(name=openai_function_json['name'], description=openai_function_json['description'], parameter=openai_function_json['parameters'])
            self.functions.append(openai_function_json)
            self.functions_str.append(new_function)

            self.api_name_reflect[openai_function_json["name"]] = pure_api_name
            self.tool_names.append(standard_tool_name)
            self.cate_names.append(cate_name)

        finish_func = {
            "name": "Finish",
            "description": "If you believe that you have obtained a result that can answer the task, please call this function to provide the final answer. Alternatively, if you recognize that you are unable to proceed with the task in the current state, call this function to restart. Remember: you must ALWAYS call this function at the end of your attempt, and the only part that will be shown to the user is the final answer, so it should contain sufficient information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "return_type": {
                        "type": "string",
                        "enum": ["give_answer","give_up_and_restart"],
                    },
                    "final_answer": {
                        "type": "string",
                        "description": "The final answer you want to give the user. You should have this field if \"return_type\"==\"give_answer\"",
                    }
                },
                "required": ["return_type"],
            }
        }

        finish_func_str = """Finish:If you think you get the result which can answer the task, call this function to give the final answer. Or, if you think you can't handle the task from this status, call this function to restart. Remember: you should ALWAYS call this function at the end of your try, and the final answer is the ONLY part that will be showed to user, so final answer should contain enough information. Finish ALWAYS have this parameters:{'type': 'object', 'properties': {'return_type': {'type': 'string', 'enum': ['give_answer', 'give_up_and_restart']}, 'final_answer': {'type': 'string', 'description': 'The final answer you want to give the user. You should have this field if \"return_type\"==\"give_answer\"'}}, 'required': ['return_type']}"""

        self.functions.append(finish_func)
        self.functions_str.append(finish_func_str)
        self.functions_str = '\n'.join(self.functions_str)
        self.CALL_MAX_TIME = 3
        self.task_description = f'''You should use functions to help handle the real time user querys. Remember:
        1.ALWAYS call \"Finish\" function at the end of the task. And the final answer should contain enough information to show to the user,If you can't handle the task, or you find that function calls always fail(the function is not valid now), use function Finish->give_up_and_restart.
        2.Do not use origin tool names, use only subfunctions' names.'''
        if self.add_tool:
            self.task_description += 'You have access of the following tools:\n'

            unduplicated_reflection = {}
            for standardize_tool_name, tool_des in tool_descriptions:
                unduplicated_reflection[standardize_tool_name] = tool_des

            for k,(standardize_tool_name, tool_des) in enumerate(unduplicated_reflection.items()):
                striped = tool_des[:512].replace('\n','').strip()
                if striped == "":
                    striped = "None"

                self.task_description += f"{k+1}.{standardize_tool_name}: {striped}\n"

        # self.success = 0

    
class DFSDT(BaseAgent):
    """An implementation of DFSDT (https://arxiv.org/abs/2305.18323)

    Args:
        llm (BaseModel or BaseAPIModel): a LLM service which can chat
            and act as planner / solver.
        action_executor (ActionExecutor): an action executor to manage
            all actions and their response.
        protocol (ReWOOProtocol): a wrapper to generate prompt and
            parse the response from LLM / actions.
        max_turn (int): the maximum number of trails for LLM to generate
            plans that can be successfully parsed by ReWOO protocol.
    """

    def __init__(self,
                 llm,
                 action_executor: ActionExecutor,
                 protocol: ToolBenchProtocol = ToolBenchProtocol(),
                 max_query_count=200,
                 tree_beam_size=2,
                 single_chain_max_step=12,
                 max_turn: int = 2) -> None:
        super().__init__(
            llm=llm, action_executor=action_executor, protocol=protocol)
        self.max_query_count = max_query_count
        self.tree_beam_size = tree_beam_size
        self.single_chain_max_step = single_chain_max_step

        self.restart()

    def restart(self):
        self.status = 0
        self.terminal_node = []
        self.give_up_node = []
        self.now_expand_num = 0
        self.query_count = 0
        self.total_tokens = 0
        
    def to_json(self, answer=False, process=True):

        if process:
            json_obj = {
                "win": self.status == 1,
                "tree": self.tree.to_json_recursive(),
                "forward_args": self.forward_args,
                "compare_candidates": [],
            }
            for node in self.terminal_node:
                if node.pruned == False:  # has answer
                    json_obj["compare_candidates"].append(
                        node.get_chain_result_from_this_node(use_messages=False))
        else:
            json_obj = {}

        if answer:
            json_obj["answer_generation"] = {
                "valid_data": False,
                "query_count": self.query_count,
                "total_tokens": self.total_tokens,
                "final_answer": "",
                "finish_type": "give_answer",
                "function": self._protocol.functions,
                "chain": [],
            }
            for node in self.terminal_node:
                if node.pruned == False:
                    json_obj["answer_generation"]["valid_data"] = True
                    json_obj["answer_generation"]["finish_type"] = "give_answer"
                    json_obj["answer_generation"]["final_answer"] = node.description
                    json_obj["answer_generation"]["train_messages"] = node.get_train_messages_from_this_node(
                    )
                    break
            # do not have final answer, look for give_up
            if json_obj["answer_generation"]["valid_data"] == False:
                if len(self.give_up_node) > 0:
                    random_pos = random.randint(0, len(self.give_up_node) - 1)
                    choose_give_up_node = self.give_up_node[random_pos]
                    json_obj["answer_generation"]["valid_data"] = True
                    json_obj["answer_generation"]["finish_type"] = "give_up"
                    json_obj["answer_generation"]["final_answer"] = choose_give_up_node.description
                    json_obj["answer_generation"]["train_messages"] = choose_give_up_node.get_train_messages_from_this_node()
        return json_obj

    def request_hf_llm(self, _history, functions, process_id):
        prompts = []
        for msg in _history:
            content = msg['content']
            if msg['role'] == 'system' and functions != []:
                content = process_system_message(msg['content'], functions) 
            prompt = dict(role=msg['role'], content=content)
            prompts.append(prompt)

        response = self._llm.generate_from_template(prompts, 512, use_streaming=True)
        decoded_token_len = len(response)
        # react format prediction
        thought, action, action_input = react_parser(response)

        message = {
            "role": "assistant",
            "content": response,
            "function_call": {
                "name": action,
                "arguments": action_input
            }
        }
        return message, 0, decoded_token_len

    def request_api_llm(self, _history, functions, process_id):
        TRY_TIME = 6
        for _ in range(TRY_TIME):
            if _ != 0:
                time.sleep(15)
            json_data = chat_completion_request(
                self._llm.keys[0], _history, functions=functions, process_id=process_id)
            try:
                total_tokens = json_data['usage']['total_tokens']
                message = json_data["choices"][0]["message"]
                if process_id == 0:
                    print(f"[process({process_id})]total tokens: {json_data['usage']['total_tokens']}")

                if "function_call" in message.keys() and "." in message["function_call"]["name"]:
                    message["function_call"]["name"] = message["function_call"]["name"].split(".")[-1]

                return message, 0, total_tokens
            except BaseException as e:
                print(f"[process({process_id})]Parsing Exception: {repr(e)}. Try again.")
                if json_data is not None:
                    print(f"[process({process_id})]OpenAI return: {json_data}")
            

        return {"role": "assistant", "content": str(json_data)}, -1, 0

    def dfs(self, now_node, answer, with_filter=True):
        final_answer_back_length = 2
        prune_back_length = 2

        now_node.expand_num = self.now_expand_num
        self.now_expand_num += 1

        if now_node.get_depth() >= self.single_chain_max_step or now_node.pruned or now_node.is_terminal:
            if now_node.is_terminal:  # final answer
                self.status = 1
                self.terminal_node.append(now_node)
                return final_answer_back_length
            else:
                now_node.pruned = True
                if now_node.observation_code == 4:
                    self.give_up_node.append(now_node)
                    return prune_back_length
                else:
                    return 1

        next_tree_split_nodes = []
        for i in range(self.tree_beam_size):
            temp_now_node = now_node
        
            """If a node have children now, We will prompt the model to generate different nodes than all the existing nodes"""
            delete_former_diversity_message = False
            diversity_message = None
            if len(temp_now_node.children) > 0:

                former_candidates_des = ""
                js_list = []
                for k, child in enumerate(temp_now_node.children):
                    temp_node = child
                    while not temp_node.is_terminal and temp_node.node_type != "Action Input" and len(temp_node.children) > 0:
                        temp_node = temp_node.children[0]
                    if temp_node.node_type == "Action Input":
                        obj_dict = {
                            "name": temp_node.father.description,
                            "arguments": temp_node.description,
                            "function_output": temp_node.observation,
                            "mento-carlo-action-value": temp_node.compute_weight(),
                        }
                        js_list.append(obj_dict)

                if len(js_list) > 0:
                    former_candidates_des = former_candidates_des + \
                        f"{json.dumps(js_list,indent=2)}\n"
                    if temp_now_node.observation != "":
                        former_candidates_des = former_candidates_des + \
                            f"again, your former observation: {temp_now_node.observation}\n"
                    diverse_prompt = DIVERSITY_PROMPT
                    diverse_prompt = diverse_prompt.replace(
                        "{previous_candidate}", former_candidates_des)
                    diversity_message = {
                        "role": "user", "content": diverse_prompt}
                    temp_now_node.messages.append(diversity_message)

                    delete_former_diversity_message = True

            now_depth = temp_now_node.get_depth() // 3

            if isinstance(self._llm, BaseAPIModel):
                llm_call_func = self.request_api_llm
            elif isinstance(self._llm, BaseModel):
                llm_call_func = self.request_hf_llm
            else:
                llm_call_func = self._llm.parse
            new_message, error_code, total_tokens = llm_call_func(temp_now_node.messages, 
                self._protocol.functions, self._protocol.process_id)

            self.query_count += 1
            self.total_tokens += total_tokens
            if self.query_count >= self.max_query_count:  # a big return value will cause the Algo to exit
                return 100000

            # We need to exclude the diversity_message, because it will influence child nodes
            if delete_former_diversity_message:
                temp_now_node.messages[-1]["valid"] = False

            # parse nodes from OpenAI-message like CoT method
            assert new_message["role"] == "assistant"
            if "content" in new_message.keys() and new_message["content"] != None:
                temp_node = tree_node()
                temp_node.node_type = "Thought"
                temp_node.description = new_message["content"]
                child_io_state = deepcopy(temp_now_node.io_state)
                # child_io_state.retriever=None

                temp_node.io_state = child_io_state
                # TODO: add status_check
                temp_node.is_terminal = child_io_state.state == 3   # action_status = 3
                temp_node.messages = deepcopy(temp_now_node.messages)
                temp_node.father = temp_now_node
                temp_now_node.children.append(temp_node)
                temp_node.print(self._protocol.process_id)
                temp_now_node = temp_node

                if error_code != 0:
                    temp_now_node.observation_code = error_code
                    temp_now_node.pruned = True

            if "function_call" in new_message.keys():
                # on_agent_action
                function_name = new_message["function_call"]["name"]
                temp_node = tree_node()
                temp_node.node_type = "Action"
                temp_node.description = function_name
                child_io_state = deepcopy(temp_now_node.io_state)
                # child_io_state.retriever=None

                temp_node.io_state = child_io_state
                temp_node.is_terminal = child_io_state.state == 3
                temp_node.messages = deepcopy(temp_now_node.messages)
                temp_node.father = temp_now_node
                temp_now_node.children.append(temp_node)

                temp_node.print(self._protocol.process_id)
                temp_now_node = temp_node

                function_input = new_message["function_call"]["arguments"]
                temp_node = tree_node()
                temp_node.node_type = "Action Input"
                temp_node.description = function_input
                child_io_state = deepcopy(temp_now_node.io_state)

                # TODO convert to SearchAction
                search_param = dict(
                    action_name=temp_now_node.description, 
                    action_input=function_input,
                    cate_names=self._protocol.cate_names,
                    tool_names=self._protocol.tool_names,
                    api_name_reflect=self._protocol.api_name_reflect,
                    functions=self._protocol.functions,
                )
                action_return = self._action_executor(
                    name='RapidAPISearch',
                    command=search_param,
                )
                # adapt ActionReturn format to toolbench format
                observation, status = action_return.result, action_return.state
                if len(observation) > self._protocol.max_observation_length:
                    observation = observation[:self._protocol.max_observation_length] + '...'

                temp_node.observation = observation
                temp_node.observation_code = status

                temp_node.io_state = child_io_state
                temp_node.is_terminal = child_io_state.state == 3
                temp_node.messages = deepcopy(temp_now_node.messages)
                temp_node.father = temp_now_node
                temp_now_node.children.append(temp_node)
                temp_node.print(self._protocol.process_id)
                temp_now_node = temp_node

                if status != 0:
                    # return code defination can be seen in Downstream_tasks/rapid_api
                    if status == 4:
                        temp_now_node.pruned = True
                    elif status == 1:  # hallucination api name
                        assert "function_call" in new_message.keys()
                        new_message["function_call"]["name"] = "invalid_hallucination_function_name"
                    elif status == 3:  # final answer
                        temp_now_node.is_terminal = True
                        temp_now_node.make_finish(final_answer_back_length)

            temp_now_node.messages.append(new_message)
            if temp_now_node.node_type == "Action Input":
                temp_now_node.messages.append({
                    "role": "function",
                    "name": new_message["function_call"]["name"],
                    "content": temp_now_node.observation,
                })
            return_value = None
            if not with_filter:  # DFSDT
                result = self.dfs(temp_now_node, answer, with_filter)
                if len(self.terminal_node) >= answer:
                    return_value = 10000
                elif result > 1:
                    return_value = result-1

            else:
                next_tree_split_nodes.append(temp_now_node)
            if return_value is not None:
                return return_value

        # assert False, "Should not rearch here"
        # Sort the generated next_tree_split_nodes nodes when normal DFS
        if len(next_tree_split_nodes) > 1:
            # When using normal DFS, if we have many child nodes, we will refer to LLM to compare and choose the best one to expand first
            # remember, this operator will cost extra OpenAI calls.
            LLM_rank_args = {
                "functions": self.io_func.functions,
                "process_id": self.process_id,
                "task_description": self.io_func.task_description,
                "rank_func": rank2_subfix,
            }
            scores, rank_query_count, total_tokens = sum_based_rankn(
                self.llm, LLM_rank_args=LLM_rank_args, candidates=next_tree_split_nodes)
            self.query_count += rank_query_count
            self.total_tokens += total_tokens
            for score, node in zip(scores, next_tree_split_nodes):
                node.prior_score = score
            zip_value = list(
                zip(next_tree_split_nodes, range(len(next_tree_split_nodes))))
            zip_value.sort(
                key=lambda x: x[0].prior_score, reverse=True)  # 先做score高的
            next_tree_split_nodes, filtered_order = zip(*zip_value)
            # if self.process_id == 0:
            #     print(f"score={scores}, filtered order: {filtered_order}")

        '''
        Choose one to expand
        '''
        for i in range(len(next_tree_split_nodes)):
            result = self.dfs(
                next_tree_split_nodes[i], answer)
            if len(self.terminal_node) >= answer:
                return 10000
            elif result > 1:
                now_node.make_finish(2)
                return result - 1

        return 1

    def chat(self, message) -> AgentReturn:
        self.forward_args = locals()
        if "self" in self.forward_args.keys():
            self.forward_args.pop("self")
            
        self.restart()
        self._protocol.set_api_information(**message)

        agent_return = AgentReturn()
        action_return = ActionReturn(url=None, args=None)

        self.tree = my_tree()
        self.tree.root.node_type = "Action Input"
        # NOTE there is no io_state for tree_node
        self.tree.root.io_state = deepcopy(action_return)

        system = self._protocol.FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION
        # system = system.replace("{task_description}",
        #                         self._protocol.task_description)
        system = system.format(
            func_str=self._protocol.functions_str,
            func_list=self._protocol.func_list,
        )
        self.tree.root.messages.append({"role": "system", "content": system})

        user = self._protocol.FORMAT_INSTRUCTIONS_USER_FUNCTION
        user = user.replace("{input_description}",
                            self._protocol.input_description)
        self.tree.root.messages.append({"role": "user", "content": user})
        answer = 1
        final_response = self.dfs(self.tree.root, answer, with_filter=False)

        agent_return.response = final_response

        return agent_return
