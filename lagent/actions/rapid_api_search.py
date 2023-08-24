import os
from typing import List, Optional, Tuple, Union

import requests

from lagent.schema import ActionReturn, ActionStatusCode
from .base_action import BaseAction
import json
from termcolor import colored
import time

DEFAULT_DESCRIPTION = """一个可以从谷歌搜索结果的API。
当你需要对于一个特定问题找到简短明了的回答时，可以使用它。
输入应该是一个搜索查询。
"""

class RapidAPISearch(BaseAction):
    """Wrapper around the Serper.dev Google Search API.

    To use, you should pass your serper API key to the constructor.

    Code is modified from lang-chain GoogleSerperAPIWrapper
    (https://github.com/langchain-ai/langchain/blob/ba5f
    baba704a2d729a4b8f568ed70d7c53e799bb/libs/langchain/
    langchain/utilities/google_serper.py)

    Args:
        api_key (str): API KEY to use serper google search API,
            You can create a free API key at https://serper.dev.
        timeout (int): Upper bound of waiting time for an serper request.
        search_type (str): Serper API support ['search', 'images', 'news',
            'places'] types of search, currently we only support 'search'.
        k (int): select first k results in the search results as response.
        description (str): The description of the action. Defaults to
            None.
        name (str, optional): The name of the action. If None, the name will
            be class nameDefaults to None.
        enable (bool, optional): Whether the action is enabled. Defaults to
            True.
        disable_description (str, optional): The description of the action when
            it is disabled. Defaults to None.
    """
    def __init__(self,
                 toolbench_key: str = None,
                 description: str = DEFAULT_DESCRIPTION,
                 name: Optional[str] = None,
                 enable: bool = True,
                 disable_description: Optional[str] = None) -> None:
        super().__init__(description, name, enable, disable_description)

        self.toolbench_key = toolbench_key
        self.service_url = "http://8.218.239.54:8080/rapidapi"


    def __call__(self, **kwargs):
        tool_return = ActionReturn(url=None, args=None)
        msg, status_code = self._search(**kwargs)
        tool_return.state = status_code
        tool_return.result = msg
        return tool_return

    def _search(self, 
                action_name, 
                action_input, 
                cate_names, 
                tool_names, 
                api_name_reflect, 
                functions,
                **kwargs) -> ActionReturn:
        """Need to return an observation string and status code:
            0 means normal response
            1 means there is no corresponding api name
            2 means there is an error in the input
            3 represents the end of the generation and the final answer appears
            4 means that the model decides to pruning by itself
            5 represents api call timeout
            6 for 404
            7 means not subscribed
            8 represents unauthorized
            9 represents too many requests
            10 stands for rate limit
            11 message contains "error" field
            12 error sending request
        """

        if action_name == "Finish":
            try:
                json_data = json.loads(action_input,strict=False)
            except:
                json_data = {}
                if '"return_type": "' in action_input:
                    if '"return_type": "give_answer"' in action_input:
                        return_type = "give_answer"
                    elif '"return_type": "give_up_and_restart"' in action_input:
                        return_type = "give_up_and_restart"
                    else:
                        return_type = action_input[action_input.find('"return_type": "')+len('"return_type": "'):action_input.find('",')]
                    json_data["return_type"] = return_type
                if '"final_answer": "' in action_input:
                    final_answer = action_input[action_input.find('"final_answer": "')+len('"final_answer": "'):]
                    json_data["final_answer"] = final_answer
            if "return_type" not in json_data.keys():
                return "{error:\"must have \"return_type\"\"}", 2
            if json_data["return_type"] == "give_up_and_restart":
                return "{\"response\":\"chose to give up and restart\"}",4
            elif json_data["return_type"] == "give_answer":
                if "final_answer" not in json_data.keys():
                    return "{error:\"must have \"final_answer\"\"}", 2
                
                # self.success = 1 # succesfully return final_answer
                return "{\"response\":\"successfully giving the final answer.\"}", 3
            else:
                return "{error:\"\"return_type\" is not a valid choice\"}", 2
        else:
            for k, function in enumerate(functions):
                if function["name"].endswith(action_name):
                    pure_api_name = api_name_reflect[function["name"]]
                    payload = {
                        "category": cate_names[k],
                        "tool_name": tool_names[k],
                        "api_name": pure_api_name,
                        "tool_input": action_input,
                        "strip": 'truncate',
                        "toolbench_key": self.toolbench_key
                    }
                    # if self.process_id == 0:
                    print(colored(f"query to {cate_names[k]}-->{tool_names[k]}-->{action_name}",color="yellow"))
                    # if self.use_rapidapi_key:
                    #     payload["rapidapi_key"] = self.rapidapi_key
                    #     response = get_rapidapi_response(payload)
                    # else:
                    time.sleep(2) # rate limit: 30 per minute
                    headers = {"toolbench_key": self.toolbench_key}
                    try:
                        response = requests.post(self.service_url, json=payload, headers=headers, timeout=15)
                    except:
                        return json.dumps({"error": f"time out", "response": ""}), 12
                    if response.status_code != 200:
                        return json.dumps({"error": f"request invalid, data error. status_code={response.status_code}", "response": ""}), 12
                    try:
                        response = response.json()
                    except:
                        print(response)
                        return json.dumps({"error": f"request invalid, data error", "response": ""}), 12
                    # 1 Hallucinating function names
                    # 4 means that the model decides to pruning by itself
                    # 5 represents api call timeout
                    # 6 for 404
                    # 7 means not subscribed
                    # 8 represents unauthorized
                    # 9 represents too many requests
                    # 10 stands for rate limit
                    # 11 message contains "error" field
                    # 12 error sending request
                    if response["error"] == "API not working error...":
                        status_code = 6
                    elif response["error"] == "Unauthorized error...":
                        status_code = 7
                    elif response["error"] == "Unsubscribed error...":
                        status_code = 8
                    elif response["error"] == "Too many requests error...":
                        status_code = 9
                    elif response["error"] == "Rate limit per minute error...":
                        print("Reach api calling limit per minute, sleeping...")
                        time.sleep(10)
                        status_code = 10
                    elif response["error"] == "Message error...":
                        status_code = 11
                    else:
                        status_code = 0
                    return json.dumps(response), status_code
                    # except Exception as e:
                    #     return json.dumps({"error": f"Timeout error...{e}", "response": ""}), 5
            return json.dumps({"error": f"No such function name: {action_name}", "response": ""}), 1
