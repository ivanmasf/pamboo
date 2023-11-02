from __future__ import annotations
from typing import Optional
import os
import warnings

import openai
import pandas as pd

# openai.api_key = os.getenv("OPENAI_API_KEY")

OPENAI_LLMS = ["gpt-3.5-turbo", "text-ada-001"]
AVAILABLE_LLMS = {"OpenAI": OPENAI_LLMS}

### Prompt Templates
# Feel free to alter them here in order to use less tokens or get better results

# def initial_task_prompt(task_prompt):
#     prompt = 'You are an expert Data Scientist. Below, you will be given a task to solve.'
#     prompt += ' Your answer can take only two forms:'
#     prompt += '\n1. If the code to solve the task fits in one prompt: ONLY contain Python code. Any comments must be #commented.'
#     prompt += '\n2. Otherwise: ONLY bullet points numbered in format 1. ,'
#     prompt += ' where each bullet point is a smaller step to complete main task you were given.'
#     prompt += '\nHere is your task:\n'
#     prompt += task_prompt
#     return prompt


def initial_task_prompt(task_prompt):
    prompt = (
        "You are an expert Data Scientist. Below, you will be given a task to solve."
    )
    prompt += " Your answer can ONLY contain python code. DO NOT add any description unless in python comments."
    prompt += "\nHere is your task:\n"
    prompt += task_prompt
    return prompt


### Classes


class LLM:
    """
    If the user just wants to prompt an LLM, they can instantiate one without an Agent.

    Otherwise an LLM can only belong to one Agent.
    Reciprocally, an Agent can only have one LLM.
    """

    def __init__(
        self, new_model_name=None, agent: Optional[Agent] = None, var_dict: dict = {}
    ):
        self.agent = agent
        self.model_provider = None
        self._model_name = None
        self._write_prompt_code = None
        self.var_dict = var_dict
        self.prompts_sent = []
        self.code_run = []

        if new_model_name is not None:
            self.set_model_name(new_model_name)

    @property
    def model_name(self):
        return self._model_name

    def set_model_name(self, new_model_name):
        self._model_name = new_model_name
        self._initialize_model()

    def _initialize_model(self):
        # Raise error if no model_name to initialize
        if self._model_name is None:
            message = self._error_message(f"Empty model_name to initialize LLM")
            raise ValueError(message)

        # Let's try to find the model_name in the ones that are implemented
        model_provider = None
        for llm_provider, models in AVAILABLE_LLMS.items():
            if self._model_name in models:
                model_provider = llm_provider
                break

        # If no model provider was found for the current model_name, raise error
        if model_provider is None:
            message = self._error_message(
                f"model_name '{self._model_name}' not found in AVAILABLE_LLMS"
            )
            raise ValueError(message)

        self.model_provider = model_provider
        self._initialize_prompt_code()

    def _initialize_prompt_code(self):
        """
        This method creates the function that writes code to prompt the LLM based on what provider we get it from.
        If it's an OpenAI LLM, we use its API, if its Llama, we use llama_cpp, and so on.
        """
        if self.model_name == "text-ada-001":

            def write_prompt_code(prompt):
                return f"""
response = openai.ChatCompletion.create(
    model={repr(self._model_name)},
    prompt={repr(prompt)}
)
"""

        elif self.model_name == "gpt-3.5-turbo":

            def write_prompt_code(prompt):
                # code = 'response = openai.ChatCompletion.create('
                # code += f'\n    model={repr(self._model_name)},'
                # code += '\n    messages=['
                # code += '\n        {"system": "You are a Senior Data Analyst and expert in Pandas. You are always concise and precise."},'
                # code += f'        {{"role": "user", "content": {}}},'
                # code += '\n    ])'
                A = self._model_name
                B = prompt
                code = f"""response = openai.ChatCompletion.create(
    model='{A}',
    messages=[
        {{"role": "user", "content": {repr(B)}}},
    ]
)"""
                return code

            self._write_prompt_code = write_prompt_code
        else:
            message = self._err_msg(
                f"No prompt_code function was implemented for model_provider '{self.model_provider}'"
            )
            raise ValueError(message)

    def _err_msg(self, message):
        """
        Appends the agent name on error message if this LLM has an agent.
        """
        if self.agent:
            return message + f" for LLM of Agent {self.agent.name}."
        else:
            return message + "."

    @property
    def prompt_code(
        self,
    ):
        """
        The user can ask an LLM for an example of what is its prompt code
        """
        if self._write_prompt_code is None:
            return None
        else:
            return self._write_prompt_code("YOUR_PROMPT_HERE")

    def is_model_initialized(self):
        # Do we have a model_name?
        if self._model_name is None:
            message = self._err_msg(f"Model not initialized: No model_name")
            raise ValueError(message)
        if self.model_provider is None:
            message = self._err_msg(f"Model not initialized: No model_provider")
            raise ValueError(message)
        if self._write_prompt_code is None:
            message = self._err_msg(f"Model not initialized: No prompt_code")
            raise ValueError(message)

    def prompt(self, prompt):
        """
        Prompts the LLM with the given message.
        The most important function in this class.

        In order to do this, the model must have been correctly initialized.
        """
        self.is_model_initialized()

        # Let's write the code to prompt the LLM
        prompt_code = self._write_prompt_code(prompt)

        # Execute the code having var_dict available in the namespace
        exec(prompt_code, globals(), self.var_dict)

        # Let's log the prompts and code
        self.prompts_sent.append(prompt)


class Task:
    def __init__(self, name, request_prompt, agent=None):
        self.name = name
        self.request_prompt = request_prompt
        self.solution = None
        self.is_done = False
        self.agent = None

        if agent is not None:
            self.assign_agent(agent)

    def assign_agent(self, agent):
        # Does this task already have an Agent?
        if self.agent is not None:
            # Is it the same agent as the new one?
            if agent != self.agent:
                raise ValueError(
                    f"Trying to assign Agent {agent.name} to Task {self.name}, but the latter already belongs to {self.agent.name}."
                )
        # The task now has a new Agent
        self.agent = agent
        # The agent also gets assigned this task
        if self.agent.task != self:
            agent.assign_task(self)


class Response:
    def __init__(self, rtype, object_dict=None, message=None, plan=None, code=None):
        self.rtype = rtype
        self.object_dict = object_dict
        self.message = message
        self.plan = plan
        self.code = code


class Agent:
    """
    The class that solves tasks for Pamboo.
    It either solves the task itself by writing and running code,
    or it creates a plan and delegates tasks to other Agents.

    An Agent can only have one LLM. Think of the LLM as the "brain" of the Agents.
    """

    def __init__(
        self,
        name: str,
        llm_name: str,
        task: Optional[Task] = None,
        var_dict: dict = {},
        verbose=True,
    ):
        self.name = name
        self.llm_name = llm_name
        self.llm_responses = []
        self.var_dict = var_dict
        self.verbose = verbose
        self.task = None

        # Initializing LLM
        self._initialize_llm()

        # If a task was already provided, start responding right away
        if task is not None:
            self.assign_task(task)

            self.response = self.respond_task()

    def assign_task(self, new_task):
        # Does the agent already have a task?
        if self.task is not None:
            # Is the old task different from the new one?
            if self.task.name != new_task.name:
                if not self.task.is_done:
                    # If old task is not done, raise error
                    raise ValueError(
                        f"Trying to assign task {new_task.name} to Agent {self.name}, but the latter is already working on task {self.task.name}."
                    )
                else:
                    message = f"\nAgent {self.name} is being given new task {new_task.name}, which will overwrite completed task {self.task.name}."
                    message += "\nMake sure that:"
                    message += "\n1 - You don't care if reference to old task is lost"
                    message += "\n2 - This behavior is desired"
                    message += (
                        "\n3 - var_dict and llm_name are properly set for the new task."
                    )
                    warnings.warn(message, RuntimeWarning)
                    # The old task does not belong to this agent anymore
                    # But did it still belong to it?
                    old_task_agent = self.task.agent
                    if not old_task_agent == self:
                        raise ValueError(
                            f"Old task of Agent {self.name} belonged to another agent {old_task_agent}."
                        )
                    # If old task did still belong to this agent, now it doesn't anymore
                    self.task.agent = None
        # The agent now has a new task
        self.task = new_task
        # If task is still the same, we just check if it is correctly assigned to this agent as well
        if self.task.agent != self:
            new_task.assign_agent(self)

    def _make_initial_prompt(self):
        # Can the agent solve the task right away, or does it need to break it into smaller tasks and delegate?
        # Let's create the prompt to know
        if self.verbose:
            self.speak(
                f"Can I solve task {self.task.name} right away or do I create a plan?"
            )

        # Creating the initial prompt
        prompt = initial_task_prompt(self.task.request_prompt)

        # Prompting the llm
        self.llm.prompt(prompt)

    def respond_task(self):
        """
        Here the agent either solves the task or creates a plan to solve it.
        If it has to create a plan, it will delegate the tasks to other agents.
        It will also know what steps need to be done before other steps and delegate accordingly.

        This is a loop that only ends in the following cases:
        1 - Agent found a correct solution by itself
        2 - Agent found a correct solution by delegating tasks to other agents
        2 - Agent decided he has failed at the task
        3 - User interrupted
        """
        if not self.task:
            raise ValueError(
                f"Agent {self.name} cannot start responding to task because it is not set."
            )

        # Make the initial attempt to solve or create plan
        self._make_initial_prompt()

        ### TODO implement different ways to get response depending on model
        # Get LLM response from initial prompt
        self.llm_responses.append(
            self.llm.var_dict["response"]["choices"][0]["message"]["content"]
        )
        self.llm_responses[0] = self.llm_responses[0].replace("```python", "")
        self.llm_responses[0] = self.llm_responses[0].replace("```", "")

        ### TODO implement case where he needs to make a plan instead of solving the task right away
        # Solving task of printing
        self.speak("Here is the solution to the task:\n")
        exec(self.llm_responses[0], globals(), self.var_dict)

        self.speak("Code to achieve solution:")
        print(self.llm_responses[0])
        # while not self.task.is_done:

        #     # Case 1: Agent solved the task - need to check it
        #     if self.response.rtype == 'solution':
        #         print(f'\nAgent {self.name}: Task Solved.')
        #         return

        #     # Case 2: Agent created a plan to solve the task - need to delegate tasks
        #     if self.response.rtype == 'plan':
        #         print(f'\nAgent {self.name}: Plan devised.')
        #         return

        #     # Case 3: Agent failed to create solution or plan
        #     if self.response.rtype == 'failed':
        #         print(f'\nAgent {self.name}: Response Failed.')
        #         return

    def has_task(self):
        if self.task is None:
            return False
        else:
            return True

    def _initialize_llm(self):
        self.llm = LLM(self.llm_name, self, self.var_dict)

    def speak(self, msg):
        print(f"\n{self.name}: {msg}")


class Pamboo:
    """
    This is the class users interact with.
    It receives the main task prompt from the user and gets the ball rolling to solve it by:
    1 - Creating a Task object for the main task
    2 - Creating a Root Agent which is assigned the main task

    When all agents have finished solving the task and have confirmed it is done,
    it then tells the user what was done and where to find the results.
    """

    def __init__(self, task_prompt: str, llm_name: str = "gpt-3.5-turbo"):
        print(f"\nPamboo: Starting work on the following task:\n{task_prompt}")
        self.main_task = Task("main_task", task_prompt)
        self.root_agent = Agent("root", llm_name, self.main_task)


if __name__ == "__main__":

    openai.api_key = input('\nPlease enter your openAI API key: ')

    df = pd.read_csv("test.csv")

    print("\nHere's the content of test.csv, so you can check that the correct answer is 10:")
    print(df)

    pb = Pamboo(
        'load test.csv into a dataframe and print the average of column "b" for row which column "c" is "dog"'
    )
