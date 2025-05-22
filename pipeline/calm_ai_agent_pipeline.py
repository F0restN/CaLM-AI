"""
title: CaLM Agent
author: ydn
date: 2024-09-16
version: 1.0
license: MIT
description: An AI agen on top of Caregiving Language Model (CaLM) can answer caregiving topics for ADRD domain
requirements: openai, einops, langchain-postgres, psycopg[binary], langchain, langchain-huggingface, torch, sentence-transformers, langgraph, semantic-router
"""

import os 
import json 
import operator 
import torch 

from typing import Generator, Iterator, List, Union, Annotated, TypedDict, Dict
from pydantic import BaseModel
from openai import OpenAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings 
from langchain_postgres import PGVector

from langchain.docstore.document import Document
from langchain_core.agents import AgentAction
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers.json import parse_json_markdown 

from langgraph.graph import StateGraph, END 
from semantic_router.utils.function_call import FunctionSchema

## USE Groq for conversational flow + function calling logic (speed is important for now)
GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1"
GROQ_LLAMA31_MODEL = "llama-3.1-70b-versatile"

## init groq client 
groq_client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url=GROQ_API_URL
)

## Prompts Templates 
calm_sys_prompt = """You are a helpful clinician expert in spinal cord injury"""
calm_rag_prompt_template = """<s>[INST] <<SYS>>
Generate a comprehensive, informative and helpful answer for given question solely based on the information provided (Source and Content). You must only use information from the provided contents. Combine content information together into a coherent answer. Do not repeat text. Anytime you cite piece of information for the answer, follow by add its numerical source in ```[]``` format after text you cite. Only cite the most relevant content that answer the question accurately. Format list of References in this markdown format ```[index](Source link)``` and added at the end. Do not include reference that is not cited in the answer.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If provided information is not providing enough related information to answer the question don't make up an answer. If you don't know the answer to a question, say you don't have answer and please don't share false information.
<</SYS>>

Information:
{information}

Question: {question} [/INST]"""
calm_context_template = """Content: {content}\nSource([{source_no}] <{source}>)"""
calm_rag_model = "calm-7b:latest"

system_prompt = """You are CaLM, an AI expert in providing answer related to caregiving in specific ADRD condition.
Given the user's query you must decide what to do with it based on the list of tools provided to you.

Your goal is to provide the user with the best possible answer in positive manner and safe and avoidig in providing false information.

Note, when using a tool, you provide the tool name and the arguments to use in JSON format. For each call, you MUST ONLY use one tool AND the response format must ALWAYS be in the pattern:

```json
{
    "name": "<tool_name>",
    "parameters": {"<tool_input_key>": <tool_input_value>}
}
```

Remember, NEVER use the search tool more than 3x as that can trigger infinite loop.

After using the `ask_adrd_expert_fn` tool, you must return the answer from this tool directly with final_answer tool. Note, if the user asks a question or says something unrelated to caregiving and adrd topic, you must use the `final_answer` tool directly."""

class AgentState(TypedDict):
    """
    Agent state consist of:
    input - input query
    chat_history - chat history to get previous context
    intermediate_steps - either string or agent action's internal process using ReAct style 
    output - agent state output
    """
    input: str
    chat_history: list[BaseMessage]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
    output: dict[str, Union[str, List[str]]]

class AgentAction(BaseModel):
    tool_name: str
    tool_input: dict
    tool_output: str | None = None

    @classmethod
    def from_llm(cls, response: str):
        try:
            # parse output from ollama
            output = parse_json_markdown(response)
            return cls(
                tool_name=output['name'],
                tool_input=output['parameters'],
                # tool_output=output.get("output", None)
            )
        except Exception as e:
            print(f"Error parsing llm output response: {response}")
            raise 

    def __str__(self):
        text = f"Tool: {self.tool_name}\nInput: {self.tool_input}"
        if self.tool_output is not None:
            text += f"\nOutput: {self.tool_output}"
        return text 

def action_to_message(action: AgentAction):
    # create assistant "input" message
    assistant_content = json.dumps({
        "name": action.tool_name,
        "parameters": action.tool_input
    })
    assistant_message = {
        "role": "assistant",
        "content": assistant_content
    }
    # create user "response" message
    user_message = {
        "role": "user",
        "content": action.tool_output
    }

    return [assistant_message, user_message]

def create_scratchpad(intermedite_steps: list[AgentAction]):
    # filter for actions that have a tool output
    intermedite_steps = [action for action in intermedite_steps if action.tool_output is not None]
    # format the intermediate steps into a "assistant" input and "user" response list
    scratch_pad_messages = []
    for action in intermedite_steps:
        scratch_pad_messages.extend(action_to_message(action))
    return scratch_pad_messages

def format_additional_context(contexts: List[Dict]) -> str:
    """Format additional knowledge to match with CaLM RAG prompt format"""
    source_track = []
    additional_context = []

    for i, doc in enumerate(contexts, 1):
        # merge source index 
        if doc.metadata.get("source", 'Source URL Not Found') not in source_track:
            source_track.append(doc.metadata.get("source", 'Source URL Not Found'))
        
        source_idx = i if doc.metadata.get("source", 'Source URL Not Found') not in source_track else source_track.index(doc.metadata.get("source", 'Source URL Not Found')) + 1

        additional_context.append(
            calm_context_template.format(
                content=doc.page_content.replace("\n", " "),
                source_no=source_idx,
                source=doc.metadata.get("source", 'Source URL Not Found')
            )
        )

    return '\n'.join(additional_context)

class Pipeline:
    class Valves(BaseModel):
        calm_embedding_model_id: str
        calm_api_key: str = "calmapikey" 
        calm_api_url: str
        db_host: str
        db_port: str
        db_user: str
        db_pass: str
        db_name: str
        adrd_collection_name: str

    def __init__(self):
        self.name = "CaLM AI Agent"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.valves = self.Valves(
            **{
                "calm_embedding_model_id": os.getenv("CALM_EMBEDDING_MODEL_ID", "BAAI/bge-base-en"),
                "calm_api_url": os.getenv("CALM_API_URL", "http://localhost:11434/v1/"),
                "db_host": os.getenv("DB_HOST", "localhost"),
                "db_port": os.getenv("DB_PORT", "5432"),
                "db_user": os.getenv("DB_USER", "vectordbadmin"),
                "db_pass": os.getenv("DB_PASS", "HelloVectorDB#2130"),
                "db_name": os.getenv("DB_NAME", "vectordb"),
                "adrd_collection_name": os.getenv("ADRD_COLLECTION_NAME", "calm-collection-0917")
            }
        )

    def init_embedding(self):
        """Initi CaLM Embedding Moddel"""
        print("--INIT EMBEDDING--")
        try:
            self.calm_embedding = HuggingFaceEmbeddings(
                model_name=self.valves.calm_embedding_model_id,
                encode_kwargs={"normalized_embeddings": True},
                model_kwargs={"trust_remote_code": True, "device": self.device},
            )
            print("--DONE INIT EMBEDDING--")
        except Exception as e:
            print(e)
        
        return self.calm_embedding
    
    def init_pg_vector(self):
        print("--INIT PGVECTOR--")
        dburi = f"postgresql+psycopg://{self.valves.db_user}:{self.valves.db_pass}@{self.valves.db_host}:{self.valves.db_port}/{self.valves.db_name}"
        
        try:
            self.pgvector_store = PGVector(
                embeddings=self.calm_embedding,
                collection_name=self.valves.adrd_collection_name,
                connection=dburi,
                use_jsonb=True,
            )
            return self.pgvector_store
        except Exception as e:
            print(e)


    def init_calm(self):
        """Init CaLM via ollama api endpoint using openai compatible instance"""
        print("--INIT CaLM--")
        try:
            self.calm = OpenAI(
                api_key=self.valves.calm_api_key,
                base_url=self.valves.calm_api_url
            )
        except Exception as e:
            print(e)
        
        return self.calm 
    
    def retrieve_contexts(self, query, top_n: int = 3) -> List[Document]:
        return []
    
    def llm_generate(self, llm: OpenAI, msgs: List[dict], model: str, temperature: float = 0.4, max_tokens: int = 1024) -> str:
        resp = llm.chat.completions.create(
            model=model, 
            messages=msgs,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content
    
    def ask_adrd_expert_fn(self, query: str) -> str:
        """
        Provides an access to an expert in caregiving in Alzheimer's Disease and Related Dementia domain, providing best answer for any question related to ADRD. 
        Best for getting an answer related to caregiving in ADRD, the answer is baked on trustable sources and literatures.
        """

        # pull relevant contexts 
        docs = self.pgvector_store.similarity_search(query, k=3, filter={})
        fmt_contexts = format_additional_context(docs)

        # prompt format 
        rag_prompt = calm_rag_prompt_template.format(
            question=query,
            information=fmt_contexts
        )

        # model generation 
        generated_output = self.calm.completions.create(
            model=calm_rag_model,
            prompt=rag_prompt
            )
        print(generated_output)

        # re-format output 
        
        output = generated_output.choices[0].text
        return output.strip()
    
    def final_answer(self, answer: str) -> str:
        """
        Returns a natural language response to the user.
        The final natural language answer to the user's question, should provide as much context as possible.
        Do not share false information.
        """
        return answer 
    
    def get_system_tools_prompt(self, system_prompt: str, tools: list[dict]):
        tools_str = "\n".join([str(tool) for tool in tools])
        return (
            f"{system_prompt}\n\n"
            f"You may use the following tools:\n{tools_str}"
        )
    
    def call_llm(self,user_input: str, chat_history: list[dict], intermediate_steps: list[AgentAction]) -> AgentAction:
        # format the intermediate steps into a scratchpad
        scratchpad = create_scratchpad(intermediate_steps)
        # if the scratchpad is not empty, we add small reminder message to the agent about the original question and prevent to infinite loop
        if scratchpad:
            scratchpad += [{
                "role": "user",
                "content": (
                    f"Please continue, as a reminder my query was '{user_input}'."
                    "Only answer to the original query, and nothing else - but use the "
                    "information I provided to you to do so. Please provide as much "
                    "information as possible in the `answer` field of the final_answer tool. "
                )
            }]
            # we determine the list of tools available to the agent based on whether 
            # or not we have already used the ask sci expert fn or tools
            tools_used = [action.tool_name for action in intermediate_steps]
            tools = []
            if "ask_adrd_expert_fn" in tools_used:
                # we do this because the LLM has a tendency to go of the rails and keep searching for the same thing
                tools = [self.final_answer_schema]
                scratchpad[-1]['content'] = " You must now use the `final_answer` tool."
            else:
                # this shouldn't happen, but we include it just in case
                tools = [self.ask_adrd_expert_schema, self.final_answer_schema]
        else:
            # this would indicate we are on the first run, in which case we allow all tools to be used
            tools = [self.ask_adrd_expert_schema, self.final_answer_schema]

        # costruct our list of messages
        messages = [
            {"role": "system", "content": self.get_system_tools_prompt(system_prompt=system_prompt, tools=tools)},
            *chat_history,
            {"role": "user", "content": user_input},
            *scratchpad
        ]

        res = self.llm_generate(
            llm=groq_client,
            msgs=messages, 
            model=GROQ_LLAMA31_MODEL,
            temperature=0.4
            )
        return AgentAction.from_llm(res)
    
    ## Langgraph Functions 
    def calm_ai_agent(self, state: TypedDict):
        print("--RUN CaLM AI AGENT--")
        chat_history = state['chat_history']
        out = self.call_llm(
            user_input=state['input'],
            chat_history=chat_history,
            intermediate_steps=state['intermediate_steps']
        )

        return {
            "intermediate_steps": [out]
        }
    
    # fn for conditional path
    def router(self, state: TypedDict):
        print("--ROUTER--")
        # return the tool name to use
        if isinstance(state['intermediate_steps'], list):
            return state['intermediate_steps'][-1].tool_name
        else:
            # if we output bad format go to final answer
            print('router invalid format')
            return "final_answer"
        
    # tool run handler 
    def run_tool(self, state: TypedDict):
        # use this as helper function so we repeat less code
        tool_name = state['intermediate_steps'][-1].tool_name
        tool_args = state['intermediate_steps'][-1].tool_input
        print(f"RUN_TOOL | {tool_name}.invoke(input={tool_args})")
        # run tool
        out = self.tool_str_to_func[tool_name](**tool_args)
        action_out = AgentAction(
            tool_name=tool_name,
            tool_input=tool_args,
            tool_output=str(out)
        )
        if tool_name == "final_answer": # return output directly for final answer
            return {"output": out}
        elif tool_name == "ask_adrd_expert_fn": # return output directly from ask_adrd_expert_fn
            return {"output": out}
        else:
            return {'intermediate_steps': [action_out]}


    # construct agent's langgraph 
    def init_agent_graph(self):
        graph = StateGraph(AgentState)

        # add nodes to the graph
        graph.add_node('calm_ai_agent', self.calm_ai_agent)
        graph.add_node('ask_adrd_expert_fn', self.run_tool)
        graph.add_node('final_answer', self.run_tool)

        # set entrypoint 
        graph.set_entry_point('calm_ai_agent')

        # conditional path 
        graph.add_conditional_edges(
            source='calm_ai_agent',
            path=self.router
        )

        # create edges from each available tools back to the calm_ai_agent
        # for tool_obj in [self.ask_adrd_expert_schema, self.final_answer_schema]:
        #     tool_name = tool_obj['function']['name']
        #     if tool_name != 'final_answer':
        #         graph.add_edge(tool_name, 'calm_ai_agent')

        # set final_answer to end node
        graph.add_edge('final_answer', END)
        graph.add_edge('ask_adrd_expert_fn', END)

        # compile graph -> agent 
        self.runnable_agent = graph.compile()
        return self.runnable_agent
    
    async def on_startup(self):
        print(f"on_startup:{__name__}")

        
        # perfrom init 
        self.calm_embedding = self.init_embedding()
        self.pgvector_store = self.init_pg_vector()
        self.calm = self.init_calm()

        # convert existing functions to json prompt using semantic router's Function Schema
        self.ask_adrd_expert_schema = FunctionSchema(self.ask_adrd_expert_fn).to_ollama()
        self.final_answer_schema = FunctionSchema(self.final_answer).to_ollama()

        # fmt sys prompt 
        self.agent_sys_prompt = self.get_system_tools_prompt(
            system_prompt=system_prompt,
            tools=[self.ask_adrd_expert_schema, self.final_answer_schema]
        )

        # tool str to function map 
        self.tool_str_to_func = {
            'ask_adrd_expert_fn': self.ask_adrd_expert_fn,
            'final_answer': self.final_answer
        }

        # init runnable agent 
        self.runnable_agent = self.init_agent_graph()

        print("on_startup: done")

        pass 

    async def on_shutdown(self):
        pass 

    # async def on_valves_updated(self):
    #     # This function is called when the valves are updated.
    #     pass

    async def inlet(self, body: dict, user: dict) -> dict:
        # This function is called before the OpenAI API request is made. You can modify the form data before it is sent to the OpenAI API.
        print(f"inlet:{__name__}")

        # print(body)
        # print(user)

        return body
    
    async def outlet(self, body: dict, user: dict) -> dict:
        # This function is called after the OpenAI API response is completed. You can modify the messages after they are received from the OpenAI API.
        print(f"outlet:{__name__}")

        # print(body)
        # print(user)

        return body

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.
        print(f"pipe:{__name__}")

        # # If you'd like to check for title generation, you can add the following check
        # if body.get("title", False):
        #     print("Title Generation Request")

        print(messages)
        print(user_message)
        print(body)

        # test runnable agent 
        resp = self.runnable_agent.invoke({
            'input': user_message,
            'chat_history': messages[-4:-1] # get last 3 messages as a short conversation context for the agent
        })
        print(resp)

        return resp.get('output', 'err... agent... lost signal...')