from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation
from typing import Annotated, List, Sequence, Tuple, TypedDict, Union
from langchain_core.tools import tool
import json
from langchain_experimental.utilities import PythonREPL
from typing import Annotated
from langchain_community.vectorstores.faiss import FAISS
from embedder import FastEmbeddings
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation
from langchain_core.messages import (
    FunctionMessage,
    BaseMessage
)
from tools import python_repl
from langchain_openai import ChatOpenAI
from actors import create_agent, agent_node
from langchain_core.agents import AgentAction, AgentFinish
import operator
import functools
from langchain.chat_models.gigachat import GigaChat
from langchain.tools.retriever import create_retriever_tool

class AgentState(TypedDict):
    # The input string
    input: str
    # The list of previous messages in the conversation
    chat_history: list[BaseMessage]
    # The outcome of a given call to the agent
    # Needs `None` as a valid type, since this is what this will start as
    agent_outcome: Union[AgentAction, AgentFinish, None]
    # List of actions and corresponding observations
    # Here we annotate this with `operator.add` to indicate that operations to
    # this state should be ADDED to the existing values (not overwrite it)
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

class MainActor:
    def __init__(self, ):
        self.embeddings = FastEmbeddings()
        self.token = "Yjg4MTQzMmUtNDAwMS00NDk0LThjOGUtNmU5ZWQ2YzQ4NDQ2OmQ4MWMxZGZiLTFmNGYtNDk5NS05OGQzLTBiMzYyYWJmNjk3OA=="
        
    def build_tools(self):
        papers = FAISS.load_local("papers_db/",
                 embeddings=self.embeddings).as_retriever()
        self.links_retriever = create_retriever_tool(
            papers,
            "поиск_по_связанным_статьям",
            "Этот инструмент позволит тебе обогатить твой ответ за счет поиска по ключевым словам внутри большого учебника по машинному обучению",
)
        base_retriever = FAISS.load_local("fais_db/",
                         embeddings=self.embeddings).as_retriever()
        self.base_retriever = create_retriever_tool(
            base_retriever,
            "поиск_по_учебнику_глубокого_обучения",
            "Этот инструмент позволит тебе обогатить твой ответ за счет поиска по ключевым словам внутри большого учебника по машинному обучению",
        )

        self.tools = [self.links_retriever, self.base_retriever]
        self.tool_executor = ToolExecutor(self.tools)
    

    def self_reflective_rag(self):


    def build_agents(self, ):
        
        self.res_llm = GigaChat(
            credentials=self.token,
            verify_ssl_certs=False,
            model="GigaChat-Pro",
            scope='GIGACHAT_API_CORP',
            temperature=0.5,
            top_p=0.5

        )

        research_agent = create_agent(
        self.res_llm,
        [self.base_retriever, self.links_retriever],
        system_message="Ты исследователь в области глубокого обучения с многолетним опытом анализа статей с научных конференций. \
                        Ты умеешь проводить семинары по глубокому обучению, объясняя в подробностях своим ученикам нюансы теории глубокого обучения. \
                        Ответь на заданный твоим учеником вопрос используя весь свой арсенал знаний, используй научную терминологию там, где это применимо.",
    )
        research_node = functools.partial(agent_node, agent=research_agent, name="Исследователь")
        # Chart Generator
        #ДОБАВИТЬ KNOWLEDGE MAP ПО ГЕНЕРАЦИИ ПСЕВДОКОДА


        self.engineer_llm = GigaChat(
            credentials=self.token,
            verify_ssl_certs=False,
            model="GigaChat-Pro",
            scope='GIGACHAT_API_CORP',
            temperature=0.3,
            top_p=0.2

        )
        engineer_agent = create_agent(
            self.engineer_llm,
            [python_repl],
            system_message="Используй полное и детальное описание алгоритма для того чтобы сгенерировать псевдокод для его выполнения. \
                            Твой ответ должен помочь пользователю понять тонкости работы описанного алгоритма.",
        )

        engineer_node = functools.partial(agent_node, agent=engineer_agent, name="Инженер-программист")

    

        self.lector_llm = GigaChat(
            credentials=self.token,
            verify_ssl_certs=False,
            model="GigaChat-Pro",
            scope='GIGACHAT_API_CORP',
            temperature=0.5,
            top_p=0.7
        )
        
        lector_agent = create_agent(
            self.engineer_llm,
            [python_repl],
            system_message="Используй полное и детальное описание алгоритма для того чтобы сгенерировать псевдокод для его выполнения. \
                            Твой ответ должен помочь пользователю понять тонкости работы описанного алгоритма.",
        )

        lector_node = functools.partial(agent_node, agent=lector_agent, name="Лектор-преподаватель")


    def router(self, state):
        # This is the router
        messages = state["messages"]
        last_message = messages[-1]
        if "function_call" in last_message.additional_kwargs:
            # The previus agent is invoking a tool
            return "call_tool"
        if "FINAL ANSWER" in last_message.content:
            # Any agent decided the work is done
            return "end"
        return "continue"

    def build_graph(self):

        workflow = StateGraph(AgentState)

        workflow.add_node("Researcher", research_node)
        workflow.add_node("Chart Generator", chart_node)
        workflow.add_node("call_tool", self.tool_node)

        workflow.add_conditional_edges(
            "Researcher",
            self.router,
            {"continue": "Chart Generator", "call_tool": "call_tool", "end": END},
        )
        workflow.add_conditional_edges(
            "Chart Generator",
            self.router,
            {"continue": "Researcher", "call_tool": "call_tool", "end": END},
        )

        workflow.add_conditional_edges(
            "call_tool",
            # Each agent node updates the 'sender' field
            # the tool calling node does not, meaning
            # this edge will route back to the original agent
            # who invoked the tool
            lambda x: x["sender"],
            {
                "Researcher": "Researcher",
                "Chart Generator": "Chart Generator",
            },
        )
        workflow.set_entry_point("Researcher")
        graph = workflow.compile()

    def tool_node(self, state):
        """This runs tools in the graph

        It takes in an agent action and calls that tool and returns the result."""
        messages = state["messages"]
        # Based on the continue condition
        # we know the last message involves a function call
        last_message = messages[-1]
        # We construct an ToolInvocation from the function_call
        tool_input = json.loads(
            last_message.additional_kwargs["function_call"]["arguments"]
        )
        # We can pass single-arg inputs by value
        if len(tool_input) == 1 and "__arg1" in tool_input:
            tool_input = next(iter(tool_input.values()))
        tool_name = last_message.additional_kwargs["function_call"]["name"]
        action = ToolInvocation(
            tool=tool_name,
            tool_input=tool_input,
        )
        # We call the tool_executor and get back a response
        response = self.tool_executor.invoke(action)
        # We use the response to create a FunctionMessage
        function_message = FunctionMessage(
            content=f"{tool_name} response: {str(response)}", name=action.tool
        )
        # We return a list, because this will get added to the existing list
        return {"messages": [function_message]}
