import operator
from prompts import research_prompt
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
import os
import logging
import json
from typing import Annotated, Sequence, TypedDict
from langchain.chat_models.gigachat import GigaChat
from langchain_core.tools import tool
from langchain import hub
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.utils.function_calling import convert_to_gigachat_tool, convert_to_gigachat_function
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage, SystemMessage, AIMessage
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores.faiss import FAISS

from embedder import FastEmbeddings, download_papers, embed_docs
from parser import get_correct_name_from_topk





def paper_search(query:Annotated[str, "Запрос для поиска внутри статьи, по которой пользователь задает вопрос."], 
                 db):
    '''
    Инструмент для доступа к полному тексту статьи.
    Пример: paper_search("О чем рассказывается в данной статье?") -> Данная статья рассказывает о новом методе оптимизации
    больших язоковых моделей
    '''
    logging.warning("---- Used internal RAG ----")
    response = db.similarity_search(query=query)
    answer = " ".join([i.page_content[:2500] for i in response])
    logging.warning(f"CONTEXT LENGTH {len(answer)}")

    return answer


@tool
def ref_search( link:Annotated[str, "Название статьи из списка предоставленных."],
               query:Annotated[str, "Запрос для поиска внутри загруженной статьи."]):
    '''
    Используй этот инструмент для того чтобы создать поисковый индекс из предоставленных тебе статей
    и получить нужную тебе информацию из связанных статей. Принимай взвешенное решение об использовании этого 
    инструмента на основе данных тебе названий связанных статей.
    '''
    references = st.session_state.refs
    name_ref = get_correct_name_from_topk(titles=[i['misc'] for i in references], 
                                label=link)
    
    download_id = references[name_ref]['id']

    doc = download_papers([download_id])
    return answer

class WorkFlow():
    def __init__(self):
        self.tools = [ref_search, paper_search]
       
    def add_current_paper_rag(self, db) -> None:
        self.curr_db = db

    def entry(self, human_input:str):
        self.rag_model = GigaChat(
            credentials=os.environ['GIGA_TOKEN'],
            verify_ssl_certs=False,
            model="GigaChat-Pro-preview",
            scope='GIGACHAT_API_CORP',
            temperature=0.5
        )

        inputs = {
            "messages": [
                SystemMessage(content = f"Пользователь задает тебе вопрос по данной научной статье: {st.session_state.name}.\
                Тебе предоставлены названия статей, связанных со статьей о которой тебе задают вопросы. Ссылки : {[i['misc'] for i in st.session_state.refs if 'misc' in i.keys()][:20]}. \
                Напиши запрос для получения информации из статьи, чтобы наиболее полно ответить на запрос. Внимательно рассмотри предоставленные ссылки. Если считаешь\
                небходимым, то так же напиши название ссылки в формате .\
                Формат твоего ответа: <'Запрос1:запрос для поиска внутри данной тебе статьи'>\
                Пример формирования запроса: <UserQuery: 'Расскажи о том, какие данные используются в этой статье', Запрос1: 'Данные, результат, эксперимент, объем данных'>,\
                <UserQuery: 'Какой алгоритм предлагается в статье?', Запрос1: 'Алгоритм, нововведение, доказательство, сложность, формула, эффективность'>.\
                <UserQuery: 'Какие ограничения метода авторы находят?', Запрос1: 'Выводы, ограничения, домен, данные, модель'>\
                При необходимости получения информации из ссылок: <Запрос2: название ссылки : запрос для поиска внутри статьи из ссылок> \
                Весь текст твоих запросов должен  содержать ключевые слова, отражащие как запрос пользователя и начную сферу, которой принадлежит статья. \
                "),
                HumanMessage(
                    content="UserQuery: " + human_input
                )
            ]
        }
        outputs = self.rag_model.invoke(inputs['messages'])                
        response = self.call_rag_paper(query = outputs.content[0].split('"')[-1])
        return response
   
    def call_rag_paper(self, query):
        paper_info = paper_search(query, 
                                  db=self.curr_db)
        
        rag_output = {"context":paper_info, 
                      "question": "Суммаризуй предоставленный тебе контекст и переведи его на русский язык, не теряя наученые термины и нюансы описания алгоритма."}

        
        prompt = ChatPromptTemplate.from_messages(
          ("human", "Ты ассистент внутри вопросно ответно системы. Твоя цель - четко следовать инструкция и максимально использовать данный тебе контекст. Твои ответы должны быть четкими, чистыми и понятными. Перепроверь свой результат перед тем как\
                возвращать его пользователю. Используй не более трех предложений в своем ответе.\
        Question: {question} \
        Context: {context} \
        Answer:"),
        )

        rag_chain =  prompt | self.rag_model | StrOutputParser()

        response = rag_chain.invoke(rag_output)
        return response


    def call_researcher(self, human_input, rag_response):
        model = GigaChat(
                    credentials=os.environ['GIGA_TOKEN'],
                    verify_ssl_certs=False,
                    model="GigaChat-Pro-preview",
                    scope='GIGACHAT_API_CORP',
                    temperature=0.6,
                    top_p=0.4)
        input_prompt = {"name":st.session_state.name, 
                        "context":rag_response} 
        res_chain = research_prompt | model
        response = res_chain.invoke(input_prompt) 

        if "Запрос1" in response.content:
            output = self.call_rag_paper(response.content.split("Запрос1:")[-1])
            logging.warning(f"first query {output}")

        return response

    def call_engineer(self):
        return

    def execute_steps(self, human_input:str):
        base_paper_data = self.entry(human_input=human_input)
        research_plan = self.call_researcher(human_input=human_input,
                                             rag_response=base_paper_data)

        return research_plan

    def build_agents(self, ):
        
        self.res_llm = GigaChat(
            credentials=os.environ["GIGA_TOKEN"],
            verify_ssl_certs=False,
            model="GigaChat-Pro",
            scope='GIGACHAT_API_CORP',
            temperature=0.5,
            top_p=0.5

        )


        self.engineer_llm = GigaChat(
            credentials=os.environ["GIGA_TOKEN"],
            verify_ssl_certs=False,
            model="GigaChat-Pro",
            scope='GIGACHAT_API_CORP',
            temperature=0.3,
            top_p=0.2)

         

        self.lector_llm = GigaChat(
            credentials=os.environ["GIGA_TOKEN"],
            verify_ssl_certs=False,
            model="GigaChat-Pro",
            scope='GIGACHAT_API_CORP',
            temperature=0.5,)
 




