import operator
from code_writer import CodeWrite
from prompts import get_research_prompt, lector_prompt
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




class WorkFlow():
    def __init__(self):
        self.coder = CodeWrite("code_prompt.txt") 
        self.base_db = FAISS.load_local("fais_db/", 
                                        embeddings=FastEmbeddings(),
                                        allow_dangerous_deserialization=True)
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
        response = self.call_rag_paper(query = outputs.content[0].split('"')[-1],
                                       db=self.curr_db)
        return response
   
    def call_rag_paper(self, query, db):
        paper_info = paper_search(query, 
                                  db=db)
        
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

    def rewrite(self, input):
       pass 


    def call_researcher(self, human_input, rag_response):
        model = GigaChat(
                    credentials=os.environ['GIGA_TOKEN'],
                    verify_ssl_certs=False,
                    model="GigaChat-Pro",
                    scope='GIGACHAT_API_CORP',
                    temperature=0.6,
                    top_p=0.4)
        
        input_prompt = {"name":st.session_state.name, 
                        "context":rag_response,
                        "user_query":human_input,
                        "query1":" ",
                        "query2":" ",
                        "code1": " "} 
        res_chain = get_research_prompt() | model | StrOutputParser()
        response = res_chain.invoke(input_prompt) 
        for rec in range(3):
            if "END" in response:
                break
            if "Запрос1" in response:
                output = self.call_rag_paper(response.split("Запрос1:")[-1],
                                             db=self.curr_db)
                input_prompt['query1'] = output
            if "Код1" in response:
                output = self.call_engineer(response.split("Код1:")[-1])
                input_prompt['code1'] = output
            if "Запрос2" in response:
                output = self.call_rag_paper(response.split("Запрос2:")[-1],
                                             db=self.base_db)
                input_prompt['query2'] = output


            logging.warning(input_prompt)        
            response = res_chain.invoke(input_prompt)

        return response, get_research_prompt().invoke(input_prompt)

    def call_engineer(self, query):
        code = self.coder.code_generator(query)
        logging.warning(code)
        return code

    def execute_steps(self, human_input:str):
        base_paper_data = self.entry(human_input=human_input)
        response, prompt = self.call_researcher(human_input=human_input,
                                             rag_response=base_paper_data)
        lector_output = self.call_lector(input=human_input,
                                         response=prompt)
        
        logging.warning(f"LECTOR OUTPUT {lector_output}")
        return lector_output
    def call_lector(self, input, response):
        model = GigaChat(credentials=os.environ['GIGA_TOKEN'],
                        verify_ssl_certs=False,
                        model="GigaChat-Pro",
                        scope='GIGACHAT_API_CORP',
                        temperature=0.3,
                        top_p=0.5)
        prompt = lector_prompt()
        inputs = {"user_input":input,
                  "query":response}
        lector_chain = prompt | model | StrOutputParser()
        lector_perm = lector_chain.invoke(inputs)
        return lector_perm

    def build_response(self, response, result):
        if result:
            response = ""
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
 




