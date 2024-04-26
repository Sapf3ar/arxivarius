from langchain_community.llms.gigachat import GigaChat
from langchain_core.utils.function_calling import convert_to_gigachat_function, convert_to_gigachat_tool
from typing_extensions import Annotated
from langchain_community.document_loaders import ArxivLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
import streamlit as st
import logging
from embedder import check_preload, download_papers, embed_docs, show_outline
from parser import get_correct_name_from_topk, get_download_id, parse_links
#from selfrag import SelfRagExecutor, paper_search, ref_search
import pprint
import os

from selfrag import WorkFlow

os.environ["GIGA_TOKEN"] = "Yjg4MTQzMmUtNDAwMS00NDk0LThjOGUtNmU5ZWQ2YzQ4NDQ2OmQ4MWMxZGZiLTFmNGYtNDk5NS05OGQzLTBiMzYyYWJmNjk3OA=="
check_preload()

def init_keys():
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
    
    if "file_uploaded" not in st.session_state:
        st.session_state["file_uploaded"] = False

init_keys()


model = WorkFlow()

def make_file_upload():
    st.session_state.file_uploaded = True
logging.warning(st.session_state.file_uploaded)
with st.sidebar:
    if st.session_state.file_uploaded:
        show_outline()
        st.markdown("Processing references")
    paper_link = st.text_input("Link from arxiv",
                               on_change=make_file_upload
                               )
    logging.warning("link" + paper_link)
    if paper_link:
        logging.warning("link" + paper_link + "uploaded")
        try:
            logging.warning("uploading")
            paper =  ArxivLoader(query=paper_link.split("/")[-1][:-3], 
                                 load_all_available_meta=False,
                                 load_max_docs=1).load()[0]
            
        except Exception as e:
            st.markdown("Invalid link or arxiv api is unreachable :(")
            st.session_state.file_uploaded = False
            logging.warning("EXCEPTION" + str(e))
        else:
            
            st.session_state.sum = paper.metadata['Summary']
            st.session_state.name = paper.metadata['Title']
            st.session_state.refs = parse_links(paper_link) 
            model.add_current_paper_rag(embed_docs([paper]))
                                        #agent_rag = SelfRagExecutor(references=st.session_state.refs, tools=[ref_search, paper_search]).build_graph())A
            #st.session_state.self_rag = agent_rag 
            '''
            my_bar = st.progress(0, text="Fetching references")
            paper_ids = []
            start = 0
            references = references[:5]
            incr = 1/len(references)
            for ref_id in range(len(references)):
                ref_q = references[ref_id]
                id_ = get_download_id(ref_q)
                if id_ is not None:
                    paper_ids.append(id_)
                    start += incr
                my_bar.progress(start, "Fetching references" )
            start =1.0
            my_bar.progress(start, "Fetching references" )

            papers = download_papers(paper_ids)
            embed_docs(papers)
            '''




with st.container():
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    if prompt := st.chat_input():

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        outputs= model.execute_steps(st.session_state.messages[-1]['content'])
        pprint.pprint(outputs)


        msg = outputs
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)


