from langchain.schema import HumanMessage, SystemMessage
from langchain.chat_models.gigachat import GigaChat

import PyPDF2
import logging
from refextract import extract_references_from_file, extract_references_from_url

class Router:
    def __init__(self, token, model_list: list[dict[str, str]]) -> None:
        """              name              prompt
        model_list: [{"Researcher": "You are Researcher"}, {"Professor": "You are Professor"}]
        """
        models = {}
        for model_dict in model_list:
            for name,prompt in model_dict.items():
                pass
            model = GigaChat(credentials=token, verify_ssl_certs=False,) #model="GigaChat-Pro")
            messages = [SystemMessage(content=prompt)]
            models[name] = [model, messages]
        self.models = models
        self.references = []

    def extract_text_from_pdf(self, file_path: str) -> str:
        pdf_file_obj = open(file_path, 'rb')
        pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page_obj = pdf_reader.pages[page_num]
            text += page_obj.extract_text()
        pdf_file_obj.close()
        return text

    def parse_links(self, path: str):
        if "pdf" in path and "http" not in path:
            reference = extract_references_from_file(str(path))
            return reference
        elif "http" in path:
            reference = extract_references_from_url(path)
            return reference
        raise ValueError(f"Link is broken {path}")

    def prepare_article(self, path: str) -> None:
        self.article_text = self.extract_text_from_pdf(path)
        self.references = self.parse_links(path)['references']

        # RAG.add(article_text)

    def __call__(self, user_input: str) -> str:
        rag_agent, rag_agent_mes = self.models["RAG_agent"][0], self.models["RAG_agent"][1]
        # maybe better firstly extract facts from article_text and give only them
        rag_agent_mes.append(HumanMessage(content=self.article_text))
        modified_message =  \
f"Внимательно посмори на текст статьи в предыдущем сообщении и скажи, \
нужна ли нам дополнительная информация, или на вопрос '{user_input}' \
мы сможем ответить и без неё. \
напиши ДА если нам нужна дополнительная информация \
напиши НЕТ если нам не нужна дополнительная информация \
"
        rag_agent_mes.append(HumanMessage(content=modified_message))
        rag_agent_ans = rag_agent.invoke(rag_agent_mes).content
        # print(f"RAG: {rag_agent_ans}\n\n\n")
        extra_content = ""
        if "ДА" in rag_agent_ans:
            # extra_content = RAG(references)
            pass
        
        researcher_agent, researcher_agent_mes = self.models["Researcher"][0], self.models["Researcher"][1]
        researcher_agent_mes.append(HumanMessage(content=self.article_text))
        researcher_agent_mes.append(HumanMessage(content=user_input))
        researcher_ans = researcher_agent.invoke(researcher_agent_mes).content
        # print(f"Researcher: {researcher_ans}\n\n\n")

        researcher_reviever_ans = self.router_loop(user_input, extra_content, researcher_ans)

        professor_agent, professor_agent_mes = self.models["Professor"][0], self.models["Professor"][1]
        professor_agent_mes.append(HumanMessage(content=researcher_reviever_ans))
        professor_ans = professor_agent.invoke(professor_agent_mes).content
        # print(f"Professor: {professor_ans}")

        return professor_ans

    def router_loop(self, user_input: str, extra_content: str, researcher_ans: str) -> str:
        '''
        researcer - reviever conversation
        '''
        return researcher_ans


if __name__ == "__main__":
    article_path = "https://arxiv.org/pdf/2203.16776.pdf"
    
    chat = Router("ZGE1NDFkYjEtZWNjZC00Njc4LWFlZjUtYTNkODc2ZGRlOWQwOjg4ZGI4YmU5LWRlOTgtNDg3Ny1hNjZkLWQ4NTg0ODc5OGJlNw==",
                  [{"RAG_agent": "я рыбак"},
                   {"Researcher": "ты научный деятель, который помогает своему близкому другу разобраться с вопросом"},
                   {"Professor": "ты высоко квалифицированный специалист в области математики, искусственного интеллекта и преподавания. ты должен понятным яхыком объяснять сложные научные термины, при этом НЕ меняя полученное сообщение, а только дополняя его информацией"}])

    chat.prepare_article(article_path)

    ans = chat("привет, расскажи про LODR")
    print(ans)
