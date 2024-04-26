from langchain.schema import HumanMessage, SystemMessage
from langchain.chat_models.gigachat import GigaChat


class Router:
    def __init__(self, model_list: list[dict[str, str]]) -> None:
        """              name              prompt
        model_list: [{"Researcher": "You are Researcher"}, {"Professor": "You are Professor"}]
        """
        models = {}
        for model_dict in model_list:
            for name,prompt in model_dict.items():
                pass
            model = GigaChat(credentials=token, verify_ssl_certs=False, scope='GIGACHAT_API_CORP', model="GigaChat-Pro", profanity_check=False)
            models[name] = [model, prompt]
        self.models = models
        self.references = []
        self.article_text = ""

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
        logging.warning(path)
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
        rag_agent, rag_agent_prompt = self.models["RAG_agent"][0], self.models["RAG_agent"][1]
        # maybe better firstly extract facts from article_text and give only them
        modified_message =  \
        f"""
        {rag_agent_prompt}
        
        Текст основной статьи:
        {self.article_text}
        
        Доступные статьи:
        1) description = LODR decoding method; url = 'https://arxiv.org/pdf/2203.16776.pdf'
        2) description = fishing methods; url = 'https://some-som.pdf'
        
        Внимательно посмори на текст основной статьи в предыдущем сообщении и описания доступных тебе статей. Если ты не можешь ответить на вопрос - 
        ты можешь обратиться к дополнительным статьям:
        ans = ДА если нам нужна дополнительная информация
        ans = НЕТ если нам не нужна дополнительная информация
        
        верни ans и если and == ДА, верни description и name параметры той статьи, в которой можно найти нужную информацию
        """
        rag_agent_mes = [SystemMessage(modified_message), HumanMessage(user_input)]
        rag_agent_ans = rag_agent.invoke(rag_agent_mes).content
        print(f"RAG: {rag_agent_ans}\n\n\n")
        extra_content = ""
        if "ДА" in rag_agent_ans:
            # extra_content = RAG(references)
            extra_content = """Utilizing text-only data with an external language model (ELM)
            in end-to-end RNN-Transducer (RNN-T) for speech recognition is challenging. Recently, a class of methods such as density
            ratio (DR) and internal language model estimation (ILME) have
            been developed, outperforming the classic shallow fusion (SF)
            method. The basic idea behind these methods is that RNN-T
            posterior should first subtract the implicitly learned internal language model (ILM) prior, in order to integrate the ELM. While
            recent studies suggest that RNN-T only learns some low-order
            language model information, the DR method uses a well-trained
            neural language model with full context, which may be inappropriate for the estimation of ILM and deteriorate the integration performance. Based on the DR method, we propose a loworder density ratio method (LODR) by replacing the estimation
            with a low-order weak language model. Extensive empirical experiments are conducted on both in-domain and cross-domain
            scenarios on English LibriSpeech & Tedlium-2 and Chinese
            WenetSpeech & AISHELL-1 datasets. It is shown that LODR
            consistently outperforms SF in all tasks, while performing generally close to ILME and better than DR in most tests.
            Index Terms: ASR, language model, transducer"""
            pass
        
        researcher_agent, researcher_agent_prompt = self.models["Researcher"][0], self.models["Researcher"][1]
        modified_message = \
        f"""{researcher_agent_prompt}

        Текст основной статьи:
        {self.article_text}

        Дополнительная информация:
        {extra_content}
        """
        researcher_agent_mes = [SystemMessage(modified_message), HumanMessage(content=user_input)]
        researcher_ans = researcher_agent.invoke(researcher_agent_mes).content
        print(f"Researcher: {researcher_ans}\n\n\n")

        researcher_reviever_ans = self.router_loop(user_input, extra_content, researcher_ans)

        professor_agent, professor_agent_prompt = self.models["Professor"][0], self.models["Professor"][1]
        modified_message = \
        f"""{professor_agent_prompt}

        Текст основной статьи:
        {self.article_text}
        """
        professor_agent_mes = [SystemMessage(modified_message), HumanMessage(content=researcher_reviever_ans)]
        professor_ans = professor_agent.invoke(professor_agent_mes).content
        print(f"Professor: {professor_ans}")

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