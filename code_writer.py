from langchain.schema import HumanMessage, SystemMessage
from langchain.chat_models.gigachat import GigaChat
import os


class CodeWrite:
    def __init__(self, prompt_path: str):
        self.llm = GigaChat(credentials=os.environ['GIGA_TOKEN'],
                            verify_ssl_certs=False,
                            scope='GIGACHAT_API_CORP',
                            model="GigaChat-Pro-preview",
                            profanity_check=False,
                            temperature=0.2,
                            top_p=0.1)

        self.prompt = open(prompt_path, "r").read()

        self.messages = [
                SystemMessage(
                    content=self.prompt
                )
            ]

    def reload_state(self) -> None:
        self.messages = [
                SystemMessage(
                    content=self.prompt
                )
            ]

    def code_generator(self, query: str) -> str:
        self.messages.append(HumanMessage(content=query))
        answer = self.llm.invoke(self.messages)
        self.reload_state()
        return answer.content
