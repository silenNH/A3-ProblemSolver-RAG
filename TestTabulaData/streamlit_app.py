import os
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
import streamlit as st 
from data import load_data
from dotenv import load_dotenv
from pandasai.responses.response_parser import ResponseParser
from pandasai.callbacks import BaseCallback

class StreamlitCallback(BaseCallback):
    def __init__(self, container) -> None:
        """Initialize callback handler."""
        self.container = container

    def on_code(self, response: str):
        self.container.code(response)


class StreamlitResponse(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)

    def format_dataframe(self, result):
        st.dataframe(result["value"])
        return

    def format_plot(self, result):
        st.image(result["value"])
        return

    def format_other(self, result):
        st.write(result["value"])
        return


st.write("Lets thest the Credit Card Froud Dataset 🧐")

df=load_data("./data")

with st.expander("Dataframe Preview"):
    st.write(df.tail(3))

query=st.text_area("Chat with the Data Frame 🗣️")
container =st.container()

load_dotenv(override=True)
if query:
    llm=OpenAI(api_token=os.environ["OPENAI_API_KEY"])
    query_engine = SmartDataframe(df, 
    config={
        "llm":llm,
        "response_parser":StreamlitResponse,
        "callback": StreamlitCallback(container),
    })
    answer=query_engine.chat(query)
    #st.write(answer)
#st.write(df.tail(100))