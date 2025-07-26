from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatOllama

import gradio as gr

def capital_getter(country):
    prompt = PromptTemplate.from_template("What is the capital of {topic}?")

    model = ChatOllama(model = "llama2")

    chain = ({"topic": RunnablePassthrough()} | prompt | model | StrOutputParser())

    return chain.invoke(country)

demo = gr.Interface(
    fn = capital_getter,
    inputs = ["text"],
    outputs = ["text"]
)

demo.launch(share = True)