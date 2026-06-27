from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm_obj1 = ChatOpenAI()

llm_obj2 = ChatOpenAI()

Prompt1 = PromptTemplate(
    template = "Write 2 lines about the topic {topic}",
    input_variables=['topic']
)

Prompt2 = PromptTemplate(
    template = "Prepare 2 quiz questions on the topic {topic}",
    input_variables=['topic']
)

Prompt3 = PromptTemplate(
    template = "Merge these two into single document \n notes -> {notes} quiz -> {quiz}",
    input_variables=['notes','quiz']
)

parser_obj = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': Prompt1 | llm_obj1 | parser_obj,
    'quiz': Prompt2 | llm_obj2 | parser_obj
})

merge_chain = Prompt3 | llm_obj1 | parser_obj

chain = parallel_chain | merge_chain

final_resp = chain.invoke({'topic':'Artifical Intelligence'})

print(f"Final merged document is : {final_resp}")

chain.get_graph().print_ascii()

