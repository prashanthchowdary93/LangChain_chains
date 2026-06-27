from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch,RunnableLambda
from langchain_core.output_parsers import StrOutputParser , PydanticOutputParser
from typing import Literal
from pydantic import Field, BaseModel

load_dotenv()

llm_obj1 = ChatOpenAI()

class Feedback(BaseModel):
    sentiment:Literal['pos','neg'] = Field(description="Classify the sentiment of given feedback") 

parser_pyd_obj = PydanticOutputParser(pydantic_object=Feedback)

prompt = PromptTemplate(
    template="Classify the sentiment analysis of this feedback \n {feedback} \n {format_instruction}",
    input_variables=["feedback"],
    partial_variables={"format_instruction":parser_pyd_obj.get_format_instructions()}
)

print(f"Prompt is : {prompt}")

prompt1 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback1}',
    input_variables=['feedback1']
)

prompt2 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback2}',
    input_variables=['feedback2']
)

parser = StrOutputParser()

cond_chain = RunnableBranch(
    (lambda x:x.sentiment == 'pos' , prompt1 | llm_obj1 | parser),
    (lambda x:x.sentiment == 'neg' , prompt2 | llm_obj1 | parser),
     RunnableLambda(lambda x:  "sentiment  is not found ")
)

classifier_chain = prompt | llm_obj1 | parser_pyd_obj

chain = classifier_chain | cond_chain

resp = chain.invoke({"feedback":"This is a very good performance phone,  battery drains very slow as well"})

print(f"Final resp is : {resp}")

chain.get_graph().print_ascii()