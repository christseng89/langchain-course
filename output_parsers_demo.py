from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
  ChatPromptTemplate,
)

load_dotenv()


def print_description(text):
  print(f"\n\033[94m{'***'} {text} {'***'}\033[0m\n")


parser = StrOutputParser()
prompt = ChatPromptTemplate.from_template("wire a short poem about {topic} within 30 words")
model = init_chat_model(model="gpt-4o-mini", temperature=0)
chain = prompt | model | parser

response = chain.invoke({"topic": "nature"})
print_description("StrOutputParser result type")
print(
  f"Type: {type(response)}\n\nResponse: {response}"
)  # <class 'str'> A short poem about nature...


# JsonOutputParser example
from langchain_core.output_parsers import JsonOutputParser

prompt = ChatPromptTemplate.from_template(
  "Return a JSON object with 'name','gender' and 'age' for: {description}"
)
parser = JsonOutputParser()
chain = prompt | model | parser

result = chain.invoke({"description": "A 25-year-old male developer named Alex"})
print_description("JsonOutputParser result")
print(result)  # {'name': 'Alex', 'age': 25}

# PydanticOutputParser example
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class Person(BaseModel):
  name: str = Field(description="The person's name")
  age: int = Field(description="The person's age")
  occupation: str = Field(description="The person's occupation")


prompt = ChatPromptTemplate.from_template(
  "Return a JSON object with 'name', 'age', and 'occupation' for: {description}"
).partial(format_instructions=parser.get_format_instructions())
parser = PydanticOutputParser(pydantic_object=Person)
chain = prompt | model | parser

result = chain.invoke({"description": "A 30-year-old artist named Maria"})
print_description("PydanticOutputParser result")
print(result)  # Person(name='Maria', age=30, occupation='artist')


# Structured Output via with_structured_output
class MovieReview(BaseModel):
  title: str = Field(description="The title of the movie")
  review: str = Field(description="A brief review of the movie within 50 words")
  rating: int = Field(description="The rating of the movie out of 10")


# Bind the schema to the model
structured_model = model.with_structured_output(MovieReview)

result = structured_model.invoke("Review: Inception is a mind-bending thriller. 9/10")
print_description("Structured Output (with_structured_output) result")
print(f"Title: {result.title}\n\nReview: {result.review}\n\nRating: {result.rating}")


# Same result using chain = prompt | model | parser
parser = PydanticOutputParser(pydantic_object=MovieReview)
prompt = ChatPromptTemplate.from_template(
  "Review this movie and return a JSON object with 'title', 'review within 30 words', and 'rating' out of 10.\n"
  "Movie: {input}"
).partial(format_instructions=parser.get_format_instructions())

parser = PydanticOutputParser(pydantic_object=MovieReview)
chain = prompt | model | parser

result = chain.invoke({"input": "Review Inception"})
print_description("Structured Output (chain) result")
print(f"Title: {result.title}\n\nReview: {result.review}\n\nRating: {result.rating}")
