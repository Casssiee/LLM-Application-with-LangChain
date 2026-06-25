from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate

# from dotenv import load_dotenv
# load_dotenv()

model = init_chat_model("gpt-4o-mini", model_provider="openai")

prompt_template_str = """
You are a helpful assistant

{input}
"""

prompt_template = PromptTemplate.from_template(prompt_template_str)


def get_response(input):
    prompt = prompt_template.format(input=input)
    response = model.invoke(prompt)
    return response

review_text = """
APP00342
Team: Advanced Analytics and Planning
Text: Honestly not sure how much point there is filling this in because last year's survey didn't lead to anything visible, but here goes.

Recognition is the big one for me. The team pulled off the migration on time and under budget and there wasn't so much as a thank you from above — meanwhile when something goes wrong we hear about it within the hour. It's demoralising. People notice that imbalance more than management thinks.

The hybrid policy change has gone down badly. We were told it was three days flexible, then suddenly it's a mandated three fixed days with no consultation, and for people with caring responsibilities or a long commute that's a real problem. A few good people have already started looking elsewhere because of it. I don't have a perfect answer but at minimum we should have been asked before it was decided.

Workload is unsustainable in our area specifically. We're the team everything funnels through but we have no ability to say no, so everyone else's deadlines become our crisis. There's no clear prioritisation framework so it's just whoever shouts loudest.

And the kitchen on level 4 has been broken for two months, which is a small thing but it's the kind of small thing that makes you feel like nobody's looking after the basics.

"""

prompt = f"Extract structured data from this survey:\n{review_text}"

# print(get_response("..."))

from pydantic import BaseModel, Field
from typing import Optional

class Area(BaseModel):
    area: str = Field(description="Area name")
    feedback: str = Field(description="Feedback section")
    problem: Optional[str] = Field(description="Problem section")
    suggestion: Optional[str] = Field(description="Suggestion section")

class Top_level(BaseModel):
  ticket: str = Field(pattern=r"^APP\d{5}$", description="Unique ticket number")
  sentiment: str = Field(description="Overall sentiment")
  areas: list[Area] = Field(description="Each area")


structured_llm = model.with_structured_output(Top_level)
result = structured_llm.invoke(prompt)  # Returns validated Pydantic object

print(result)


