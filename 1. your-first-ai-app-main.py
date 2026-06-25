# Your code goes here
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
# import os
# from dotenv import load_dotenv
# load_dotenv()

prompt_template_str = """
  Your task is to explain the term or acronym **{concept}** to me in a way that is:
  
  1. Clear and intuitive
  2. Concise (in under 150 words)
  3. Tailored specifically to me and what I already know
  
  Use the following information about me to personalize your explanation:
  
  - Role: Analytics Specialist transitioning to Advanced Analytics Analyst in People Analytics
  - Industry: HR / Workforce Analytics (university sector, moving to banking)
  - Background: Experienced with Python, supervised and unsupervised machine learning
  - Goal: Build LLM-powered workforce insights applications
  
  The personalization should be subtle and natural. Avoid forced references that don't genuinely enhance understanding.
  """

# Create a prompt template
prompt_template = PromptTemplate.from_template(prompt_template_str)

# Create a model interface
model = init_chat_model("gpt-4o-mini", model_provider="openai")

def generate_explanation(concept):
  
  # Format the prompt with the input
  prompt = prompt_template.format(concept=concept)
  
  # Call the model with the prompt
  response = model.invoke(prompt)

  return response.text

import gradio as gr

demo = gr.Interface(
    fn=generate_explanation,
    inputs=[gr.Textbox(label="Enter a term or acronym", lines=1)],
    outputs=[gr.Textbox(label="Explanation", lines=5)],
    flagging_mode="never",
    title="Acronym & Term Explainer",
    description="Get a personalized explanation for any term or acronym"
)

# demo.launch()


  

