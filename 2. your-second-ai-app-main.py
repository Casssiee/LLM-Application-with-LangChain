from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate

# Model: gpt-4o-mini | Provider: openai
prompt_template_str = """
  Your task is to explain the intuition behind HR/people analytics metrics **{metrics}** to me in a way that is. And also explains what is the intuition behind

  1. Clear and intuitive
  2. Consice (in under 150 words)
  3. Tailored specifically to me and using examples whenever is necessary.

  Use the following information about me to personalised your explanation:

  - Role: Senior manager in People Analystics
  - Industry: HR
  - Background: Little experience with coding knowledge but have 20 years of industry experience
  - Goal: Build LLM-powered workforce metrics explainer to assist senior manager to be able to present the data to executives

  The personalisation should be subtle and natural. Aviod forced references that don't geuniely enhance understanding. Also not using any acronmys.
  """

prompt_template = PromptTemplate.from_template(prompt_template_str)

model = init_chat_model("gpt-4o-mini", model_provider="openai")

def generate_explanation(metrics):
  prompt = prompt_template.format(metrics=metrics)
  response = model.invoke(prompt)
  return response.text

import gradio as gr

demo = gr.Interface(
    fn=generate_explanation,
    inputs=[gr.Textbox(label="Enter a metrics", lines=1)],
    outputs=[gr.Textbox(label="Explanation", lines=5)],
    flagging_mode="never",
    title="HR Metrics Explainer",
    description="Get a personalized explanation for any metrics"
)
