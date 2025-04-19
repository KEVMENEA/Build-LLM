import streamlit as st
import asyncio
import os
from together import AsyncTogether, Together

#  Set up the Streamlit app and API key input.
st.title("Mixture-of-Agents LLM App")
together_api_key = st.text_input("Enter your Together API key:", type="password")

# Initialize Together AI clients.
if together_api_key:
    os.environ["TOGETHER_API_KEY"] = together_api_key
    client = Together(api_key=together_api_key)
    async_client = AsyncTogether(api_key=together_api_key)
    
# Define the models and aggregator system prompt.
reference_models = [
    "Qwen/Qwen2-72B-Instruct",
    "Qwen/Qwen2-72B-Chat",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "databricks/dbrx-instruct",
]

aggregator_model = "mistralai/Mixtral-8x22B-Instruct-v0.1"

aggregator_system_prompt = """
You have been provided with a set of responses from various open-source models to the latest user query.
Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate 
the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response 
should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the 
instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and 
reliability. 

Responses from models:
"""

# Implement the LLM call function.
async def run_llm(model):
    response = await async_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=0.7,
        max_tokens=512,
    )
    return model, response.choices[0].message.content

#  Define the main function to run all LLMs and aggregate results.
async def main():
    results = await asyncio.gather(*[run_llm(model) for model in reference_models])
    
    # Display individual model responses
    st.subheader("Individual Model Responese:")
    for model, response in results:
        with st.expander(f"Response from {model}") :
            st.write(response)
    
    # Aggregate responses
    st.subheader("Aggregated Response:")
    finalStream = client.chat.completions.create(
        model=aggregator_model,
        messages=[
            {"role": "system", "content": aggregator_system_prompt},
            {"role": "user", "content": ",".join(response for _, response in results)},
            
        ],
        stream=True
    )
    # Display aggregated response
    response_container = st.empty()
    full_response = ""
    for chunk in finalStream:
        content = chunk.choices[0].delta.content or ""
        full_response += content
        response_container.markdown(full_response + "")
    response_container.markdown(full_response)
    
    # Set up the user interface and trigger the main function.
    user_prompt = st.text_input("Enter your question:")
    
    if st.button("Get Answer"):
        if user_prompt:
            asyncio.run(main())
        else:
            st.warning("Please enter a question.")
