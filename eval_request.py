import os
from promptflow.core import AzureOpenAIModelConfiguration
from typing import Any, Dict, List
from dotenv import load_dotenv
from dotenv import dotenv_values
#load_dotenv()
import uuid

from promptflow.core import Prompty, AzureOpenAIModelConfiguration
from promptflow.evals.evaluators import ChatEvaluator
from promptflow.evals.evaluators import RelevanceEvaluator, CoherenceEvaluator
from promptflow.evals.evaluate import evaluate
from promptflow.evals.evaluators import ViolenceEvaluator
from promptflow.evals.evaluators import RelevanceEvaluator
from promptflow.evals.synthetic import AdversarialSimulator
from promptflow.evals.synthetic import AdversarialScenario
from azure.identity import DefaultAzureCredential

from promptflow.tracing import start_trace
start_trace()

from dotenv import dotenv_values

config = dotenv_values("env.env")

#config = load_dotenv("env.env")
azure_deployment = config["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
api_key = config["AZURE_OPENAI_API_KEY"]
api_version = config["AZURE_OPENAI_API_VERSION"]
api_base = config["AZURE_OPENAI_ENDPOINT"]
embeddingmodel = config["AZURE_OPENAI_EMBEDDING_MODEL"]
azuresearchendpoint = config["AZURE_SEARCH_ENDPOINT"]
azure_chatdeployment = config["AZURE_OPENAI_CHAT_DEPLOYMENT"]
azure_endpoint=config["AZURE_OPENAI_ENDPOINT"]

# Initialize Azure OpenAI Connection with your environment variables
model_config = AzureOpenAIModelConfiguration(
    azure_endpoint=azure_endpoint,
    api_key=api_key,
    azure_deployment=azure_chatdeployment,
    api_version=api_version,
)

from promptflow.evals.evaluators import RelevanceEvaluator
from promptflow.tracing import start_trace
start_trace()

azure_ai_project={
        "subscription_id": "80ef7369-572a-4abd-b09a-033367f44858",
        "resource_group_name": "rg-babalai",
        "project_name": "babal-sweden",
    }

def chateval():
    results = []

    model_config = AzureOpenAIModelConfiguration(
        azure_deployment=config["AZURE_OPENAI_CHAT_DEPLOYMENT"],
        api_version=config["AZURE_OPENAI_API_VERSION"],
        azure_endpoint=config["AZURE_OPENAI_ENDPOINT"],
        api_key=config["AZURE_OPENAI_API_KEY"]
    )
    chat_eval = ChatEvaluator(model_config=model_config)

    chat_history=[
        {"role": "user", "content": "Does Azure OpenAI support customer managed keys?"},
        {"role": "assistant", "content": "Yes, customer managed keys are supported by Azure OpenAI."}
    ]
    chat_input="Do other Azure AI services support this too?"

    prompty = Prompty.load("chat2.prompty", model={'configuration': model_config})
    response = prompty(chat_history=chat_history, chat_input=chat_input)

    print(response)

    relevance_evaluator = RelevanceEvaluator(model_config=model_config)
    coherence_evaluator = CoherenceEvaluator(model_config=model_config)
    path = "profiles.jsonl"
    result = evaluate(
        data=path,
        evaluators={
            "coherence": coherence_evaluator,
            "relevance": relevance_evaluator,
        },
        evaluator_config={
            "coherence": {
                "answer": "${data.answer}",
                "question": "${data.question}"
            },
            "relevance": {
                "answer": "${data.answer}",
                "context": "${data.context}",
                "question": "${data.question}"
            }
        },
        azure_ai_project=azure_ai_project,
        #output_path="./evalresults/evaluation_results.json",
        evaluation_name="chat_eval_gpt4og_" + str(uuid.uuid4())[:8],
    )
    # print(result)
    results.append("{ 'model' : 'gpt-4o-g', 'metrics' : " + str(result['metrics']) + "," + "'studio_url' : '" + str(result['studio_url']) + "' }")
    print(results)


def eval():
    # Initialzing Relevance Evaluator
    relevance_eval = RelevanceEvaluator(model_config)
    # Running Relevance Evaluator on single input row
    relevance_score = relevance_eval(
        answer="The Alpine Explorer Tent is the most waterproof.",
        context="From the our product list,"
        " the alpine explorer tent is the most waterproof."
        " The Adventure Dining Table has higher weight.",
        question="Which tent is the most waterproof?",
    )
    print(relevance_score)

if __name__ == "__main__":
    #eval()
    rs = chateval()