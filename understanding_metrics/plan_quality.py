import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain.agents import create_agent
from deepeval.integrations.langchain import CallbackHandler
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import PlanQualityMetric

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(dotenv_path=project_root / ".env")

from agent.models import model
from evaluations.model import eval_model


def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


agent = create_agent(
    model=model,
    tools=[multiply],
    system_prompt="Be concise.",
)

# Goldens are the inputs you want to evaluate.
dataset = EvaluationDataset(goldens=[
    Golden(input="What is 8 multiplied by 6?"),
    Golden(input="What is 7 multiplied by 9?"),
])

metrics = [PlanQualityMetric(model=eval_model)]


# The `PlanQualityMetric` is passed into the `evals_iterator`.
for golden in dataset.evals_iterator(metrics=metrics):
    agent.invoke(
        {"messages": [{"role": "user", "content": golden.input}]},
        config={"callbacks": [CallbackHandler()]},
    )