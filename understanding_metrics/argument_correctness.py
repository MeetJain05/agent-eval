
from langchain.agents import create_agent
from deepeval import evaluate
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import ArgumentCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(dotenv_path=project_root / ".env")

from agent.models import model
from evaluations.model import eval_model

# Step 1: Define the tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


# Step 2: Create LangChain agent
agent = create_agent(
    model=model,
    tools=[multiply],
    system_prompt="Be concise. Use tools for any math questions.",
)


# Step 3: Define goldens
dataset = EvaluationDataset(
    goldens=[
        Golden(
            input="What is 8123 multiplied by 62323?"
        ),
        Golden(
            input="What is 7 multiplied by 92323?"
        ),
    ]
)


# Step 4: Execute agent and create test cases
for golden in dataset.goldens:

    print(f"\nEvaluating: {golden.input}")

    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": golden.input}
            ]
        }
    )

    actual_output = result["messages"][-1].content

    # Extract actual tool calls
    tools_called = []

    for message in result["messages"]:
        for tool_call in getattr(message, "tool_calls", []) or []:
            tools_called.append(
                ToolCall(
                    name=tool_call["name"],
                    input_parameters=tool_call["args"],
                )
            )

    # Create LLMTestCase
    test_case = LLMTestCase(
        input=golden.input,
        actual_output=actual_output,
        tools_called=tools_called,
    )

    dataset.add_test_case(test_case)

    print("Agent Response:", actual_output)
    print("Tools Called:", tools_called)


# Step 5: Configure Argument Correctness metric
metric = ArgumentCorrectnessMetric(
    threshold=1.0,
    model=eval_model,
    include_reason=True,
    verbose_mode=True,
)


# Step 6: Run evaluation
evaluate(
    test_cases=dataset.test_cases,
    metrics=[metric],
)
