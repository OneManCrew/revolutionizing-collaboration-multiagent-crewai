import asyncio
from typing import List, Dict, Any
from crewai import Agent, Task, Crew
from transformers import pipeline, set_seed

class TransformerAgent(Agent):
    """
    A Transformer-based agent leveraging pre-trained models for task execution.
    """
    def __init__(self, name: str, model_name: str):
        super().__init__(name)
        self.generator = pipeline('text-generation', model=model_name)
        set_seed(42)

    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """
        Execute a given task using transformer-based text generation.

        :param task: The task to be executed.
        :return: The result of the task execution.
        """
        try:
            result = self.generator(task.input_data, max_length=50, num_return_sequences=1)
            return {"status": "success", "output": result[0]['generated_text']}
        except Exception as e:
            return {"status": "error", "message": str(e)}

async def main():
    """
    Initialize a crew of TransformerAgents and execute tasks collaboratively.
    """
    agents = [TransformerAgent(name=f"Agent-{i}", model_name="gpt2") for i in range(3)]
    crew = Crew(agents=agents)

    tasks = [Task(input_data="Optimize collaboration for AI systems."),
             Task(input_data="Generate integration strategies for multi-agent frameworks."),
             Task(input_data="Explore decentralized network architectures.")]

    results = await crew.execute_tasks(tasks)
    for result in results:
        print(result)

if __name__ == '__main__':
    asyncio.run(main())