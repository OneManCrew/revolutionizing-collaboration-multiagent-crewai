import asyncio
from crewai import Agent, Task, Crew
from transformers import pipeline
from typing import List, Dict

class TransformerAgent(Agent):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = pipeline('text-generation', model=model_name)

    async def execute_task(self, task: Task) -> Dict[str, str]:
        """
        Executes a text generation task using a transformer model.

        Args:
            task (Task): The task containing input text for generation.

        Returns:
            Dict[str, str]: Generated text based on the input.
        """
        input_text = task.data.get('input_text', '')
        if not input_text:
            raise ValueError("Input text is required for text generation.")

        try:
            result = self.model(input_text, max_length=100, num_return_sequences=1)
            return {'generated_text': result[0]['generated_text']}
        except Exception as e:
            raise RuntimeError(f"Error in text generation: {e}")

async def main():
    agent = TransformerAgent(model_name='gpt-2')
    crew = Crew(agents=[agent])

    async def orchestrate_tasks() -> List[Dict[str, str]]:
        """
        Orchestrates multiple text generation tasks using CrewAI's multi-agent system.

        Returns:
            List[Dict[str, str]]: A list of generated text results from each task.
        """
        tasks = [
            Task(agent_id=agent.id, data={'input_text': 'The future of AI is'}),
            Task(agent_id=agent.id, data={'input_text': 'Collaborative systems can'}),
            Task(agent_id=agent.id, data={'input_text': 'Transformers enable'}),
        ]
        results = await crew.process_tasks(tasks)
        return results

    results = await orchestrate_tasks()
    for result in results:
        print(result['generated_text'])

if __name__ == "__main__":
    asyncio.run(main())