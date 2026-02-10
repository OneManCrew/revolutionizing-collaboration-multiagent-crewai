import asyncio
from crewai import Agent, Task, Crew, Process
from openai import OpenAI_API

class DataAnalysisAgent(Agent):
    def __init__(self, api_key: str):
        self.api = OpenAI_API(api_key)

    async def analyze_data(self, data: str) -> dict:
        """
        Analyzes the provided data using an AI model.

        :param data: The data to analyze
        :return: Analysis results as a dictionary
        """
        try:
            response = await self.api.completion.create(prompt=data)
            return {'result': response['choices'][0]['text']}
        except Exception as e:
            return {'error': str(e)}

async def main():
    api_key = "your_openai_api_key"
    data_agent = DataAnalysisAgent(api_key)
    crew = Crew(agents=[data_agent])
    task = Task(name="Data Analysis Task", agent=data_agent, data="Analyze this sample data")
    process = Process(crew=crew, tasks=[task])

    results = await process.execute()
    for result in results:
        print(result)

# Execute the async main function
if __name__ == "__main__":
    asyncio.run(main())