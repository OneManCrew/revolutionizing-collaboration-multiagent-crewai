import asyncio
from crewai import MultiAgentSystem, Agent, Task, Crew
from crewai.exceptions import CoordinationError
from typing import List, Dict

class CrewAIAgent(Agent):
    def __init__(self, name: str, capabilities: List[str]):
        super().__init__(name)
        self.capabilities = capabilities

    async def perform_task(self, task: Task) -> str:
        """
        Perform the assigned task asynchronously and return the result.
        """
        if task.type not in self.capabilities:
            raise CoordinationError(f"Agent {self.name} cannot perform task: {task.type}")
        # Simulate task processing
        await asyncio.sleep(1)
        return f"Task {task.type} completed by {self.name}"

class CrewAICoordinator:
    def __init__(self, crew: Crew):
        self.crew = crew

    async def coordinate_tasks(self, tasks: List[Task]) -> Dict[str, str]:
        """
        Coordinate task execution among agents and handle exceptions gracefully.
        """
        results = {}
        for task in tasks:
            available_agents = [agent for agent in self.crew.agents if task.type in agent.capabilities]
            if not available_agents:
                results[task.id] = "No capable agent available"
                continue
            agent = available_agents[0]  # Assign the first capable agent
            try:
                results[task.id] = await agent.perform_task(task)
            except CoordinationError as e:
                results[task.id] = str(e)
        return results

async def main():
    tasks = [Task(id="task1", type="inspection"), Task(id="task2", type="repair")]
    agents = [CrewAIAgent(name="Agent1", capabilities=["inspection"]), CrewAIAgent(name="Agent2", capabilities=["repair"])]
    crew = Crew(agents=agents)
    coordinator = CrewAICoordinator(crew)
    results = await coordinator.coordinate_tasks(tasks)
    print(results)

# Execute the asynchronous main function
asyncio.run(main())