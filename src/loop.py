import logging

from .router import AgentRouter, BelowThresholdError
from .session import Session
from tools import summary_tool

from pydantic_ai.messages import ToolReturnPart 


def chat_loop() -> None:
	"""Run a simple terminal chat loop against the routed agent."""
	router = AgentRouter()
	active_session: Session | None = None

	while True:
		try:
			query = input("You: ").strip()
		except KeyboardInterrupt:
			print("\nInterrupted. Type '/route' to reroute or 'exit' to quit.")
			continue
		except EOFError:
			print("\nGoodbye.")
			break

		if not query:
			continue

		if query.lower() in {"exit", "quit"}:
			return_message = summary_tool.handle(active_session._history)
			print("Agent: " + return_message)
			print("Goodbye.")
			break

		if query == "/route":
			active_session = None
			print("Routing reset. Your next message will select an agent.")
			continue

		if active_session is None:
			try:
				agent, metadata = router.route_with_metadata(query)
			except BelowThresholdError:
				print("Router: I'm not sure which agent fits - try rephrasing, or type /route to reset.")
				continue
			except Exception as exc:
				print(f"Router: {exc}")
				continue
			active_session = Session(agent)
			print(f"Routed to agent: {agent.name} (score={metadata['score']:.3f})")




		print("Agent: ", end="", flush=True)

		for chunk in active_session.send_stream(query):
			print(chunk, end="", flush=True)

		
		for message in reversed(active_session._history):
			for part in message.parts:
				if isinstance(part, ToolReturnPart) and part.tool_name == "death_certificate_verification":
					if part.content.get("band") in {"high", "medium"}:
						return_message = summary_tool.handle(active_session._history)
						print("Agent: " + return_message)
						print("Goodbye.")
						break
		print()


def main() -> None:
	logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
	chat_loop()
