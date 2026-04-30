# from __future__ import annotations

# from collections.abc import AsyncIterator

# from fastapi.testclient import TestClient
# from langchain_core.messages import AIMessage
# from pytest import MonkeyPatch

# from app.agent.graph import AgentConfigurationError
# from medical_voice_agent.backend.app.api.routes.chat import chat_routes as chat_routes
# from app.main import app
# from tests.workflow_support import install_test_graph, latest_human_text


# def _post_invoke(client: TestClient, *, message: str, thread_id: str):
#     return client.post(
#         "/api/v1/chat/invoke",
#         json={"message": message, "thread_id": thread_id},
#     )


# def test_chat_invoke_supports_multi_turn_memory_over_http(
#     monkeypatch: MonkeyPatch,
# ) -> None:
#     async def fake_intake_node(state: AgentState) -> dict:
#         latest = latest_human_text(state).lower()
#         patient_status = state.get("patient_status")

#         if patient_status == "new":
#             if "555-0100" in latest or "10/26/1985" in latest:
#                 return {
#                     "messages": [
#                         AIMessage(content="Thanks, you're registered as Sarah Connor.")
#                     ],
#                     "patient_id": "patient-1",
#                     "patient_name": "Sarah Connor",
#                     "last_agent": "intake",
#                 }

#             return {
#                 "messages": [
#                     AIMessage(
#                         content="What is your full name, date of birth, and phone number?"
#                     )
#                 ],
#                 "last_agent": "intake",
#             }

#         return {
#             "messages": [AIMessage(content="Let's verify your record.")],
#             "last_agent": "intake",
#         }

#     async def fake_triage_node(state: AgentState) -> dict:
#         latest = latest_human_text(state).lower()
#         if "headache" in latest:
#             return {
#                 "messages": [
#                     AIMessage(content="Neurology seems like the right specialty.")
#                 ],
#                 "specialty_id": "spec-neuro",
#                 "last_agent": "triage",
#             }
#         return {
#             "messages": [AIMessage(content="What symptoms are you having?")],
#             "last_agent": "triage",
#         }

#     async def fake_scheduling_node(state: AgentState) -> dict:
#         return {
#             "messages": [
#                 AIMessage(
#                     content=(
#                         "Do you have a preferred day or week in mind, or would you like the earliest available?"
#                     )
#                 )
#             ],
#             "last_agent": "scheduling",
#         }

#     install_test_graph(
#         monkeypatch,
#         intake_node=fake_intake_node,
#         triage_node=fake_triage_node,
#         scheduling_node=fake_scheduling_node,
#     )

#     with TestClient(app) as client:
#         thread_id = "api-booking-flow"

#         greeting = _post_invoke(client, message="hi", thread_id=thread_id)
#         status_question = _post_invoke(
#             client,
#             message="I need to book an appointment",
#             thread_id=thread_id,
#         )
#         registration_prompt = _post_invoke(
#             client,
#             message="This is my first visit",
#             thread_id=thread_id,
#         )
#         post_registration = _post_invoke(
#             client,
#             message="Sarah Connor, 10/26/1985, 555-0100",
#             thread_id=thread_id,
#         )

#     assert greeting.status_code == 200
#     assert greeting.json() == {
#         "response": "Hello! Welcome to the clinic. How can I help you today?",
#         "thread_id": thread_id,
#     }

#     assert status_question.status_code == 200
#     assert status_question.json()["response"] == (
#         "Have you been seen at this clinic before, or is this your first visit?"
#     )

#     assert registration_prompt.status_code == 200
#     assert registration_prompt.json()["response"] == (
#         "What is your full name, date of birth, and phone number?"
#     )

#     assert post_registration.status_code == 200
#     assert (
#         "Thanks, you're registered as Sarah Connor."
#         in post_registration.json()["response"]
#     )
#     assert "What symptoms are you having?" in post_registration.json()["response"]


# def test_chat_invoke_returns_503_when_agent_is_not_configured(
#     monkeypatch: MonkeyPatch,
# ) -> None:
#     async def fake_invoke_agent(*, message: str, thread_id: str) -> str:
#         raise AgentConfigurationError("agent is unavailable")

#     monkeypatch.setattr(chat_routes, "invoke_agent", fake_invoke_agent)

#     with TestClient(app) as client:
#         response = _post_invoke(
#             client,
#             message="I need to book an appointment",
#             thread_id="thread-503",
#         )

#     assert response.status_code == 503
#     assert response.json() == {"detail": "agent is unavailable"}


# def test_chat_invoke_validates_request_body() -> None:
#     with TestClient(app) as client:
#         response = client.post(
#             "/api/v1/chat/invoke",
#             json={"message": "", "thread_id": "thread-422"},
#         )

#     assert response.status_code == 422


# def test_chat_stream_returns_sse_chunks_and_done_marker(
#     monkeypatch: MonkeyPatch,
# ) -> None:
#     async def fake_ensure_agent_ready() -> None:
#         return None

#     async def fake_stream_agent_response(
#         message: str,
#         thread_id: str,
#     ) -> AsyncIterator[str]:
#         assert message == "hello"
#         assert thread_id == "thread-stream"
#         yield "Hello"
#         yield " there"

#     monkeypatch.setattr(chat_routes, "ensure_agent_ready", fake_ensure_agent_ready)
#     monkeypatch.setattr(
#         chat_routes, "stream_agent_response", fake_stream_agent_response
#     )

#     with TestClient(app) as client:
#         with client.stream(
#             "POST",
#             "/api/v1/chat",
#             json={"message": "hello", "thread_id": "thread-stream"},
#         ) as response:
#             body = "".join(response.iter_text())

#     assert response.status_code == 200
#     assert response.headers["content-type"].startswith("text/event-stream")
#     assert "data: Hello\n\n" in body
#     assert "data:  there\n\n" in body
#     assert "data: [DONE]\n\n" in body


# def test_chat_stream_returns_503_when_agent_setup_fails(
#     monkeypatch: MonkeyPatch,
# ) -> None:
#     async def fake_ensure_agent_ready() -> None:
#         raise AgentConfigurationError("missing backend config")

#     monkeypatch.setattr(chat_routes, "ensure_agent_ready", fake_ensure_agent_ready)

#     with TestClient(app) as client:
#         response = client.post(
#             "/api/v1/chat",
#             json={"message": "hello", "thread_id": "thread-stream-503"},
#         )

#     assert response.status_code == 503
#     assert response.json() == {"detail": "missing backend config"}
