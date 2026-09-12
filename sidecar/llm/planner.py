"""
LangGraph Planner
Phase 5.4: Orchestrates the full conversation pipeline

StateGraph: build_prompt → call_llm → split_sentences → route_to_tts

This is the central orchestrator that:
1. Assembles the full prompt (system + memories + screen + history)
2. Streams tokens from Ollama
3. Splits into sentence clauses
4. Routes each clause to the TTS pipeline
"""

import logging
import os
import asyncio
from datetime import datetime
from typing import Any, Optional, Callable, Awaitable

from .ollama_client import OllamaClient, OllamaError
from .sentence_splitter import SentenceSplitter
from .sampling_config import get_sampling_params
from . import intent_router
import tools
from agent_permissions import permissions

logger = logging.getLogger("sylph.llm.planner")

# Safety bound on the tool-calling agent loop. Without it, a model that keeps
# emitting tool calls (or ping-pongs between two tools) loops forever — a hard
# hang in a live voice UI. Six rounds is plenty for any real multi-tool answer.
# Phase A.
MAX_TOOL_ITERATIONS = 6

# Load system prompt template
_PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")
_SYSTEM_PROMPT_PATH = os.path.join(_PROMPT_DIR, "system_prompt.txt")

def _load_system_prompt() -> str:
    try:
        with open(_SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.warning("System prompt not found at %s", _SYSTEM_PROMPT_PATH)
        return "You are Sylph, a helpful desktop AI companion."


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_recent_emails",
            "description": "Retrieve recent emails from Gmail inbox",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {"type": "integer", "description": "Number of emails to retrieve, default is 3"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_email",
            "description": "Read a specific email's content by ID",
            "parameters": {
                "type": "object",
                "properties": {
                    "msg_id": {"type": "string", "description": "The Gmail message ID"}
                },
                "required": ["msg_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "draft_email",
            "description": "Create a new draft email in Gmail",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email address"},
                    "subject": {"type": "string", "description": "Subject line"},
                    "body": {"type": "string", "description": "Content of the email"}
                },
                "required": ["to", "subject", "body"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an existing email draft by draft ID. Requires user confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "draft_id": {"type": "string", "description": "The draft ID to send"}
                },
                "required": ["draft_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_events_today",
            "description": "List all calendar events scheduled for today",
            "parameters": {}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_events_range",
            "description": "List calendar events within a specified start and end date/time range",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_iso": {"type": "string", "description": "ISO start date-time (e.g. 2026-06-11T00:00:00Z)"},
                    "end_iso": {"type": "string", "description": "ISO end date-time (e.g. 2026-06-11T23:59:59Z)"}
                },
                "required": ["start_iso", "end_iso"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_event",
            "description": "Create a new event in Google Calendar. Requires user confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Title of the calendar event"},
                    "start_iso": {"type": "string", "description": "Start date-time in ISO format (e.g. 2026-06-11T14:00:00)"},
                    "end_iso": {"type": "string", "description": "End date-time in ISO format (e.g. 2026-06-11T15:00:00)"},
                    "description": {"type": "string", "description": "Optional event details"}
                },
                "required": ["title", "start_iso", "end_iso"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_event",
            "description": "Delete a calendar event by its event ID. Requires user confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string", "description": "The Google Calendar event ID to delete"}
                },
                "required": ["event_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information using DuckDuckGo",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_page",
            "description": "Fetch a web page's URL and read its text content",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The web page URL"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories within a local path",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "Path to list (e.g. '~/Desktop')"}
                },
                "required": ["directory"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read text/PDF content from a local file path",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Local file path (e.g. '~/Desktop/document.pdf')"}
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search local files by name or content in a directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search term"},
                    "directory": {"type": "string", "description": "Directory path to search"}
                },
                "required": ["query", "directory"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "move_avatar",
            "description": "Move your own avatar body to a position on the user's screen. Use when the user asks you to move, relocate, get out of the way, or sit somewhere.",
            "parameters": {
                "type": "object",
                "properties": {
                    "position": {
                        "type": "string",
                        "enum": ["left", "right", "center", "bottom_left", "bottom_right", "hide", "show"],
                        "description": "Target screen position"
                    }
                },
                "required": ["position"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "look_at_screen",
            "description": "Capture and analyze the user's current screen contents (OCR text and active window name)",
            "parameters": {}
        }
    }
]


# ---------------------------------------------------------------------------
# Keyword-based emotion inference (no LLM call — a fast regex-free scan)
# ---------------------------------------------------------------------------

_EMOTION_KEYWORDS: dict[str, list[str]] = {
    "happy": ["happy", "great", "wonderful", "love", "excellent", "excited",
              "glad", "yay", "awesome", "fantastic", "haha", "lol", "fun",
              "perfect", "amazing", "yes!", "absolutely", "delightful"],
    "angry": ["frustrating", "annoying", "ridiculous", "no way", "seriously",
              "unbelievable", "stop", "wrong", "mistake", "problem", "ugh",
              "terrible", "awful", "broken", "fix this", "again?"],
    "sad": ["sorry", "unfortunately", "can't", "unable", "failed", "sad",
            "difficult", "hard", "miss", "lost", "broken", "gone", "not working"],
    "relaxed": ["sure", "of course", "no problem", "happy to help", "understood",
                "makes sense", "alright", "okay", "got it", "certainly"],
    "surprised": ["wow", "really?", "that's interesting", "didn't expect",
                  "surprising", "unexpected", "oh!", "wait", "whoa", "actually"],
}


def infer_emotion(text: str) -> str:
    """
    Infer the emotional tone of a piece of text by keyword scoring.
    Returns one of 'happy', 'angry', 'sad', 'relaxed', 'surprised', 'neutral'.

    Used per-response for mood impulses AND per-clause by the TTS pipeline so
    the voice can carry the emotion of the sentence being spoken right now —
    all without a single extra LLM call.
    """
    if not text:
        return "neutral"
    text_lower = text.lower()
    scores = {label: 0 for label in _EMOTION_KEYWORDS}
    for label, words in _EMOTION_KEYWORDS.items():
        for w in words:
            if w in text_lower:
                scores[label] += 1
    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    return best if scores[best] > 0 else "neutral"


class ThinkFilter:
    """Filters out <think>...</think> tags and contents from a stream of text."""

    def __init__(self):
        self.in_think = False
        self.buffer = ""

    def filter(self, text: str) -> str:
        self.buffer += text
        out_parts = []
        
        while True:
            if self.in_think:
                idx = self.buffer.find("</think>")
                if idx != -1:
                    # Found end tag. Skip it and turn off in_think.
                    self.buffer = self.buffer[idx + 8:]
                    self.in_think = False
                else:
                    # Keep only the last 7 chars in case </think> is split
                    keep = self.buffer[-7:] if len(self.buffer) > 7 else self.buffer
                    self.buffer = keep
                    return "".join(out_parts)
            else:
                idx = self.buffer.find("<think>")
                if idx != -1:
                    before = self.buffer[:idx]
                    out_parts.append(before)
                    self.buffer = self.buffer[idx + 7:]
                    self.in_think = True
                else:
                    # No <think> found. Output everything except the last 6 chars
                    if len(self.buffer) > 6:
                        out = self.buffer[:-6]
                        out_parts.append(out)
                        self.buffer = self.buffer[-6:]
                        return "".join(out_parts)
                    else:
                        return "".join(out_parts)

    def flush(self) -> str:
        if not self.in_think:
            return self.buffer
        return ""


class ConversationPlanner:
    """
    Orchestrates the full conversation flow:
    user transcript → prompt assembly → LLM streaming → sentence splitting → TTS routing
    """

    def __init__(
        self,
        ollama_client: OllamaClient,
        tts_callback: Optional[Callable[[str], Awaitable[None]]] = None,
        memory_retriever: Optional[Callable[[str], Awaitable[list[str]]]] = None,
        confirm_callback: Optional[Callable[[str, dict], Awaitable[bool]]] = None,
        capture_callback: Optional[Callable[[], Awaitable[dict]]] = None,
        mood_callback: Optional[Callable[[str], Awaitable[None]]] = None,
        intent_callback: Optional[Callable[[str, dict], Awaitable[None]]] = None,
    ):
        self.ollama = ollama_client
        self.splitter = SentenceSplitter()
        self.tts_callback = tts_callback
        self.memory_retriever = memory_retriever
        self.confirm_callback = confirm_callback
        self.capture_callback = capture_callback
        self.mood_callback = mood_callback
        self.intent_callback = intent_callback  # broadcasts avatar/system commands to the frontend

        # Conversation state
        self._message_history: list[dict] = []
        self._max_history = 20  # Sliding window
        self._user_name = "User"
        self._current_mood = "neutral"
        self._system_prompt_template = _load_system_prompt()

        # Filler phrases for tool execution — keeps the conversation alive
        self._tool_fillers = [
            "Let me check that for you.",
            "One moment, looking into that.",
            "On it — just a second.",
            "Sure, pulling that up.",
            "Let me take a look.",
        ]
        self._filler_index = 0

        # Tool map
        self._tool_map = {
            "list_recent_emails": tools.list_recent_emails,
            "read_email": tools.read_email,
            "draft_email": tools.draft_email,
            "send_email": tools.send_email,
            "list_events_today": tools.list_events_today,
            "list_events_range": tools.list_events_range,
            "create_event": tools.create_event,
            "delete_event": tools.delete_event,
            "search_web": tools.search_web,
            "read_page": tools.read_page,
            "list_files": tools.list_files,
            "read_file": tools.read_file,
            "search_files": tools.search_files,
            "look_at_screen": self.look_at_screen,
            "move_avatar": self.move_avatar,
        }

    def set_user_name(self, name: str) -> None:
        self._user_name = name

    def set_mood(self, mood: str) -> None:
        self._current_mood = mood

    def _build_system_prompt(self, memories: list[str], screen_context: str = "") -> str:
        """Assemble the full system prompt with dynamic context."""
        now = datetime.now()

        # Format memories block
        if memories:
            memories_text = "[MEMORIES]\n" + "\n".join(f"- {m}" for m in memories) + "\n[/MEMORIES]"
        else:
            memories_text = ""

        # Format screen context block
        if screen_context:
            screen_text = f"[SCREEN_CONTEXT]\n{screen_context}\n[/SCREEN_CONTEXT]"
        else:
            screen_text = ""

        return self._system_prompt_template.format(
            mood=self._current_mood,
            time=now.strftime("%I:%M %p, %A %B %d"),
            user_name=self._user_name,
            memories_block=memories_text,
            screen_context_block=screen_text,
        )

    def _get_windowed_history(self) -> list[dict]:
        """
        Return the last N messages for context, never starting mid tool
        exchange (a window that opens with a dangling 'tool' message is
        rejected or misinterpreted by Ollama's chat template).
        """
        window = self._message_history[-self._max_history:]
        while window and window[0].get("role") == "tool":
            window = window[1:]
        return window

    async def _async_token_generator(self, text: str):
        """Yield words from a static string to simulate a streaming generator."""
        words = text.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            await asyncio.sleep(0.01)

    async def process_user_message(
        self,
        user_text: str,
        screen_context: str = "",
        image: Optional[str] = None,
    ) -> str:
        """
        Full pipeline: user message → LLM → tool execution / sentence split → TTS.

        `image` (base64 JPEG) is attached to the FIRST model call so a
        multimodal brain (qwen3.5:4b) can actually see the screen, not just read
        OCR text. Sent once — not re-attached on tool-loop iterations.
        """
        logger.info("Processing user message: '%s'%s", user_text[:80], " [+image]" if image else "")

        # Step 0: Deterministic intent routing — spatial/avatar commands never
        # touch the LLM. Parsed in <1ms, executed immediately, answered with a
        # short in-character ack. The avatar starts moving before a normal
        # pipeline would have produced its first token.
        intent = intent_router.route(user_text)
        if intent is not None:
            self._message_history.append({"role": "user", "content": user_text})
            if self.intent_callback:
                try:
                    await self.intent_callback(intent.name, intent.params)
                except Exception as e:
                    logger.error("Intent callback failed: %s", e)
            if intent.ack and self.tts_callback:
                try:
                    await self.tts_callback(intent.ack)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.warning("Intent ack TTS failed: %s", e)
            self._message_history.append({"role": "assistant", "content": intent.ack})
            return intent.ack

        # Step 1: Retrieve relevant memories
        memories: list[str] = []
        if self.memory_retriever:
            try:
                memories = await self.memory_retriever(user_text)
                logger.debug("Retrieved %d memories", len(memories))
            except Exception as e:
                logger.warning("Memory retrieval failed: %s", e)

        # Step 2: Build the full prompt
        system_prompt = self._build_system_prompt(memories, screen_context)

        # Step 3: Assemble message list
        self._message_history.append({"role": "user", "content": user_text})
        messages = [
            {"role": "system", "content": system_prompt},
            *self._get_windowed_history(),
        ]

        # Step 4: Run the tool calling agent loop (bounded — see MAX_TOOL_ITERATIONS)
        for _tool_iter in range(MAX_TOOL_ITERATIONS):
            logger.info("Calling Ollama (stream_chat_raw) to check for tool calls... (round %d/%d)",
                        _tool_iter + 1, MAX_TOOL_ITERATIONS)
            
            # Mood-based sampling parameters (Item 4)
            sampling = get_sampling_params(self._current_mood)
            logger.info("Mood '%s' → sampling: temp=%.2f, top_p=%.2f, rep_pen=%.2f",
                        self._current_mood, sampling["temperature"], sampling["top_p"], sampling["repetition_penalty"])
            
            # Connection retries live inside the client now (the old loop here
            # wrapped lazy generator creation, which can never fail — real
            # network errors surface on first iteration, handled below).
            # Attach the screenshot only on the first pass so the vision model
            # sees it once; tool-loop follow-ups are text-only.
            turn_images = [image] if (image and _tool_iter == 0) else None
            # Only advertise tools whose capability domain is enabled in settings
            # (Phase 2) so the model doesn't reach for disabled abilities.
            allowed_tools = [t for t in TOOLS if permissions.is_tool_allowed(t["function"]["name"])]
            stream = self.ollama.stream_chat_raw(
                messages, tools=allowed_tools,
                images=turn_images,
                temperature=sampling["temperature"],
                top_p=sampling["top_p"],
                repetition_penalty=sampling["repetition_penalty"],
            )

            # Buffer chunks until we can decide if it's a tool call or text response
            buffered_chunks = []
            tool_calls = []
            is_tool_call = False
            is_text_response = False

            think_filter = ThinkFilter()
            buffered_filtered_text = ""

            try:
                async for chunk in stream:
                    buffered_chunks.append(chunk)

                    # Check for tool calls
                    msg = chunk.get("message", {})
                    tc = msg.get("tool_calls", [])
                    if tc:
                        tool_calls.extend(tc)
                        is_tool_call = True
                        break

                    # Check for user-visible text
                    content = msg.get("content", "")
                    if content:
                        filtered = think_filter.filter(content)
                        if filtered:
                            buffered_filtered_text += filtered
                            is_text_response = True
                            break
            except asyncio.CancelledError:
                logger.info("Ollama stream reading was cancelled.")
                raise
            except OllamaError as e:
                # Cloud unreachable / quota / auth — explain it out loud once
                # and end the turn. The typed user_message says what actually
                # happened (offline vs. capped vs. bad key).
                logger.error("LLM turn failed before first token: %s", e.detail)
                if self.tts_callback:
                    try:
                        await self.tts_callback(e.user_message)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        pass
                self._message_history.append(
                    {"role": "assistant", "content": e.user_message}
                )
                return e.user_message
            except Exception as e:
                logger.error("Error reading from Ollama stream during buffering: %s", e)

            if not is_tool_call and not is_text_response:
                flushed = think_filter.flush()
                if flushed:
                    buffered_filtered_text += flushed
                is_text_response = True

            if is_tool_call:
                # Emit a filler phrase immediately so the user hears feedback
                # and the mic auto-mutes (tts_start goes out before synthesis).
                filler = self._tool_fillers[self._filler_index % len(self._tool_fillers)]
                self._filler_index += 1
                if self.tts_callback:
                    try:
                        await self.tts_callback(filler)
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        logger.warning("Tool filler TTS failed: %s", e)

                # It is a tool call! Read the rest of the stream to accumulate tool calls
                try:
                    async for chunk in stream:
                        tc = chunk.get("message", {}).get("tool_calls")
                        if tc:
                            tool_calls.extend(tc)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error("Error reading tool calls from Ollama stream: %s", e)

                # Store the assistant response with tool calls in history
                assistant_msg = {"role": "assistant", "content": "", "tool_calls": tool_calls}
                messages.append(assistant_msg)
                self._message_history.append(assistant_msg)

                # Process each requested tool call
                for call in tool_calls:
                    func_name = call.get("function", {}).get("name")
                    func_args = call.get("function", {}).get("arguments", {})
                    if isinstance(func_args, str):
                        try:
                            import json
                            func_args = json.loads(func_args)
                        except Exception:
                            func_args = {}

                    logger.info("Executing tool: %s with args: %s", func_name, func_args)
                    tool_result = ""

                    # Phase 2: enforce capability permissions. If the domain is
                    # off (or a file path is outside the allowlist), refuse
                    # before confirming/executing and tell the model why.
                    denied = permissions.check_tool(func_name, func_args if isinstance(func_args, dict) else {})
                    if denied:
                        logger.info("Tool %s blocked by permissions: %s", func_name, denied)
                        messages.append({"role": "tool", "name": func_name, "content": denied})
                        self._message_history.append({"role": "tool", "name": func_name, "content": denied})
                        continue

                    # Check for user confirmation
                    requires_confirm = func_name in ["send_email", "create_event", "delete_event"]
                    confirmed = True
                    if requires_confirm and self.confirm_callback:
                        logger.info("Requesting user confirmation for tool %s...", func_name)
                        confirmed = await self.confirm_callback(func_name, func_args)
                        logger.info("User confirmation result for tool %s: %s", func_name, confirmed)

                    if not confirmed:
                        tool_result = "Action cancelled by the user."
                    else:
                        func = self._tool_map.get(func_name)
                        if func:
                            try:
                                import inspect
                                if inspect.iscoroutinefunction(func):
                                    result_obj = await func(**func_args)
                                else:
                                    result_obj = func(**func_args)

                                if isinstance(result_obj, (dict, list)):
                                    import json
                                    tool_result = json.dumps(result_obj, indent=2)
                                else:
                                    tool_result = str(result_obj)
                            except Exception as err:
                                logger.error("Error executing tool %s: %s", func_name, err)
                                tool_result = f"Error: {err}"
                        else:
                            tool_result = f"Error: Tool {func_name} not found."

                    # Append tool results to message list for the next Ollama call
                    tool_msg = {
                        "role": "tool",
                        "name": func_name,
                        "content": tool_result
                    }
                    messages.append(tool_msg)
                    self._message_history.append(tool_msg)
            else:
                # It is a normal text response! Stream it to the sentence splitter and TTS
                async def text_generator():
                    # Yield any text buffered during decision-making
                    if buffered_filtered_text:
                        yield buffered_filtered_text

                    # Process the rest of the stream content
                    try:
                        async for chunk in stream:
                            content = chunk.get("message", {}).get("content", "")
                            if content:
                                filtered = think_filter.filter(content)
                                if filtered:
                                    yield filtered
                    except asyncio.CancelledError:
                        raise
                    except OllamaError as e:
                        # Stream died mid-response: whatever was said stands;
                        # append the explanation so the cutoff isn't silent.
                        logger.error("LLM stream died mid-response: %s", e.detail)
                        yield " " + e.user_message
                    except Exception as e:
                        logger.error("Error reading text from Ollama stream: %s", e)

                    # Flush any remaining text in the filter
                    flushed = think_filter.flush()
                    if flushed:
                        yield flushed

                # Pipe text generator to sentence splitter, with TTS fully
                # DECOUPLED via a bounded queue. In the old serial loop, the
                # planner awaited synthesis of clause N before reading clause
                # N+1 from Ollama — so LLM streaming and TTS executed back to
                # back instead of overlapping. With the queue, Ollama keeps
                # streaming while Kokoro synthesizes, and total turn time
                # collapses to max(LLM time, TTS time) instead of their sum.
                full_content: list[str] = []
                clause_q: asyncio.Queue = asyncio.Queue(maxsize=8)

                async def tts_consumer():
                    while True:
                        clause = await clause_q.get()
                        if clause is None:
                            return
                        logger.info("Clause → TTS: '%s'", clause[:60])
                        if self.tts_callback:
                            try:
                                await self.tts_callback(clause)
                            except asyncio.CancelledError:
                                raise
                            except Exception as e:
                                logger.error("TTS callback failed: %s", e)

                consumer_task = asyncio.create_task(tts_consumer())
                try:
                    async for clause in self.splitter.split(text_generator()):
                        full_content.append(clause)
                        await clause_q.put(clause)
                    await clause_q.put(None)
                    await consumer_task
                except asyncio.CancelledError:
                    consumer_task.cancel()
                    raise
                except Exception:
                    consumer_task.cancel()
                    raise

                response_text = " ".join(full_content).strip()
                self._message_history.append({"role": "assistant", "content": response_text})
                logger.info("Full response (%d chars): '%s'", len(response_text), response_text[:100])

                # Infer emotional tone from the response and notify the mood system
                await self._fire_emotion_from_response(response_text)

                return response_text

        # Tool-iteration budget exhausted without the model settling on a text
        # answer — stop chaining tools and say so rather than hang or return "".
        logger.warning("Tool loop hit MAX_TOOL_ITERATIONS (%d) without a text reply", MAX_TOOL_ITERATIONS)
        fallback = "I got a bit tangled chaining tools there — let me stop and answer directly next time."
        if self.tts_callback:
            try:
                await self.tts_callback(fallback)
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
        self._message_history.append({"role": "assistant", "content": fallback})
        return fallback

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._message_history.clear()
        logger.info("Conversation history cleared")

    def load_history(self, messages: list[dict]) -> None:
        """Seed conversation history from a persisted store on startup.

        Only plain user/assistant turns are restored (tool exchanges are not
        replayed — a window that opens mid tool-call confuses the chat
        template), capped to the sliding window. (Phase C: continuity across
        restarts, since main.py assigns a fresh in-memory thread each boot.)
        """
        restored = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]
        if restored:
            self._message_history = restored[-self._max_history:]
            logger.info("Restored %d messages into planner history", len(self._message_history))

    async def move_avatar(self, position: str = "center") -> str:
        """LLM-callable avatar relocation (fallback for phrasing the
        deterministic router didn't match)."""
        if not self.intent_callback:
            return "Error: avatar movement not wired to frontend."
        pos = str(position).lower().strip()
        if pos == "hide":
            await self.intent_callback("avatar_hide", {})
        elif pos == "show":
            await self.intent_callback("avatar_show", {})
        else:
            await self.intent_callback("avatar_move", {"position": pos})
        return f"Avatar is now moving to '{pos}'. Movement started successfully."

    async def look_at_screen(self) -> str:
        """Capture and analyze the screen on demand."""
        if not self.capture_callback:
            return "Error: Screen capture callback not registered."
        logger.info("look_at_screen tool invoked: requesting capture from frontend")
        # Shield the capture from cancellation — barge-in must not discard
        # the in-flight capture request/response round-trip.
        res = await asyncio.shield(self.capture_callback())
        ocr_text = res.get("ocr_text", "")
        window_name = res.get("window_name", "")
        return f"Active Window: {window_name}\nOCR Text found on screen:\n{ocr_text}"

    async def _fire_emotion_from_response(self, response_text: str) -> None:
        """
        Lightweight keyword-based emotion inference from the LLM response text.
        Maps detected sentiment → mood impulse via mood_callback.

        This avoids an extra LLM call — just a fast regex over the response.
        The inferred emotion drives the avatar expression and TTS speed for the
        *next* utterance, creating a natural carry-over feel.
        """
        if not self.mood_callback or not response_text:
            return

        best_emotion = infer_emotion(response_text)
        logger.debug("Emotion inference selected '%s'", best_emotion)

        try:
            await self.mood_callback(best_emotion)
        except Exception as e:
            logger.warning("Mood callback error in emotion inference: %s", e)

    @property
    def history(self) -> list[dict]:
        return self._message_history.copy()

    @property
    def history_length(self) -> int:
        return len(self._message_history)


