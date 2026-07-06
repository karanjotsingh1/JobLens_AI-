# ============================================================
# agent/career_agent.py
#
# PURPOSE: LangGraph-powered Career Coach Agent
#
# WHAT THIS AGENT DOES:
#   1. Takes user query + resume context + gap analysis as input
#   2. Decides which TOOL to use (web search / resume rewriter / skill planner)
#   3. Executes the tool → observes result → decides next step
#   4. Finally generates a detailed, helpful response
#
# TOOLS AVAILABLE:
#   - web_search_tool      : Searches DuckDuckGo for learning resources (FREE)
#   - rewrite_bullet_tool  : Rewrites weak resume bullet points using LLM
#   - skill_plan_tool      : Generates 30-day skill building plan with links
#   - gap_analysis_tool    : Finds missing skills between resume and JD
#
# WHY LANGGRAPH?
#   LangGraph allows the agent to LOOP — it can search the web,
#   see the results, decide to search again with better query,
#   then synthesize everything into one final answer.
#   Simple LLM chains cannot do this multi-step reasoning.
# ============================================================

import os
import sys
from typing import Annotated, TypedDict, List
from langchain_groq               import ChatGroq
from langchain_core.messages      import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools         import tool
from langgraph.graph              import StateGraph, START, END
from langgraph.graph.message      import add_messages
from langgraph.prebuilt           import ToolNode, tools_condition
from duckduckgo_search            import DDGS

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GROQ_API_KEY, GROQ_MODEL_NAME, WEB_SEARCH_RESULTS, AGENT_MAX_ITERATIONS


# ─────────────────────────────────────────────────────────────
# 1. AGENT STATE — Shared memory across all graph nodes
# ─────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    """
    TypedDict that defines what information the agent carries
    through each node in the LangGraph.

    messages: Full conversation history (add_messages = append, not replace)
    resume_context: Extracted text from user's resume PDF
    gap_analysis:   Missing skills identified between resume and JD
    role_target:    The job role user is targeting (e.g. "ML Engineer")
    level_target:   Experience level (e.g. "Fresher")
    """
    messages      : Annotated[List, add_messages]
    resume_context: str
    gap_analysis  : str
    role_target   : str
    level_target  : str


# ─────────────────────────────────────────────────────────────
# 2. TOOLS — What the agent can DO
# ─────────────────────────────────────────────────────────────

@tool
def web_search_learning_resources(query: str) -> str:
    """
    Search the web for FREE learning resources — YouTube playlists,
    courses, tutorials, and documentation for a skill or topic.

    Use this when the user wants to learn a missing skill.
    Prefer YouTube playlists, Coursera free audits, and official docs.

    Args:
        query: Search query e.g. "best free LangChain tutorial YouTube 2024"
    """
    try:
        # DuckDuckGo search — completely FREE, no API key needed
        with DDGS() as ddgs:
            results = list(ddgs.text(
                query,
                max_results=WEB_SEARCH_RESULTS,
                region="in-en",       # India-focused results
                safesearch="moderate"
            ))

        if not results:
            return "No results found. Try a different search query."

        # Format results nicely for the LLM to process
        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(
                f"[{i}] TITLE: {r.get('title', 'N/A')}\n"
                f"    URL:   {r.get('href', 'N/A')}\n"
                f"    DESC:  {r.get('body', 'N/A')[:200]}\n"
            )

        return "\n".join(formatted)

    except Exception as e:
        return f"Search failed: {str(e)}. Please try again."


@tool
def rewrite_resume_bullet(bullet_point: str, role: str) -> str:
    """
    Rewrite a weak resume bullet point to make it stronger,
    more impactful, and ATS (Applicant Tracking System) friendly.

    Use strong action verbs, quantify impact where possible,
    and use keywords relevant to the target role.

    Args:
        bullet_point: The original weak bullet point from resume
        role:         Target job role (e.g. "ML Engineer")
    """
    # This tool will be handled by the LLM itself using its context
    # We return a structured prompt that the agent node processes
    return f"REWRITE_REQUEST|bullet={bullet_point}|role={role}"


@tool
def generate_skill_learning_plan(missing_skills: str, role: str, level: str) -> str:
    """
    Generate a detailed 30-day skill building plan for missing skills.
    Include specific YouTube channels, free courses, and practice projects.

    Args:
        missing_skills: Comma-separated list of skills to learn
        role:           Target job role
        level:          Experience level (Fresher/Mid-Level/Senior)
    """
    return f"PLAN_REQUEST|skills={missing_skills}|role={role}|level={level}"


@tool
def analyze_skill_gaps(resume_skills: str, jd_requirements: str) -> str:
    """
    Compare resume skills against JD requirements to find gaps.

    Args:
        resume_skills:    Skills extracted from resume (comma-separated)
        jd_requirements:  Skills required in job description (comma-separated)
    """
    resume_set = set(s.strip().lower() for s in resume_skills.split(","))
    jd_set     = set(s.strip().lower() for s in jd_requirements.split(","))

    missing  = jd_set - resume_set
    matching = jd_set & resume_set
    extra    = resume_set - jd_set

    return (
        f"✅ MATCHING SKILLS ({len(matching)}): {', '.join(sorted(matching)) or 'None'}\n"
        f"❌ MISSING SKILLS ({len(missing)}): {', '.join(sorted(missing)) or 'None'}\n"
        f"➕ EXTRA SKILLS (not in JD) ({len(extra)}): {', '.join(sorted(extra)) or 'None'}"
    )


# List of all tools the agent can use
TOOLS = [
    web_search_learning_resources,
    rewrite_resume_bullet,
    generate_skill_learning_plan,
    analyze_skill_gaps,
]


# ─────────────────────────────────────────────────────────────
# 3. LLM SETUP — Groq with tool binding
# ─────────────────────────────────────────────────────────────
def get_llm_with_tools():
    """
    Initialize Groq LLM and bind all tools to it.
    bind_tools() tells the LLM what tools are available
    and what arguments each tool expects.
    """
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL_NAME,
        temperature=0.3,        # Low temp = focused, precise answers
        max_tokens=4096,        # Allow long detailed responses
    )
    # Bind tools so LLM knows it can call them
    return llm.bind_tools(TOOLS)


# ─────────────────────────────────────────────────────────────
# 4. AGENT NODE — The "brain" that decides what to do
# ─────────────────────────────────────────────────────────────
def agent_node(state: AgentState) -> AgentState:
    """
    The main reasoning node. Called by LangGraph at each step.

    It receives the current state (all messages + context),
    thinks about what to do next, and either:
    - Calls a tool (returns tool_call in message)
    - Gives a final answer (returns regular AI message)
    """
    llm_with_tools = get_llm_with_tools()

    # Build a detailed system prompt that includes all context
    system_prompt = f"""You are JobLens AI — an expert career coach and resume consultant.
You help users land their dream jobs by providing specific, actionable, detailed advice.

CONTEXT ABOUT THE USER:
- Target Role: {state.get('role_target', 'Not specified')}
- Experience Level: {state.get('level_target', 'Not specified')}
- Resume Summary: {state.get('resume_context', 'Resume not uploaded yet')[:1500]}
- Skill Gaps Identified: {state.get('gap_analysis', 'No gap analysis done yet')}

YOUR PERSONALITY:
- Be specific — never give vague advice like "learn Python". Say "Start with this playlist: [link]"
- Be detailed — when giving learning resources, explain WHAT the resource covers and WHY it is good
- Be encouraging but honest — tell the user exactly what they need to improve
- Always provide YouTube links, course links, and documentation links when discussing learning

TOOLS YOU HAVE:
1. web_search_learning_resources — Search for free learning resources, YouTube playlists, courses
2. rewrite_resume_bullet — Rewrite a weak resume bullet point to be strong and ATS-friendly
3. generate_skill_learning_plan — Create a detailed 30-day learning roadmap
4. analyze_skill_gaps — Compare resume skills vs JD to find what is missing

IMPORTANT RULES:
- For learning resources: ALWAYS search the web first to get real, current links
- For resume rewrites: Be specific about what makes the new version better
- For skill plans: Break down into weekly goals with specific resources per week
- Never say "I cannot help with that" — find a way to be useful
"""

    # Prepend system message to conversation
    messages = [SystemMessage(content=system_prompt)] + state["messages"]

    # Call LLM — it will either return a tool call or a final answer
    response = llm_with_tools.invoke(messages)

    # Return updated state with new message appended
    return {"messages": [response]}


# ─────────────────────────────────────────────────────────────
# 5. TOOL EXECUTION NODE — Actually runs the tool
# ─────────────────────────────────────────────────────────────
def tool_executor_node(state: AgentState) -> AgentState:
    """
    Processes tool calls made by the agent node.
    After tool execution, the result is added to messages
    so the agent can see what the tool returned.

    Special handling for REWRITE_REQUEST and PLAN_REQUEST —
    these are handled by calling the LLM again with detailed prompts.
    """
    tool_node = ToolNode(TOOLS)
    result    = tool_node.invoke(state)

    # Check if any tool returned a special request that needs LLM
    updated_messages = result.get("messages", [])
    processed        = []

    for msg in updated_messages:
        if isinstance(msg, ToolMessage):
            content = msg.content

            # Handle bullet rewrite request — call LLM for this
            if content.startswith("REWRITE_REQUEST"):
                parts  = dict(p.split("=", 1) for p in content.split("|")[1:])
                bullet = parts.get("bullet", "")
                role   = parts.get("role", "Software Engineer")

                llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL_NAME, temperature=0.2)
                rewrite_prompt = f"""Rewrite this resume bullet point for a {role} role.

ORIGINAL: {bullet}

RULES FOR REWRITING:
1. Start with a STRONG action verb (Engineered, Architected, Optimized, Deployed, Built, Developed)
2. Include WHAT you did, HOW you did it, and the IMPACT/RESULT
3. Add relevant technical keywords for {role}
4. Keep it to 1-2 lines max
5. Quantify impact where possible (e.g., "reduced latency by 40%", "processed 10K+ requests/day")

Return 3 different versions of the rewritten bullet, labeled VERSION 1, VERSION 2, VERSION 3.
After each version, explain in 1 line what makes it stronger.
"""
                llm_response = llm.invoke(rewrite_prompt)
                msg          = ToolMessage(
                    content   = llm_response.content,
                    tool_call_id = msg.tool_call_id
                )

            # Handle skill plan request — call LLM for detailed plan
            elif content.startswith("PLAN_REQUEST"):
                parts  = dict(p.split("=", 1) for p in content.split("|")[1:])
                skills = parts.get("skills", "")
                role   = parts.get("role", "ML Engineer")
                level  = parts.get("level", "Fresher")

                llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL_NAME,
                               temperature=0.2, max_tokens=4096)
                plan_prompt = f"""Create a detailed 30-day learning plan for a {level} targeting {role}.

SKILLS TO LEARN: {skills}

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:

## 🎯 30-Day Learning Roadmap: {role} ({level})

### Week 1 (Days 1-7): [Skill Name] Foundations
**Goal:** What you will achieve this week
**Daily time commitment:** X hours/day

📺 YouTube Resources:
- [Channel Name] — "Playlist/Video Title" → [URL]
  What it covers: detailed description of what this resource teaches
  Why this resource: why this specific channel/video is recommended

🌐 Free Courses:
- [Platform] — "Course Name" → [URL]
  What it covers: description
  Time to complete: X hours

📚 Documentation/Reading:
- [Resource name] → [URL]
  What to focus on: specific sections or chapters

🛠️ Practice Project:
Build [specific small project] using what you learned this week

---
### Week 2 (Days 8-14): [Next Skill]
[Same format as Week 1]

---
### Week 3 (Days 15-21): [Next Skill]
[Same format]

---
### Week 4 (Days 22-30): Integration + Portfolio
**Goal:** Combine all learned skills into one portfolio project
**Project idea:** [specific project idea that uses all {skills}]
**GitHub-ready steps:** How to structure and document this project

💡 Tips for {role} Interviews:
- [Specific tip 1]
- [Specific tip 2]
- [Specific tip 3]

Remember: Provide REAL YouTube channel names, real course names, and accurate URLs.
Mention popular educators like: Krish Naik, Codebasics, StatQuest, Sentdex, Andrej Karpathy,
3Blue1Brown, Tech With Tim, Nicholas Renotte, etc. wherever relevant.
"""
                llm_response = llm.invoke(plan_prompt)
                msg          = ToolMessage(
                    content      = llm_response.content,
                    tool_call_id = msg.tool_call_id
                )

        processed.append(msg)

    return {"messages": processed}


# ─────────────────────────────────────────────────────────────
# 6. BUILD LANGGRAPH — Connect nodes with edges
# ─────────────────────────────────────────────────────────────
def build_career_agent():
    """
    Build and compile the LangGraph agent.

    GRAPH STRUCTURE:
    START → agent_node → (if tool call) → tool_executor_node → agent_node → ...
                       → (if no tool call) → END

    This loop allows the agent to:
    1. Decide to search web
    2. See search results
    3. Decide to search again with refined query
    4. See new results
    5. Finally synthesize and answer
    """
    graph = StateGraph(AgentState)

    # Add nodes to the graph
    graph.add_node("agent",  agent_node)
    graph.add_node("tools",  tool_executor_node)

    # START always goes to agent first
    graph.add_edge(START, "agent")

    # After agent runs: if it made a tool call → go to tools
    #                   if it gave final answer → go to END
    graph.add_conditional_edges(
        "agent",
        tools_condition,   # LangGraph built-in: checks if last message has tool_calls
        {
            "tools": "tools",   # If tool call detected → execute tools
            END:     END,       # If no tool call → we're done
        }
    )

    # After tools run → always go back to agent for next reasoning step
    graph.add_edge("tools", "agent")

    # Compile the graph into a runnable
    return graph.compile()


# ─────────────────────────────────────────────────────────────
# 7. PUBLIC INTERFACE — Used by Streamlit UI
# ─────────────────────────────────────────────────────────────
def run_career_agent(
    user_query     : str,
    resume_context : str = "",
    gap_analysis   : str = "",
    role_target    : str = "Software Engineer",
    level_target   : str = "Fresher",
    chat_history   : list = None
) -> str:
    """
    Main entry point called from the Streamlit UI.

    Args:
        user_query:     What the user typed in the chat box
        resume_context: Extracted text from uploaded resume
        gap_analysis:   Gap analysis result from RAG comparison
        role_target:    Target job role
        level_target:   Experience level
        chat_history:   Previous messages in the conversation

    Returns:
        Agent's final response as a string
    """
    agent = build_career_agent()

    # Build message list from chat history
    messages = []
    if chat_history:
        for msg in chat_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

    # Add current user query
    messages.append(HumanMessage(content=user_query))

    # Initial state for the agent
    initial_state = {
        "messages"      : messages,
        "resume_context": resume_context,
        "gap_analysis"  : gap_analysis,
        "role_target"   : role_target,
        "level_target"  : level_target,
    }

    # Run the agent (it loops internally until it reaches END)
    final_state = agent.invoke(
        initial_state,
        config={"recursion_limit": AGENT_MAX_ITERATIONS}
    )

    # Extract the last AI message as the final response
    final_messages = final_state.get("messages", [])
    for msg in reversed(final_messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content

    return "I could not generate a response. Please try again."
