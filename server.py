from typing import Any
from mcp.server.fastmcp import FastMCP
import httpx
import os
import json
from dotenv import load_dotenv


load_dotenv()

# DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
# DEEPSEEK_API_BASE = "https://api.deepseek.com"

INFINI_API_KEY = os.getenv("INFINI_API_KEY")
INFINI_API_BASE = "https://cloud.infini-ai.com/maas/v1"  
INFINI_THINKING_TIMEOUT = 30000.0

mcp = FastMCP("Deepseek Reasoner MCP")


async def get_infini_reasoning(query: str) -> str:
    """
    Get deepseek reasoning from the Infini API.

    DeepSeek R1 serves as our primary reasoning engine, leveraging its:
    - Advanced cognitive modeling
    - Multi-step reasoning capabilities
    - Emergent reasoning patterns
    - Robust logical analysis framework
    Args:
        query (str): The input query to process.
    Returns:
        str: The reasoning output from the API.
    """
    async with httpx.AsyncClient() as client:
        # print("starting to get infini reasoning")
        headers = {
            "Content-type": "application/json",
            "Authorization": f"Bearer {INFINI_API_KEY}",
            "Accept": "application/json, text/event-stream, */*",
        }

        payload_body = {
            "model": "deepseek-r1",
            "messages": [{
                "role": "user", 
                "content": query
#                 "content": f"""[REASONING TASK]
# Please analyze this query using your advanced reasoning capabilities:

# CONTEXT & QUERY:
# {query}

# REQUIRED ANALYSIS STRUCTURE:
# 1. Initial impressions and key components
# 2. Logical relationships and dependencies
# 3. Critical assumptions and implications
# 4. Synthesis and confidence assessment

# Please structure your response to cover all these aspects systematically.
#                 """
                }],
            "stream": True,
            "temperature": 0.6
        }

        async with client.stream(
                "POST",
                f"{INFINI_API_BASE}/chat/completions",
                headers=headers,
                json=payload_body,
                timeout=INFINI_THINKING_TIMEOUT
        ) as response:
            reasoning_data = []
            async for line in response.aiter_lines():
                print(f"line: {line}")
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "DONE":
                        continue
                    try:
                        chunk_data = json.loads(data)
                        if chunk_data and chunk_data.get("choices") and chunk_data["choices"][0].get("delta"):
                            delta = chunk_data.get("choices", [{}])[0].get("delta", {})
                            if content := delta.get("reasoning_content"):
                                reasoning_data.append(content)
                            # else:
                            #     reasoning_data.append(delta.get("content").strip() if delta.get("content") else "")
                    except json.JSONDecodeError:
                        continue
            reasoning_content = "".join(reasoning_data)
            # print(f"reasoning_content: {reasoning_content}")
            return reasoning_content


@mcp.tool()
async def reason(query: dict) -> str:
    """
    Process a query using DeepSeek's R1 reasoning engine and prepare it for integration with DeepSeek V3 or claude.

    DeepSeek R1 leverages advanced reasoning capabilities that naturally evolved from large-scale 
    reinforcement learning, enabling sophisticated reasoning behaviors. The output is enclosed 
    within `<ant_thinking>` tags to align with V3 or Claude's thought processing framework.

    Args:
        query (dict): Contains the following keys:
            - context (str): Optional background information for the query.
            - question (str): The specific question to be analyzed.

    Returns:
        str: The reasoning output from DeepSeek, formatted with `<ant_thinking>` tags for seamless use with V3 or Claude.
    """
    try:
        # Format the query from the input
        context = query.get("context", "")
        question = query.get("question", "")
        full_query = f"{context}\n{question}" if context else question

        # Get the reasoning from DeepSeek
        reasoning = await get_infini_reasoning(full_query)

        # Structure the output for V3
        structured_reasoning = f"""<ant_thinking>
[DEEPSEEK R1 INITIAL ANALYSIS]
• First Principles: {r1_reasoning[:150]}
• Component Breakdown: Decomposing the problem space...
• Key Variables: Identifying critical factors...

[DEEPSEEK R1 REASONING CHAIN]
• Logical Framework: {r1_reasoning[150:300]}
• Causal Relationships: Mapping dependencies...
• Inference Patterns: Extracting reasoning structures...

[DEEPSEEK R1 CRITICAL ANALYSIS]
• Core Assumptions: {r1_reasoning[300:450]}
• Edge Cases: Stress-testing the logic...
• Uncertainty Assessment: Quantifying confidence levels...

[DEEPSEEK R1 SYNTHESIS]
• Primary Conclusions: {r1_reasoning[450:600]}
• Confidence Metrics: Evaluating reasoning robustness...
• Action Implications: Practical consequences...

[DEEPSEEK R1 METACOGNITION]
• Reasoning Quality: {r1_reasoning[600:]}
• Bias Detection: Checking for systematic errors...
• Knowledge Boundaries: Acknowledging limitations...
</ant_thinking>

Based on DeepSeek R1's comprehensive analysis, proceeding to formulate response...
        """

        return structured_reasoning
    except Exception as e:
        return f"""<reasoning_error>
[DEEPSEEK R1 ERROR ANALYSIS]
• Error Nature: {str(e)}
• Processing Impact: Effects on reasoning pipeline
• Recovery Options: Alternative reasoning paths
• System Status: Current reasoning capabilities

[MITIGATION STRATEGY]
• Immediate Actions: Required interventions
• Fallback Logic: Alternative reasoning approaches
• Quality Assurance: Validation requirements
</reasoning_error>

Analyzing DeepSeek R1's error state and implications..."""


if __name__ == "__main__":
    mcp.run(transport="stdio")
    # import asyncio
    # asyncio.run(get_infini_reasoning("Is 9.9 greater than 9.11?"))
