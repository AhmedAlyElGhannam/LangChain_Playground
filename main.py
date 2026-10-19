from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

load_dotenv()

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0) # basically returns the value corresponding to the key (hash map/dict shenanigans)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"    >> Executing apply_discount(price='{price}', discount_tier='{discount_tier}')")
    discount_percentage = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentage.get(discount_tier, 0) # basically returns the value corresponding to the key (hash map/dict shenanigans)
    return round(price * (1 - (discount / 100)), 2)


def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}
    _llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = _llm.bind_tools(tools)

    # strict rules prevent it from hallucinating
    # VERY important when dealing with an unga bunga model
    messages = [
        SystemMessage(
            "You are a helpful shopping assistant. "
            "You have access to a product catalog tool"
            "and a discount tool.\n\n"
            "STRICT RULES - you must follow these exactly:\n"
            "1. NEVER guess or assume any product price. " 
            "You MUST call get_product_price first to get the real price.\n"
            "2. Only call apply_discount AGTER you have received "
            "a price from get_product_price - do NOT pass a made-up number.\n"
            "3. NEVER calculate discounts yourself using math. "
            "Always use the apply_discount tool.\n"
            "4. If the user does not specify a discount tier, " 
            "ask them which tier to use - do NOT assume one."
        ),
        HumanMessage(content=question)
    ]

    # keeps model thinking until MAX_ITERATIONS limit is hit
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        # do action -> may result in a tool call
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        # if no tool calls, this is the final answer
        if not tool_calls:
            # this is the final answer
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        # only process one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"    [Tool Selected] {tool_name} with args: {tool_args} & id: {tool_call_id}")

        # detects invalid tool invocation 
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        # call tool -> its result is observation
        observation = tool_to_use.invoke(tool_args)
        print(f"    [Tool Result] {observation}")

        # feedback loop -> refeed the ai_message (result of first thought iteration) + tool results (observation) into the agent message
        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )
    

if __name__ == "__main__":
    print("NI HAO!")
    run_agent("What is the price of a laptop after applying a gold discount?")
