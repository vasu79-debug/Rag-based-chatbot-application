import laya
import asyncio

async def test():
    runner = laya.load("convaiinnovations/laya")
    questions = {
        "intent": {
            "type": "choice",
            "instructions": "What is the user's primary intent?",
            "criteria": {
                "subscription": "Questions about subscriptions",
                "other": "Anything else"
            }
        },
        "needs_browser": {
            "type": "noul",
            "instructions": "Does this request require browser interaction?"
        }
    }
    
    res = await asyncio.to_thread(
        laya.decide,
        runner,
        state={"message": "delete my subscription"},
        questions=questions,
        return_details=True
    )
    
    ans = res.answers
    print(ans)
    print("Intent choice:", ans["intent"].choice)
    print("Intent conf:", ans["intent"].confidence)
    print("Browser noul:", ans["needs_browser"].noul)

if __name__ == "__main__":
    asyncio.run(test())
