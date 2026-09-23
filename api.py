from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langgraph.types import Command

from graph import app

api = FastAPI(
    title="AI Travel Planner API",
    description="AI-powered travel planning API",
    version="1.0.0",
)

class TravelPlanRequest(BaseModel):

    user_query: str = Field(
        ...,
        min_length=3,
        description="User's travel planning request"
    )

    user_id: str = Field(
        default="android-user"
    )

    trip_constraints: dict[str, Any] = Field(
        default_factory=dict
    )


class ApprovalRequest(BaseModel):

    approved: bool

    feedback: str = ""

@api.get("/")
async def root():

    return {
        "success": True,
        "service": "AI Travel Planner API",
        "status": "running"
    }


@api.get("/health")
async def health():

    return {
        "success": True,
        "status": "healthy"
    }

@api.post("/api/v1/ai/travel-plan")
async def create_travel_plan(
    request: TravelPlanRequest
):

    thread_id = str(uuid4())

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    initial_state = {

        "user_id":
            request.user_id,

        "user_query":
            request.user_query,

        "trip_constraints":
            request.trip_constraints,

        "messages": [],

        "llm_calls": 0
    }

    try:

        result = await app.ainvoke(
            initial_state,
            config=config
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    interrupts = result.get(
        "__interrupt__"
    )

    if interrupts:

        interrupt_value = (
            interrupts[0].value
            if hasattr(
                interrupts[0],
                "value"
            )
            else interrupts[0]
        )

        return {

            "success": True,

            "status": "approval_required",

            "thread_id":
                thread_id,

            "approval":
                interrupt_value,

            "itinerary":
                result.get(
                    "itinerary",
                    ""
                ),

            "approval_request":
                result.get(
                    "approval_request",
                    ""
                )
        }

    return {

        "success": True,

        "status": "completed",

        "thread_id":
            thread_id,

        "response":
            result.get(
                "final_response",
                ""
            ),

        "itinerary":
            result.get(
                "itinerary",
                ""
            )
    }
@api.post(
    "/api/v1/ai/travel-plan/{thread_id}/approve"
)
async def approve_travel_plan(
    thread_id: str,
    request: ApprovalRequest
):

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    resume_value = {

        "approved":
            request.approved,

        "feedback":
            request.feedback
    }

    try:

        result = await app.ainvoke(
            Command(
                resume=resume_value
            ),
            config=config
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


    return {

        "success": True,

        "status": "completed",

        "thread_id":
            thread_id,

        "approved":
            request.approved,

        "response":
            result.get(
                "final_response",
                ""
            ),

        "itinerary":
            result.get(
                "itinerary",
                ""
            )
    }
