from semantic_router import Route, SemanticRouter

routes = [
    Route(
        name="code",
        description="Handles programming-related queries.",
        utterances=["code", "function", "bug", "debug"]
    ),
    Route(
        name="image",
        description="Handles image-related queries.",
        utterances=["photo", "picture", "image", "screenshot"]
    ),
]

semantic_router = SemanticRouter(routes=routes)  # ✅ Router Object

def combined_router(query):  # ✅ Function name changed
    selected_route = semantic_router.get_route(query)
    return selected_route.name

query = "Can you help me debug this function?"
route = combined_router(query)
print(f"Query routed to: {route}")