import json
from server import PromptServer
from aiohttp import web

class DynamicTypeNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "operation": (["STRING", "INT", "FLOAT", "JSON", "IMAGE"], ),
                "input_value": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("*",)
    FUNCTION = "process"
    CATEGORY = "utils"
    OUTPUT_NODE = True

    def process(self, operation, input_value):
        if operation == "STRING":
            return (input_value,)
        elif operation == "INT":
            try:
                return (int(input_value),)
            except ValueError:
                return ("Error: Cannot convert to INT",)
        elif operation == "FLOAT":
            try:
                return (float(input_value),)
            except ValueError:
                return ("Error: Cannot convert to FLOAT",)
        elif operation == "JSON":
            try:
                return (json.loads(input_value),)
            except json.JSONDecodeError:
                return ("Error: Invalid JSON",)
        elif operation == "IMAGE":
            try:
                

    @classmethod
    def IS_CHANGED(cls, operation, input_value):
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, operation, input_value):
        if operation not in ["STRING", "INT", "FLOAT", "JSON"]:
            return "Invalid operation"
        return True

# Custom route to get the actual return type
@PromptServer.instance.routes.get("/get_dynamic_type")
async def get_dynamic_type(request):
    operation = request.query.get("operation", "STRING")
    input_value = request.query.get("input_value", "")

    node = DynamicTypeNode()
    result = node.process(operation, input_value)

    if isinstance(result[0], str) and result[0].startswith("Error:"):
        return web.json_response({"type": "STRING", "error": result[0]})

    if operation == "STRING":
        return web.json_response({"type": "STRING"})
    elif operation == "INT":
        return web.json_response({"type": "INT"})
    elif operation == "FLOAT":
        return web.json_response({"type": "FLOAT"})
    elif operation == "JSON":
        return web.json_response({"type": "JSON"})

    return web.json_response({"type": "UNKNOWN"})

