from flask import Flask, request, jsonify
from flask_cors import CORS
import asyncio
import json

from ExecutionPlanGenerator import ExecutionPlanGenerator
from ExecutionPlan import ExecutionPlan
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/generate-plan", methods=["GET"])
def generate_plan():
    try:
        # Get data from query parameters
        data = {
            "user_doc_format": request.args.get("user_doc_format"),
            "user_question": request.args.get("user_question"),
            "user_requirements": json.loads(request.args.get("user_requirements", "[]")),
            "additional_context": request.args.get("additional_context", ""),
            "max_depth": request.args.get("max_depth", "1")
        }

        # Validate required fields
        required_fields = [
            "user_doc_format",
            "user_question"
        ]
        for field in required_fields:
            if not data[field]:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Create generator and process
        generator = ExecutionPlanGenerator(
            data["user_doc_format"],
            data["user_question"],
            data["user_requirements"],
            data["additional_context"],
            max_depth=int(data["max_depth"]),
        )

        # Run the async process
        result = asyncio.run(generator.process())

        plan = ExecutionPlan(result)
        plan_string = ExecutionPlan.plan_to_string(plan.plan)

        return jsonify({"plan": plan_string, "status": "success"})

    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500


if __name__ == "__main__":
    app.run(debug=True)
