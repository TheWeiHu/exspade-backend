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

@app.route("/generate-plan", methods=["POST"])
def generate_plan():
    try:
        # Get data from request body
        data = request.get_json()
        
        # Validate required fields
        required_fields = [
            "user_doc_format",
            "user_question"
        ]
        for field in required_fields:
            if not data.get(field):
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Create generator and process
        generator = ExecutionPlanGenerator(
            data.get("user_doc_format"),
            data.get("user_question"),
            data.get("user_requirements", []),
            data.get("additional_context", ""),
            max_depth=int(data.get("max_depth", 1)),
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
