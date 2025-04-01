from flask import Flask, request, jsonify
from flask_cors import CORS
import asyncio
import json
import traceback

from ExecutionPlanGenerator import ExecutionPlanGenerator
from ExecutionPlan import ExecutionPlan
from dotenv import load_dotenv
from Logger import logger, log_request, log_error

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/generate-plan", methods=["POST"])
def generate_plan():
    try:
        # Get data from request body
        data = request.get_json()
        
        # Log the incoming request
        log_request(logger, data)
        
        # Validate required fields
        required_fields = [
            "user_doc_format",
            "user_question"
        ]
        for field in required_fields:
            if not data.get(field):
                error_msg = f"Missing required field: {field}"
                log_error(logger, error_msg)
                return jsonify({"error": error_msg}), 400

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

        # Log successful response
        log_request(logger, {"status": "success", "plan": plan_string})
        return jsonify({"plan": plan_string, "status": "success"})

    except Exception as e:
        # Log the error with full traceback
        error_msg = f"Error occurred: {str(e)}"
        log_error(logger, error_msg, traceback.format_exc())
        return jsonify({"error": str(e), "status": "error"}), 500


if __name__ == "__main__":
    app.run(debug=True)
