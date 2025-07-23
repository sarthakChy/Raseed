#!/usr/bin/env python3
"""
Standalone runner for the Financial Analysis Agent.
This script allows you to run and test the FinancialAnalysisAgent independently.
"""

import asyncio
import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Assuming your project structure - adjust imports as needed
from agents.financial_analysis_agent import FinancialAnalysisAgent


class FinancialAgentRunner:
    """Standalone runner for the Financial Analysis Agent."""
    
    def __init__(
        self, 
        project_id: str = "massive-incline-466204-t5",
        location: str = "us-central1",
        model_name: str = "gemini-2.0-flash-001"
    ):
        """Initialize the runner with configuration."""
        # Validate project_id is a string
        if not isinstance(project_id, str) or not project_id.strip():
            raise ValueError(f"Invalid project_id: {project_id}. Must be a non-empty string.")
        
        self.project_id = project_id.strip()
        self.location = location
        self.model_name = model_name
        self.agent = None
        self.logger = logging.getLogger(__name__)
        
        # Log configuration for debugging
        self.logger.info(f"Runner initialized with project_id: {self.project_id}")
        self.logger.info(f"Location: {self.location}, Model: {self.model_name}")
    
    async def initialize_agent(self, user_id: Optional[str] = None) -> bool:
        """Initialize the financial analysis agent."""
        try:
            self.logger.info("Initializing Financial Analysis Agent...")
            self.logger.info(f"Using project_id: {self.project_id}")
            
            # Validate Google Cloud setup before initializing
            if not await self._validate_gcp_setup():
                return False
            
            self.agent = FinancialAnalysisAgent(
                agent_name="standalone_financial_agent",
                project_id=self.project_id,
                location=self.location,
                model_name=self.model_name,
                user_id=user_id
            )
            await self.agent.db_connector.initialize()
            
            self.logger.info("Financial Analysis Agent initialized successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize agent: {e}")
            return False
    
    async def _validate_gcp_setup(self) -> bool:
        """Validate Google Cloud setup and permissions."""
        try:
            import google.auth
            from google.cloud import aiplatform
            
            # Check authentication
            credentials, auth_project = google.auth.default()
            self.logger.info(f"Authentication successful. Detected project: {auth_project}")
            
            if auth_project and auth_project != self.project_id:
                self.logger.warning(f"Auth project ({auth_project}) differs from configured project ({self.project_id})")
            
            # Initialize aiplatform with explicit project
            aiplatform.init(project=self.project_id, location=self.location)
            
            # Test basic access (this will fail gracefully if no access)
            try:
                # This is a lightweight check
                from google.cloud.aiplatform import Model
                # Just initialize, don't actually call API yet
                self.logger.info("Google Cloud AI Platform access validated")
                return True
            except Exception as api_error:
                self.logger.error(f"Google Cloud AI Platform access validation failed: {api_error}")
                return False
                
        except Exception as e:
            self.logger.error(f"Google Cloud setup validation failed: {e}")
            self.logger.error("Make sure you have:")
            self.logger.error("1. Valid Google Cloud credentials")
            self.logger.error("2. Vertex AI API enabled in your project")
            self.logger.error("3. Correct project ID")
            return False
    
    async def run_analysis(self, query: str, analysis_type: str = "general", user_id: Optional[str] = None) -> Dict[str, Any]:
        """Run a financial analysis with the given query."""
        if not self.agent:
            if not await self.initialize_agent(user_id):
                return {"error": "Failed to initialize agent"}
        
        try:
            self.logger.info(f"Running analysis: {query}")
            
            request = {
                "query": query,
                "analysis_type": analysis_type,
                "user_id": user_id,
                "context": {
                    "timestamp": datetime.now().isoformat(),
                    "standalone_mode": True
                }
            }
            
            result = await self.agent.process(request)
            return result
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return {"error": str(e)}
    
    async def get_financial_summary(self, user_id: str, time_period: str = "1_month") -> Dict[str, Any]:
        """Get a comprehensive financial summary."""
        if not self.agent:
            if not await self.initialize_agent(user_id):
                return {"error": "Failed to initialize agent"}
        
        try:
            self.logger.info(f"Generating financial summary for user {user_id}")
            result = await self.agent.get_financial_summary(user_id, time_period)
            return result
            
        except Exception as e:
            self.logger.error(f"Summary generation failed: {e}")
            return {"error": str(e)}
    
    def print_supported_analysis_types(self):
        """Print all supported analysis types."""
        if self.agent:
            types = self.agent.get_supported_analysis_types()
            print("\nSupported Analysis Types:")
            for i, analysis_type in enumerate(types, 1):
                print(f"{i:2d}. {analysis_type}")
        else:
            print("Agent not initialized. Please initialize first.")
    
    async def interactive_mode(self):
        """Run the agent in interactive mode."""
        print("=" * 60)
        print("Financial Analysis Agent - Interactive Mode")
        print("=" * 60)
        
        # Initialize agent
        user_id = input("Enter user ID (or press Enter for default): ").strip() or "test_user"
        
        if not await self.initialize_agent(user_id):
            print("Failed to initialize agent. Exiting.")
            return
        
        print(f"\nAgent initialized for user: {user_id}")
        self.print_supported_analysis_types()
        
        print("\nAvailable commands:")
        print("- 'analyze <query>' - Run financial analysis")
        print("- 'summary [time_period]' - Get financial summary (1_month, 3_months, 6_months, 1_year)")
        print("- 'types' - Show supported analysis types")
        print("- 'quit' or 'exit' - Exit interactive mode")
        print("- 'help' - Show this help message")
        
        while True:
            try:
                command = input("\n> ").strip()
                
                if command.lower() in ['quit', 'exit']:
                    print("Goodbye!")
                    break
                
                elif command.lower() == 'help':
                    print("\nAvailable commands:")
                    print("- 'analyze <query>' - Run financial analysis")
                    print("- 'summary [time_period]' - Get financial summary")
                    print("- 'types' - Show supported analysis types")
                    print("- 'quit' or 'exit' - Exit interactive mode")
                
                elif command.lower() == 'types':
                    self.print_supported_analysis_types()
                
                elif command.lower().startswith('analyze '):
                    query = command[8:].strip()
                    if query:
                        print(f"\nAnalyzing: {query}")
                        result = await self.run_analysis(query, user_id=user_id)
                        self._print_result(result)
                    else:
                        print("Please provide a query to analyze.")
                
                elif command.lower().startswith('summary'):
                    parts = command.split()
                    time_period = parts[1] if len(parts) > 1 else "1_month"
                    
                    print(f"\nGenerating summary for period: {time_period}")
                    result = await self.get_financial_summary(user_id, time_period)
                    self._print_result(result)
                
                elif command.strip() == "":
                    continue
                
                else:
                    print("Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def _print_result(self, result: Dict[str, Any]):
        """Print analysis result in a formatted way."""
        if result.get("success"):
            print("\n✅ Analysis completed successfully!")
            if "analysis" in result:
                print(f"\nAnalysis:\n{result['analysis']}")
            if "execution_time" in result:
                print(f"\nExecution time: {result['execution_time']:.2f}s")
        else:
            print(f"\n❌ Analysis failed: {result.get('error', 'Unknown error')}")
        
        # Print additional details if in debug mode
        if os.getenv("DEBUG", "").lower() == "true":
            print(f"\nFull result:\n{json.dumps(result, indent=2)}")


async def main():
    """Main entry point for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Financial Analysis Agent standalone")
    parser.add_argument("--query", "-q", type=str, help="Query to analyze")
    parser.add_argument("--user-id", "-u", type=str, default="test_user", help="User ID")
    parser.add_argument("--analysis-type", "-t", type=str, default="general", help="Analysis type")
    parser.add_argument("--summary", "-s", action="store_true", help="Generate financial summary")
    parser.add_argument("--time-period", "-p", type=str, default="1_month", help="Time period for summary")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--project-id", type=str, default="massive-incline-466204-t5", help="Google Cloud project ID")
    parser.add_argument("--location", type=str, default="us-central1", help="Vertex AI location")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash-001", help="Model name")
    parser.add_argument("--validate-only", action="store_true", help="Only validate setup, don't run analysis")
    
    args = parser.parse_args()
    
    try:
        # Create runner with validated inputs
        runner = FinancialAgentRunner(
            project_id=args.project_id,
            location=args.location,
            model_name=args.model
        )
        
        if args.validate_only:
            # Just validate setup
            if await runner._validate_gcp_setup():
                print("✅ Google Cloud setup validation passed!")
            else:
                print("❌ Google Cloud setup validation failed!")
            return
        
        if args.interactive:
            # Run in interactive mode
            await runner.interactive_mode()
        
        elif args.summary:
            # Generate summary
            result = await runner.get_financial_summary(args.user_id, args.time_period)
            runner._print_result(result)
        
        elif args.query:
            # Run single analysis
            result = await runner.run_analysis(args.query, args.analysis_type, args.user_id)
            runner._print_result(result)
        
        else:
            # Default to interactive mode if no specific action
            print("No specific action provided. Starting interactive mode...")
            await runner.interactive_mode()
            
    except Exception as e:
        logging.error(f"Failed to run financial agent: {e}")
        print(f"❌ Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check your Google Cloud credentials")
        print("2. Verify your project ID is correct")
        print("3. Ensure Vertex AI API is enabled")
        print("4. Run with --validate-only to test setup")


if __name__ == "__main__":
    # Set up environment (adjust as needed for your setup)
    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", r"C:\Users\Jaidivya Kumar Lohan\Desktop\Raseed\server\serviceAccountKey.json")
    
    # Run the main function
    asyncio.run(main())