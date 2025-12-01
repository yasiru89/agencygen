"""
A2A (Agent-to-Agent) Communication Example
==========================================

This demonstrates how to expose AgencyGen-created agents as network
services using the A2A protocol, allowing other agents to communicate
with them across systems.

To run this example:
    1. Start the server: python example_a2a.py server
    2. In another terminal: python example_a2a.py client

Requirements:
    pip install google-adk uvicorn
"""

import sys
import os

# Load API key
if os.path.exists(".env"):
    with open(".env", "r") as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                os.environ[key] = value

from agency_gen import (
    create_single_agent,
    create_a2a_app,
    connect_to_remote_agent,
)


# =============================================================================
# SERVER: Expose an agent via A2A
# =============================================================================

def run_server(port: int = 8000):
    """
    Run an agent as an A2A server.
    
    This exposes the agent over HTTP so other agents can communicate with it.
    """
    print(f"\n🚀 Starting A2A Server on port {port}...")
    
    # Create an agent to expose
    product_agent = create_single_agent(
        name="product_catalog",
        instruction="""You are a product catalog assistant.
        
You have information about these products:
- Laptop Pro X: $1299, 16GB RAM, 512GB SSD
- Wireless Mouse: $49, Ergonomic design
- USB-C Hub: $79, 7 ports

When asked about products, provide accurate information.
If asked about a product you don't know, say so."""
    )
    
    print(f"   Created agent: {product_agent.name}")
    
    # Create the A2A app
    try:
        app = create_a2a_app(product_agent, app_name="product_catalog_a2a")
        print(f"   A2A app created!")
        print(f"\n📡 Agent is now accessible at: http://localhost:{port}")
        print(f"   Other agents can connect using:")
        print(f'   >>> remote = connect_to_remote_agent("http://localhost:{port}")')
        print(f"\n   Press Ctrl+C to stop the server.\n")
        
        # Run with uvicorn
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=port)
        
    except ImportError as e:
        print(f"\n❌ Error: {e}")
        print("\nTo use A2A, you may need additional dependencies:")
        print("  pip install uvicorn fastapi")


# =============================================================================
# CLIENT: Connect to a remote agent
# =============================================================================

async def run_client(server_url: str = "http://localhost:8000"):
    """
    Connect to a remote A2A agent and query it.
    """
    print(f"\n🔗 Connecting to remote agent at {server_url}...")
    
    try:
        # Connect to the remote agent
        remote_agent = connect_to_remote_agent(server_url)
        print(f"   Connected!")
        
        # Query the remote agent
        queries = [
            "What laptops do you have?",
            "How much is the wireless mouse?",
            "Do you have any keyboards?",
        ]
        
        for query in queries:
            print(f"\n📤 Query: {query}")
            # Note: The exact API depends on ADK version
            # This is a simplified example
            response = await remote_agent.run(query)
            print(f"📥 Response: {response}")
            
    except ImportError as e:
        print(f"\n❌ Error: {e}")
    except Exception as e:
        print(f"\n❌ Connection error: {e}")
        print("   Make sure the server is running!")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("  A2A (Agent-to-Agent) Communication Example")
    print("="*60)
    
    if len(sys.argv) < 2:
        print("""
Usage:
    python example_a2a.py server    # Start the A2A server
    python example_a2a.py client    # Connect as a client

The A2A protocol allows agents to communicate over networks,
enabling multi-agent systems across different machines/services.
""")
        return
    
    mode = sys.argv[1].lower()
    
    if mode == "server":
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
        run_server(port)
    elif mode == "client":
        url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"
        import asyncio
        asyncio.run(run_client(url))
    else:
        print(f"Unknown mode: {mode}")
        print("Use 'server' or 'client'")


if __name__ == "__main__":
    main()

