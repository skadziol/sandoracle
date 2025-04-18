#!/bin/bash

# Script to run SandoSeer in simulation mode with reduced logs
# This temporarily overrides the log level without modifying your .env file

# Set temporary environment variables for quieter logging
export RUST_LOG=warn
export FILE_LOG_LEVEL=info
export DETAILED_LOGGING=false
export SIMULATION_MODE=true  # Ensure simulation mode is enabled

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting SandoSeer in simulation mode with reduced logs..."
echo "When you want to stop the simulation, press Ctrl+C"
echo "An opportunity summary will be shown at the end"

# Add trap to show summary when the script is interrupted
trap show_summary INT

# Function to display summary of opportunities
show_summary() {
    echo ""
    echo "===== SIMULATION SUMMARY ====="
    
    # Check if opportunity stats file exists
    if [ -f "logs/opportunity_stats.json" ]; then
        echo "Opportunity Statistics:"
        # Use jq to format the JSON if available, otherwise use cat
        if command -v jq &> /dev/null; then
            jq . logs/opportunity_stats.json
        else
            cat logs/opportunity_stats.json
        fi
    else
        echo "No opportunity statistics found."
    fi
    
    # Check if detailed opportunities log exists
    if [ -f "logs/mev_opportunities.json" ]; then
        echo ""
        echo "Detailed Opportunities:"
        echo "Found in logs/mev_opportunities.json"
        
        # Count lines in the file to get number of opportunities
        OPPORTUNITY_COUNT=$(wc -l < logs/mev_opportunities.json)
        echo "Total opportunities found: $OPPORTUNITY_COUNT"
        
        # Show the last 3 opportunities if they exist
        if [ "$OPPORTUNITY_COUNT" -gt 0 ]; then
            echo ""
            echo "Last opportunities (up to 3):"
            if command -v jq &> /dev/null; then
                tail -n 3 logs/mev_opportunities.json | jq .
            else
                tail -n 3 logs/mev_opportunities.json
            fi
        fi
    else
        echo "No opportunities were found during this simulation run."
    fi
    
    echo "=============================="
    exit 0
}

# Run sandoseer with reduced output
cargo run --bin sandoseer

# This should not be reached normally as the script is terminated by Ctrl+C
# But just in case, show the summary here too
show_summary 