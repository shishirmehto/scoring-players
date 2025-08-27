#!/bin/bash
# Script to activate the virtual environment for the FPL tools

echo "Activating virtual environment..."
source venv/bin/activate

echo "Virtual environment activated!"
echo "You can now run:"
echo "  python fpl_team_builder_with_vaastav.py --help"
echo "  python fpl_top5_all_positions_nextgw.py"
echo ""
echo "To deactivate, run: deactivate"
