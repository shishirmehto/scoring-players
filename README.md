# FPL Top 5 All Positions Next GW

A Python tool for analyzing and ranking Fantasy Premier League (FPL) players by position for the upcoming gameweek. This tool provides intelligent scoring based on multiple factors and generates both console output and CSV files for further analysis.

## Features

- **Position-specific scoring algorithms** for Goalkeepers, Defenders, Midfielders, and Forwards
- **Multi-factor scoring** incorporating:
  - Expected points for next gameweek (`ep_next`)
  - Current form
  - ICT Index (Influence, Creativity, Threat)
  - Selection percentage
  - Availability status and injury concerns
  - Position-specific bonuses (e.g., saves for goalkeepers)
- **Top 5 rankings** for each position with detailed statistics
- **CSV export** for all ranked players in each position
- **Flexible data source** - can use live FPL API or local JSON files

## Requirements

- Python 3.6+
- `requests` library (optional - for live API calls)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd scoring-players
```

2. Install dependencies (optional, for live API usage):
```bash
pip install requests
```

## Usage

### Basic Usage (Live FPL API)
```bash
python fpl_top5_all_positions_nextgw.py
```

### Using Local JSON Data
```bash
python fpl_top5_all_positions_nextgw.py path/to/bootstrap-static.json
```

## Output

The tool provides:

1. **Console Output**: Top 5 players for each position with key statistics
2. **CSV Files**: Complete rankings for each position:
   - `fpl_ranked_goalkeepers.csv`
   - `fpl_ranked_defenders.csv`
   - `fpl_ranked_midfielders.csv`
   - `fpl_ranked_forwards.csv`

## Scoring Algorithm

### Goalkeepers
- 62% Expected Points (ep_next)
- 20% Form
- 8% ICT Index
- 5% Selection Percentage
- 5% Saves per 90 minutes

### Defenders
- 60% Expected Points (ep_next)
- 18% Form
- 18% ICT Index
- 4% Selection Percentage

### Midfielders
- 60% Expected Points (ep_next)
- 22% Form
- 15% ICT Index
- 3% Selection Percentage

### Forwards
- 65% Expected Points (ep_next)
- 20% Form
- 12% ICT Index
- 3% Selection Percentage

### Availability Penalties
- Injured/Suspended: +1.0 penalty
- Doubtful: +0.3 penalty
- <75% chance of playing: +0.4 penalty
- <100% chance of playing: +0.1 penalty

## Data Sources

The tool can work with:
- **Live FPL API**: `https://fantasy.premierleague.com/api/bootstrap-static/`
- **Local JSON files**: Previously downloaded bootstrap-static data

## Technical Details

- Built with pure Python (no external dependencies required for local usage)
- Handles missing data gracefully with safe conversion functions
- Includes proper error handling for network requests
- Generates clean, formatted CSV output with UTF-8 encoding

## Use Cases

- **FPL Team Selection**: Identify the best players in each position for upcoming gameweeks
- **Transfer Planning**: Compare players across positions and teams
- **Data Analysis**: Export data for further statistical analysis
- **Research**: Understand which factors contribute most to player performance

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the scoring algorithms or add new features.

## License

This project is open source and available under the MIT License.
