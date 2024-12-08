# NYC & Philadelphia Surveillance Camera Analysis

An interactive visualization tool for exploring surveillance camera networks in New York City and Philadelphia. Built with Streamlit and PyDeck.

## Features

- Interactive 3D map visualization
- Camera density analysis using hexagonal binning
- Real-time filtering by region and camera type
- Performance-optimized for large datasets
- Comprehensive statistics and data export

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your Mapbox token:
```bash
export MAPBOX_TOKEN='your_mapbox_token_here'
```

3. Run the app:
```bash
streamlit run app.py
```

## Data Sources

- NYC: Amnesty International Decode Surveillance NYC project
- Philadelphia: Aggregated from SafeCam, Business Security Camera Program, and Real Time Crime Center

## Usage

1. Select a city from the sidebar
2. Choose between Points view (individual cameras) or Density view (concentration heatmap)
3. Use the filters to explore specific regions or camera types
4. Adjust the 3D view using navigation controls
5. Export data and statistics as needed

## Environment Variables

- `MAPBOX_TOKEN`: Required for map visualization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License 