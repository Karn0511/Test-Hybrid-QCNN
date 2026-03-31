# Dataset Pipeline

## Automated Ingestion
The dataset pipeline orchestrates downloading and normalizing multi-source data (e.g., Amazon, IMDB, Sentiment140, Twitter Airline). Raw CSVs and huggingface datasets are retrieved by `dataset_loader.py`. 

## Schema Unification
All subsets are converted into a standardized layout utilizing only generic labels:
- `text`: Utterance content 
- `label`: Consolidated to `positive`, `neutral`, and `negative`
- `source`: Dataset identifier for tracking
- `language`: Detected language ISO code

## Multilingual Streaming
Real-time ingestion is available via `/api/stream`. This creates a live Server-Sent Events (SSE) gateway simulating live social media streams directly connected to the FastAPI application, providing real-time data for Angular dashboard charting.
