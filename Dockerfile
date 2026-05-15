# Use a lightweight Python base image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Install system dependencies required by Matplotlib and numerical libraries
# libpng-dev and libfreetype6-dev are critical for chart rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpng-dev \
    libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# matplotlib is required for equity curve chart generation
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    mcp[sse] \
    pandas \
    duckdb \
    boto3 \
    quantstats \
    scipy \
    matplotlib \
    numpy

# Copy application code
COPY trading_server.py .

# Set Matplotlib to use the non-interactive Agg backend (no display required)
ENV QT_QPA_PLATFORM=offscreen
ENV PORT=8080

# Start the server
CMD ["python", "trading_server.py"]