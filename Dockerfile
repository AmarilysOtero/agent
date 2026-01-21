# Dockerfile for Agent application
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies and Azure CLI
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    lsb-release \
    gnupg \
    && curl -sL https://aka.ms/InstallAzureCLIDeb | bash \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the port
EXPOSE 8787

# Run the application
CMD ["python", "-m", "src.news_reporter.api"]
