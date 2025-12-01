FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (for some Python packages)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        build-essential \
        g++ \
        gcc \
        python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker build caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code and data
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# 🔴 IMPORTANT:
# - baseUrlPath must match your slug
# - main file is src/ui/app.py
CMD ["streamlit", "run", "src/ui/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.baseUrlPath=/tech-earnings-rag"]
