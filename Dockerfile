# ISIC 2024 Skin Lesion Classifier — Gradio Demo
#
# Build:
#   docker build -t isic2024-demo .
#
# Run:
#   docker run -p 7860:7860 isic2024-demo
#
# Then open http://localhost:7860

FROM python:3.11-slim

# System deps (none needed — app is pure Python + gradio)
WORKDIR /app

# Install Python dependencies first (cached layer)
COPY app/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app and precomputed gallery data
COPY app/ .

# Gradio needs this for non-interactive environments
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

EXPOSE 7860

CMD ["python", "app.py"]
