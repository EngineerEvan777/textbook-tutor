FROM python:3.11-slim

WORKDIR /app

# Install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY . /app

# Cloud hosts provide PORT; default to 10000 locally
ENV PORT=10000

EXPOSE 10000

CMD ["sh", "-c", "python -m uvicorn app:app --host 0.0.0.0 --port ${PORT}"]