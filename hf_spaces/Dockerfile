FROM python:3.10-slim

WORKDIR /app

RUN useradd -m -u 1000 appuser

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rag_ui.py .
COPY src/ src/

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

CMD ["streamlit", "run", "rag_ui.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
