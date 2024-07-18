FROM python:3.9-slim
WORKDIR /app
COPY . .

RUN pip install --no-cache-dir hanlp
RUN pip install --no-cache-dir hanlp[full]
RUN pip install --no-cache-dir fastapi

EXPOSE 8000

CMD ["uvicorn", "main:app","--host", "0.0.0.0", "--port", "8000"]

