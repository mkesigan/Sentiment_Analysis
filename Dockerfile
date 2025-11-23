#BASE IMAGE(ligthweight python)
FROM python:3.10-slim

#set working directory inside container
WORKDIR /app

#copy project files
COPY requirements.txt ./requirements.txt

#install dependencies
RUN pip install --no-cache-dir --default-timeout=1000 torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

#copy project files
COPY src ./src
COPY api ./api
COPY models ./models

#expose port for fastAPI
EXPOSE 8000

#Start fastAPI server using uvicorn
CMD ["uvicorn","api.main:app","--host","0.0.0.0","--port","8000"]
