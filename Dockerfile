FROM python:3.10

WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y libgl1-mesa-glx xorg

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

CMD ["python", "src/core.py"]
