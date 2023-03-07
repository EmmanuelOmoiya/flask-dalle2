FROM python:3.8.10
WORKDIR /app
COPY requirements.txt .
RUN apt update
RUN apt install -y libgl1-mesa-glx
RUN pip3 install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=5000
EXPOSE 5000

CMD ["gunicorn", "-b", "0.0.0.0:${PORT}", "app:app"]