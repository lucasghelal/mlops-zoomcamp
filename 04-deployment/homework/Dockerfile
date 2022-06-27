FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

WORKDIR /app
## COPY [ "model2.bin", "model.bin" ]

COPY ["Pipfile", "Pipfile.lock", "starter.py", "./"]

RUN pip install pipenv
RUN pipenv install --system --deploy

CMD python starter.py 2021 04