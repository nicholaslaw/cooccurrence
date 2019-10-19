FROM python:3.6.5-jessie

COPY docker_requirements.txt /
RUN pip install -r /docker_requirements.txt
CMD ["jupyter", "notebook", "--NotebookApp.token='password'", "--ip=0.0.0.0", "--allow-root"]