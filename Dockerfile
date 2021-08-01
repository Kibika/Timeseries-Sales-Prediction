FROM continuumio/miniconda:4.5.4

RUN pip install mlflow>=1.0 \
    && pip install azure-storage-blob==12.3.0 \
    && pip install numpy \
    && pip install scipy \
    && pip install pandas==0.22.0 \
    && pip install scikit-learn==0.19.1 \
    && pip install cloudpickle \
    && pip install dvc \
    && pip install seaborn \
    && pip install statsmodels \
    && pip install xgboost
