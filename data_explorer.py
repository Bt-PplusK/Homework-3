if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter
import mlflow
import mlflow.sklearn

@data_exporter
def export_data(data, *args, **kwargs):
    mlflow.set_tracking_uri('http://mlflow:5000')
    with mlflow.start_run():
        mlflow.sklearn.log_model(lr, "model")
        mlflow.log_artifact(dv, "dict_vectorizer")
        
        print("Model Intercept: ", lr.intercept_)
        print("Model Size (bytes): ", os.path.getsize("model"))

    return lr.intercept_

