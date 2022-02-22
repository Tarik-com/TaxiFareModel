# imports
import mlflow
from asyncio import LifoQueue
from sklearn.pipeline import make_pipeline
from TaxiFareModel.encoders import TimeFeaturesEncoder,DistanceTransformer
from TaxiFareModel.utils import compute_rmse
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from TaxiFareModel.data import get_data,clean_data
import memoized_property

MLFLOW_URI = "https://mlflow.lewagon.co/"
EXPERIMENT_NAME = "[BEL] [Brussels] [Tarik-com] my model  v1"
class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.experiment_name=EXPERIMENT_NAME

        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        self.pipeline=make_pipeline(
            ColumnTransformer([
                ('distance', make_pipeline(
                    DistanceTransformer(),StandardScaler()), ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
                ('time', make_pipeline(
                    TimeFeaturesEncoder('pickup_datetime'),OneHotEncoder(handle_unknown="ignore")), ['pickup_datetime'])], remainder="drop"),
            LinearRegression())
        return self.pipeline

    def run(self):
        """set and train the pipeline"""
        self.pipeline=self.set_pipeline()
        self.pipeline=self.pipeline.fit(self.X,self.y)


    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        self.score= compute_rmse(y_pred, y_test)

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)




if __name__ == "__main__":
    #get data
    df=get_data()
    # clean data
    print(clean_data(df))
    # set X and y
    y=df['fare_amount']
    X=df.drop(columns="fare_amount")
    # hold out



    trainer=Trainer(X_train,y_train)

    # train
    trainer.run()
    # evaluate
    trainer.evaluate()
    print("TODO")
