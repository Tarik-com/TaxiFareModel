# imports
from asyncio import LifoQueue
from sklearn.pipeline import make_pipeline
from TaxiFareModel.encoders import TimeFeaturesEncoder,DistanceTransformer
from TaxiFareModel.utils import compute_rmse
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from TaxiFareModel.data import get_data,clean_data

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
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




if __name__ == "__main__":
    # get data
    #df=get_data()
    # clean data
    #print(clean_data(df))
    # set X and y
    #y=df['fare_amount']
    #X=df.drop(columns="fare_amount")
    # hold out

    # train
    #Trainer.run()
    # evaluate
    #Trainer.evaluate()
    print("TODO")
