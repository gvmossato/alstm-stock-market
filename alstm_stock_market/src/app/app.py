import os
from datetime import datetime, timedelta

import dotenv
import numpy as np
import yfinance as yf
from bizdays import Calendar
from cloudant.client import Cloudant

import alstm_stock_market.src.model.params as p
from alstm_stock_market.src.data.preprocessor import Preprocessor
from alstm_stock_market.src.helpers.utils import log_app, reverse_normalize
from alstm_stock_market.src.model.model import Model

dotenv.load_dotenv()


class App:
    def __init__(self, pred_date=None):
        self.pred_date = (
            pred_date
            if pred_date is not None
            else (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
        )
        self.calendar = Calendar.load(filename=os.environ["CALENDAR"])

        if not self.calendar.isbizday(self.pred_date):
            result = {
                "status": "error",
                "message": f"{self.pred_date} is not a business day.",
            }
            log_app(result)
            raise ValueError(result)

        self.interval_start = self.calendar.offset(
            self.pred_date,
            -(p.time_step+1),
        ).strftime("%Y-%m-%d")

        self.days_since_training = self.calendar.diff(
            [
                os.environ["MAX_TRAINING_DATE"],
                self.pred_date,
            ]
        )[0]

    def _incremental_fit(self):
        train_data = yf.download(
            p.ticker,
            start=os.environ["MAX_TRAINING_DATE"],
            end=self.pred_date,
        )

        pre = Preprocessor(train_data, {"train": 1, "valdn": 0, "test": 0})
        pre.run()

        model = Model(load_weights=True)
        model.incremental_train(pre.X_train, pre.y_train)

        os.environ["MAX_TRAINING_DATE"] = pre.dates[-1].strftime("%Y-%m-%d")
        dotenv.set_key(
            os.environ["ENV_FILE_PATH"],
            "MAX_TRAINING_DATE",
            os.environ["MAX_TRAINING_DATE"],
        )

    def _make_prediction(self):
        pred_data = yf.download(
            p.ticker,
            start=self.interval_start,
            end=self.pred_date,
        ).tail(p.time_step)

        if len(pred_data) < p.time_step:
            result = {
                "status": "error",
                "message": f"Expected {p.time_step} days for {self.pred_date}, got {len(pred_data)}",
            }
            log_app(result)
            raise ValueError(result)

        self.close_prices = pred_data["Close"]

        pre = Preprocessor(pred_data, {"train": 0, "valdn": 0, "test": 1})
        pre.run()

        model = Model(load_weights=True)
        y_pred = model.predict(pre.X_test)
        self.pred_close = np.round(
            reverse_normalize(
                y_pred[0],
                pre.target_norm_mean,
                pre.target_norm_std,
            ),
            2,
        )

    def _connect(self):
        self.client = Cloudant(
            os.environ["CLOUDANT_USERNAME"],
            os.environ["CLOUDANT_API_KEY"],
            url=os.environ["CLOUDANT_URL"],
            connect=True,
        )
        self.database = self.client.create_database(
            os.environ["DATABASE"],
            throw_on_exists=False,
        )
        result = self.database.get_view_result(
            "_design/sp500View",
            "by-date",
            startkey=self.interval_start,
            endkey=self.pred_date,
            include_docs=True,
        )
        self.interval_docs = [d["doc"] for d in result]

    def _write_prediction(self):
        for doc in self.interval_docs:
            if doc["date"] == self.pred_date:
                if np.isclose(doc["pred_close"], self.pred_close):
                    result = {
                        "status": "success",
                        "message": f"Prediction for {self.pred_date} already exists, no update needed",
                        "data": doc,
                    }
                    log_app(result)
                    return

                doc["pred_close"] = self.pred_close
                self.database[doc["_id"]].update(doc)
                self.database[doc["_id"]].save()

                result = {
                    "status": "success",
                    "message": f"Updated prediction for {self.pred_date}",
                    "data": doc,
                }
                log_app(result)
                return

        doc = self.database.create_document(
            {
                "date": self.pred_date,
                "pred_close": self.pred_close,
            }
        )
        result = {
            "status": "success",
            "message": f"Created prediction for {self.pred_date}",
            "data": doc,
        }
        log_app(result)

    def _write_close_prices(self):
        for date, price in self.close_prices.items():
            for doc in self.interval_docs:
                if doc["date"] == date.strftime("%Y-%m-%d"):
                    self.interval_docs.remove(doc)

                    if ("close" not in doc) or (not np.isclose(doc["close"], price)):
                        doc["close"] = price
                        self.database[doc["_id"]].update(doc)
                        self.database[doc["_id"]].save()
                        result = {
                            "status": "success",
                            "message": f"Updated close for {date}",
                            "data": doc,
                        }
                        log_app(result)
                        break

                    result = {
                        "status": "success",
                        "message": f"Close for {date} already exists, no update needed",
                        "data": doc,
                    }
                    break

    def run(self):
        if self.days_since_training >= p.batch_size:
            self._incremental_fit()

        self._make_prediction()

        try:
            self._connect()
            self._write_prediction()
            self._write_close_prices()
        except Exception as e:
            print(e)
        finally:
            self.client.disconnect()


def main():
    App().run()


if __name__ == "__main__":
    main()
