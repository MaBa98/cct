import win32com.client
import pandas as pd
import datetime

class BloombergAPI:
    def __init__(self):
        self.session = win32com.client.Dispatch("Bloomberg.Data.1")

    def get_historical_data(self, tickers, fields, start_date, end_date):
        if isinstance(tickers, str):
            tickers = [tickers]
        if isinstance(fields, str):
            fields = [fields]

        all_data = {}

        for ticker in tickers:
            data = self.session.BDH(
                ticker,
                fields,
                start_date.strftime("%Y%m%d"),
                end_date.strftime("%Y%m%d")
            )
            df = pd.DataFrame(data)
            df.index = pd.to_datetime(df.index)
            df.columns = fields
            all_data[ticker] = df

        return all_data
