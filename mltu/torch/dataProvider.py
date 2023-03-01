from ..dataProvider import DataProvider as dataProvider

class DataProvider(dataProvider):
    def __init__(self, *args, **kwargs):
        # Inherit with all the documentation of parent class
        super(DataProvider, self).__init__(*args, **kwargs)
