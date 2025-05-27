from tests_new.utils.test_base import TestBase

class BaseAITest(TestBase):
    @classmethod
    def get_test_data(cls):
        data = super().get_test_data()
        data['features'] = np.random.random(len(data))
        data['target'] = np.random.randint(0, 2, len(data))
        return data
