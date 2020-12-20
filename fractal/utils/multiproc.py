import concurrent.futures


class Parallelize:
    def __init__(self, _func, _serialize=False):
        self.func = _func
        self.serialize = _serialize

    def __call__(self, *args):
        if self.serialize:
            results = list(map(self.func, *args))
        else:
            # multiprocessor mode
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = list(executor.map(self.func, *args))

        return results


    