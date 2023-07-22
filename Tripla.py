
class Tripla:
    def __init__(self, asset0, asset1, asset2, cash):
        self.asset0 = asset0
        self.asset1 = asset1
        self.asset2 = asset2
        self.cash = cash

    def __getitem__(self, item):
        if item == 0:
            return self.asset0
        if item == 1:
            return self.asset1
        if item == 2:
            return self.asset2
        if item == 3:
            return self.cash

    def __str__(self):
        return f'{self.asset0}, {self.asset1}, {self.asset2}, {self.cash}'