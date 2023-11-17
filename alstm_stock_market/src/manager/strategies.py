class Martingale:
    """Double the bet after each loss, return to initial bet after each win."""

    def __init__(self, initial_bet):
        self.name = "martingale"
        self.initial_bet = initial_bet
        self.current_bet = initial_bet

    def next_bet(self, win, current_cash):
        self.current_bet = self.initial_bet if win else self.current_bet * 2
        return min(self.current_bet, current_cash)


class Paroli:  # Reverse Martingale
    """Double the bet after each win, return to initial bet after each loss."""

    def __init__(self, initial_bet):
        self.name = "paroli"
        self.initial_bet = initial_bet
        self.current_bet = initial_bet

    def next_bet(self, win, current_cash):
        self.current_bet = self.initial_bet if not win else self.current_bet * 2
        return min(self.current_bet, current_cash)


class DAlembert:
    """Increase bet by initial bet after each loss, decrease by initial bet after each win."""

    def __init__(self, initial_bet):
        self.name = "dalembert"
        self.initial_bet = initial_bet
        self.current_bet = initial_bet

    def next_bet(self, win, current_cash):
        self.current_bet = (
            max(self.initial_bet, self.current_bet - self.initial_bet)
            if win
            else self.current_bet + self.initial_bet
        )
        return min(self.current_bet, current_cash)


class Fixed:
    """Always the same bet."""

    def __init__(self, initial_bet):
        self.name = "fixed"
        self.initial_bet = initial_bet

    def next_bet(self, win, current_cash):
        return min(self.initial_bet, current_cash)


class Proportional:
    """Bet a proportion of the current cash."""

    def __init__(self, initial_cash, proportion):
        self.name = f"proportional_{proportion}"
        self.cash = initial_cash
        self.proportion = proportion
        self.initial_bet = initial_cash * proportion
        self.current_bet = self.initial_bet

    def next_bet(self, win, current_cash):
        self.cash = current_cash
        self.current_bet = self.cash * self.proportion
        return min(self.current_bet, current_cash)
