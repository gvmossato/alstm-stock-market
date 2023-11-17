from alstm_stock_market.src.helpers.utils import save_txt


class Manager:
    def __init__(self, initial_cash, initial_bet, trend_predictions, market_returns):
        self.trend_preds = trend_predictions
        self.returns = market_returns

        self.bankroll = [initial_cash]
        self.bets = [initial_bet]
        self.gains_losses = []

    def run(self, strategy):
        for pred, pct_change in zip(self.trend_preds, self.returns):
            # Long: win when return > 0, lose otherwise; Short: win when return < 0, lose otherwise
            pct_gain_loss = pct_change if pred else -pct_change

            self.gains_losses.append(self.bets[-1] * pct_gain_loss)
            self.bankroll.append(max(self.bankroll[-1] + self.gains_losses[-1], 0))

            if self.bankroll[-1] == 0:
                break

            self.bets.append(
                strategy.next_bet(
                    win=self.gains_losses[-1] >= 0,
                    current_cash=self.bankroll[-1],
                )
            )

        results = {
            "bankroll": self.bankroll,
            "bets": self.bets,
            "gains_losses": self.gains_losses,
        }
        save_txt(results, strategy.name)
        return results
