import telepot

from pytorch_lightning import Callback

class TelegramCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.max_macro_f1 = 0

    def on_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        this_macro_f1 = m['macro/macro_f1'].item()
        metrics.get(self.monitor)
        if self.max_macro_f1 < this_macro_f1:
            self.max_macro_f1 = this_macro_f1

            try:
                send('best_f1: {}'.format(round(this_macro_f1, 4)))
            except Exception as e:
                print("Unable to send:", e)

        # message  = f"Epoch: {m["epoch"]}\n"
        # message += f"Train Loss: {m["train_loss"]}
        # message += f"Accuracy:" {m["accuracy"]}
        


def send(msg, chat_id = 190649040, token='792681420:AAH_wiAsCO5Bk1kan_Iy3LTaJjDl3gWOZBU'):
  bot = telepot.Bot(token=token)
  bot.sendMessage(chat_id=chat_id, text=msg)