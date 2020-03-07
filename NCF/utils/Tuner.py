import abc, itertools


class BaseTuner:
    def __init__(self, config, meta):
        self.config = config
        self.candidates = self.get_candidates()
        self.user_num = meta['user_num']
        self.item_num = meta['item_num']
        self.epochs = config['epochs']

    def get_hp(self):
        for hps in itertools.product(*self.candidates):
            yield self.get_combination(hps)

    @abc.abstractmethod
    def get_candidates(self):
        pass

    @abc.abstractmethod
    def get_combination(self, hps):
        pass


class GMFTuner(BaseTuner):
    def get_candidates(self):
        config = self.config
        candidates = [config['lr'], config['embed_dim'], config['neg_num'], config['batch_size']]

        return candidates

    def get_combination(self, hps):
        lr, embed_dim, num_neg, batch_size = hps
        hp = {
            'model': {
                'user_num': self.user_num,
                'item_num': self.item_num,
                'embed_dim': embed_dim
            },
            'etc': {
                'lr': lr,
                'num_neg': num_neg,
                'batch_size': batch_size,
                'epochs': self.epochs
            }
        }

        return hp


class MLPTuner(BaseTuner):
    pass


class NeuMFTuner(BaseTuner):
    pass