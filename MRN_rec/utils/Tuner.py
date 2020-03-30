import abc, itertools


class BaseTuner:
    def __init__(self, config, meta):
        self.config = config
        self.candidates = self.get_candidates()
        self.user_num = meta['user_num']
        self.item_num = meta['item_num']
        self.behavior_num = meta['behavior_num']
        self.device = config['device']
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


class MRNRecTuner(BaseTuner):
    def get_candidates(self):
        attrs = ['lr', 'embed_size', 'hidden_size', 'mrn_in_size', 'fcl_size', 'num_neg', 'batch_size']
        candidates = (self.config[k] for k in attrs)

        return candidates

    def get_combination(self, hps):
        lr, embed_size, hidden_size, mrn_in_size, fcl_size, num_neg, batch_size = hps
        hp = {
            'model': {
                'user_num': self.user_num,
                'item_num': self.item_num,
                'behavior_num': self.behavior_num,
                'device': self.device,
                'embed_size': embed_size,
                'hidden_size': hidden_size,
                'mrn_in_size': mrn_in_size,
                'fcl_size': fcl_size,

            },
            'etc': {
                'epochs': self.epochs,
                'lr': lr,
                'num_neg': num_neg,
                'batch_size': batch_size,
            }
        }

        return hp


class MLPTuner(BaseTuner):
    def get_candidates(self):
        config = self.config
        candidates = (config['lr'], config['embed_dim'], config['layer_num'], config['neg_num'], config['batch_size'])

        return candidates

    def get_combination(self, hps):
        lr, embed_dim, layer_num, num_neg, batch_size = hps
        hp = {
            'model': {
                'user_num': self.user_num,
                'item_num': self.item_num,
                'embed_dim': embed_dim,
                'layer_num': layer_num
            },
            'etc': {
                'lr': lr,
                'num_neg': num_neg,
                'batch_size': batch_size,
                'epochs': self.epochs
            }
        }

        return hp


class NeuMFTuner(BaseTuner):
    pass
