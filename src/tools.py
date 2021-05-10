from timeit import default_timer as timer

class ProgressBar():
    def __init__(self, task='task'):
        self.task = task
        self.started = False
        self.finished = False
    
    def reset(self):
        self.started = True
        self.finished = False
        self.prev_progress = 1e-8
        self.start = timer()

    def __call__(self, progress):
        if progress < 0.0 or progress > 1.0:
            raise ValueError('progress must be between 0 and 1')
        if not self.finished:
            if not self.started or progress < self.prev_progress:
                self.reset()
            self.prev_progress = progress
            msg = '|'
            msg += '█'*round(progress*30)
            msg += ' '*round((1-progress)*30)
            msg += f'| {progress:.1%} '
            msg += self.task
            if progress > 1e-4:
                t = timer()
                elapsed = t - self.start
                time_left = elapsed/progress-elapsed
                msg += f' Remaining: {time_left:8.1f} seconds'
            if round(progress*1000) == 1000:
                spaces = ' '*43
                elapsed = t - self.start
                print(f'\rFinished {self.task} in {elapsed:.1f} seconds {spaces}')
                self.finished = True
            else:
                print(f'\r{msg}', end='')
        elif progress < self.prev_progress:
                self.reset()

if __name__ == "__main__":
    a = []
    pb = ProgressBar('kacsa')
    for i in range(1000):
        pb(i/999)
        for x in range(50000):
            a.append(x)
            a[x] = 0
            b = a[x]
            a[x] = b
        a = []