import time
import datetime
from modules import kernel_infer, neural_infer, tree_infer, load_data

if __name__ == "__main__":
    begin = time.perf_counter()
    x_anon, y_anon = load_data(train=False)
    kernel_infer(x_anon)
    neural_infer(x_anon)
    tree_infer(x_anon)
    end = time.perf_counter()
    print(f'Total execution time = {datetime.timedelta(seconds=int(end-begin))}')