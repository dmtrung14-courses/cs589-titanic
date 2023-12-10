import time
import datetime
from sklearn.model_selection import train_test_split

from modules.preprocessing import load_data
from modules import kernel_model, neural_model, tree_model

if __name__ == "__main__":
    x, y = load_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    begin = time.perf_counter()
    print("Begin training...")
    print("Kernel model")
    kernel_model(x_train, x_test, y_train, y_test)
    print("-----------------\nNeural model")
    neural_model(x_train, x_test, y_train, y_test)
    print("-----------------\nTree model")
    tree_model(x_train, x_test, y_train, y_test)
    print("-----------------")
    end = time.perf_counter()
    print(f'Total execution time = {datetime.timedelta(seconds=int(end-begin))}')