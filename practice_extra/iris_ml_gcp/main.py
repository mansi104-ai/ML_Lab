from waitress import serve
from deploy import app
import multiprocessing

if __name__ == "__main__":
    #get cpu count
    num_cpus = multiprocessing.cpu_count()

    thread_per_worker = max(1,num_cpus-1)

    print("Threads: ",thread_per_worker)
    print('Server started')

    serve(
        # app,
        # host = '0.0.0.0',
        # port = 8080,
        threads = thread_per_worker
    )